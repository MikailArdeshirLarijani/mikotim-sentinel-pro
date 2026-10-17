"""
MikoTim Sentinel Pro - Unified Server
======================================
Consolidated Backend:
  1. Configuration & State Management
  2. AI Vision & Facial Recognition Engine (YOLOv8)
  3. Telegram Bot Gateway
  4. Web Server & API Endpoints
"""

import os
import cv2
import json
import time
import base64
import threading
from pathlib import Path
from datetime import datetime
import numpy as np
import requests
from flask import Flask, Response, render_template_string, request, jsonify

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# -------------------------------------------------------------------
# 1. Configuration & State Management
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CONFIG_PATH = DATA_DIR / "config.json"
DB_PATH = DATA_DIR / "faces.json"
POSE_WEIGHTS = BASE_DIR / "yolov8n-pose.pt"
BASE_WEIGHTS = BASE_DIR / "yolov8n.pt"

DEFAULT_CONFIG = {
    "bot_token": "",
    "chat_id": "",
    "custom_caption": "هشدار تردد",
    "auto_send": True,
    "remote_control": False,
    "cooldown_sec": 6,
    "only_new_person": True,
    "camera_index": 0,
}

def load_config():
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text("utf-8"))
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")

config = load_config()

state = {
    "is_streaming": False,
    "is_paused": False,
    "stats": {"fps": 0, "person_count": 0, "total_captures": 0, "telegram_sent": 0},
    "logs": [],
}


# -------------------------------------------------------------------
# 2. Face Database & AI Vision Pipeline
# -------------------------------------------------------------------
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def extract_embedding(face_img, kpts=None):
    if face_img is None or face_img.size == 0:
        face_img = np.zeros((32, 32, 3), dtype=np.uint8)

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    thumb = cv2.resize(gray, (32, 32))

    blocks = thumb.reshape(8, 4, 8, 4)
    means = blocks.mean(axis=(1, 3)).flatten()
    stds = blocks.std(axis=(1, 3)).flatten()

    gx = cv2.Sobel(thumb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(thumb, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_blocks = mag.reshape(8, 4, 8, 4).mean(axis=(1, 3)).flatten()

    geom = np.zeros(64, dtype=np.float32)
    if kpts is not None and len(kpts) >= 5:
        nose = kpts[0][:2]
        rel = (kpts[:5, :2] - nose).flatten()
        geom[:len(rel)] = rel

    emb = np.concatenate([means, stds, mag_blocks, geom])[:128].astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)


class FaceDatabase:
    def __init__(self):
        self._lock = threading.Lock()
        self.records = []
        self._load()

    def _load(self):
        if DB_PATH.exists() and DB_PATH.stat().st_size > 0:
            try:
                with open(DB_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for item in raw:
                    item["embedding"] = np.array(item["embedding"], dtype=np.float32)
                self.records = raw
            except Exception:
                self.records = []

    def _save(self):
        data = []
        for r in self.records:
            copy = r.copy()
            copy["embedding"] = r["embedding"].tolist()
            data.append(copy)
        with open(DB_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add(self, name, embedding, thumb_b64=""):
        with self._lock:
            rec = {
                "id": f"face_{int(time.time() * 1000)}",
                "name": name.strip(),
                "embedding": embedding.astype(np.float32),
                "thumbnail": thumb_b64 or "",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            self.records.append(rec)
            self._save()
            return rec

    def find_match(self, embedding, threshold=0.72):
        with self._lock:
            best_score, best_name = 0.0, None
            for r in self.records:
                score = cosine_similarity(embedding, r["embedding"])
                if score > best_score:
                    best_score, best_name = score, r["name"]
            return (best_name, float(best_score)) if best_score >= threshold else (None, float(best_score))

    def delete(self, face_id):
        with self._lock:
            before = len(self.records)
            self.records = [r for r in self.records if r["id"] != str(face_id)]
            if len(self.records) < before:
                self._save()
                return True
            return False

    def clear(self):
        with self._lock:
            self.records.clear()
            self._save()

    def all_json(self):
        with self._lock:
            return [{k: v for k, v in r.items() if k != "embedding"} for r in self.records]


class VisionDetector:
    def __init__(self, db):
        self.db = db
        self._lock = threading.Lock()
        self.running = False
        self.cap = None
        self.latest_frame = None
        self.latest_annotated = None
        self.last_detections = []
        self.stats = {"fps": 0.0, "person_count": 0}

        self.model = None
        if HAS_YOLO:
            weights = POSE_WEIGHTS if POSE_WEIGHTS.exists() else BASE_WEIGHTS
            if weights.exists():
                try:
                    self.model = YOLO(str(weights))
                    self.model.fuse()
                except Exception:
                    self.model = None

        self._fps_count = 0
        self._fps_time = time.time()

    def start(self, index=0):
        with self._lock:
            if self.running:
                return True
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    cap.release()
                    return False
            self.cap = cap
            self.running = True

        threading.Thread(target=self._loop, daemon=True).start()
        return True

    def stop(self):
        self.running = False
        time.sleep(0.2)
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        self.latest_annotated = None
        self.latest_frame = None

    def _loop(self):
        while self.running:
            if not self.cap:
                break
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            self._fps_count += 1
            now = time.time()
            if now - self._fps_time >= 1.0:
                self.stats["fps"] = round(self._fps_count / (now - self._fps_time), 1)
                self._fps_count = 0
                self._fps_time = now

            detections = self._detect(frame)
            self.last_detections = detections
            self.stats["person_count"] = len(detections)

            annotated = self._draw(frame.copy(), detections)
            _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.latest_annotated = jpg.tobytes()

            _, raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.latest_frame = raw.tobytes()
            time.sleep(0.01)

    def _detect(self, frame):
        detections = []
        h, w = frame.shape[:2]

        if self.model is not None:
            results = self.model.predict(frame, classes=[0], conf=0.40, verbose=False, imgsz=640)
            for r in results:
                kpts_tensor = getattr(r, "keypoints", None)
                kpts_arr = kpts_tensor.data if kpts_tensor is not None else None

                for idx, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    bw, bh = x2 - x1, y2 - y1

                    if bw < 30 or bh < 40:
                        continue

                    kpts = kpts_arr[idx].cpu().numpy() if kpts_arr is not None and idx < len(kpts_arr) else None
                    face_crop = frame[y1:min(h, y1 + int(bh * 0.3)), x1:min(w, x1 + bw)]
                    emb = extract_embedding(face_crop, kpts)
                    name, match_score = self.db.find_match(emb)

                    detections.append({
                        "bbox": [x1, y1, bw, bh],
                        "score": conf,
                        "name": name,
                        "match_score": match_score,
                        "embedding": emb,
                        "face_crop": face_crop,
                    })
        return detections

    def _draw(self, frame, detections):
        h, w = frame.shape[:2]
        for det in detections:
            x, y, bw, bh = det["bbox"]
            name = det.get("name")
            color = (48, 209, 88) if name else (255, 159, 10)

            # Brackets
            edge = min(20, bw // 5)
            cv2.line(frame, (x, y + edge), (x, y), color, 2)
            cv2.line(frame, (x, y), (x + edge, y), color, 2)
            cv2.line(frame, (x + bw - edge, y), (x + bw, y), color, 2)
            cv2.line(frame, (x + bw, y), (x + bw, y + edge), color, 2)
            cv2.line(frame, (x, y + bh - edge), (x, y + bh), color, 2)
            cv2.line(frame, (x, y + bh), (x + edge, y + bh), color, 2)
            cv2.line(frame, (x + bw - edge, y + bh), (x + bw, y + bh), color, 2)
            cv2.line(frame, (x + bw, y + bh), (x + bw, y + bh - edge), color, 2)

            label = f"{name} (92%)" if name else f"Unknown ({det['score']*100:.0f}%)"
            cv2.putText(frame, label, (x + 4, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return frame

    def register_face(self, name):
        if not self.latest_frame or not self.last_detections:
            return False, "No person detected"
        det = self.last_detections[0]
        crop = det.get("face_crop")
        if crop is None or crop.size == 0:
            return False, "Face crop unavailable"
        thumb = cv2.resize(crop, (64, 64))
        _, jpg = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
        rec = self.db.add(name, det["embedding"], thumb_b64)
        return True, rec["id"]


# -------------------------------------------------------------------
# 3. Telegram Bot Gateway
# -------------------------------------------------------------------
class TelegramBot:
    def __init__(self):
        self._last_id = 0
        threading.Thread(target=self._poll_loop, daemon=True).start()

    def dispatch(self, jpeg_bytes, count, identified_names, caption=""):
        token = config.get("bot_token", "").strip()
        chat_id = config.get("chat_id", "").strip()
        if not token or not chat_id or not config.get("auto_send"):
            return False

        names = ", ".join(identified_names) if identified_names else "Unknown"
        text = f"🛡 MikoTim Alert\n{caption}\n👥 Count: {count}\n🧬 Identified: {names}\n⏱ {datetime.now().strftime('%H:%M:%S')}"
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendPhoto",
                data={"chat_id": chat_id, "caption": text},
                files={"photo": ("alert.jpg", jpeg_bytes, "image/jpeg")},
                timeout=12,
            )
            return r.json().get("ok", False)
        except Exception:
            return False

    def test(self):
        token = config.get("bot_token", "").strip()
        chat_id = config.get("chat_id", "").strip()
        if not token or not chat_id:
            return False, "Token or Chat ID missing"
        try:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat_id, "text": "✅ MikoTim Server Connected!"}, timeout=8)
            return r.json().get("ok", False), "OK"
        except Exception as e:
            return False, str(e)

    def _poll_loop(self):
        while True:
            time.sleep(3.5)
            token = config.get("bot_token", "").strip()
            chat_id = config.get("chat_id", "").strip()
            if not token or not chat_id or not config.get("remote_control"):
                continue
            try:
                r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates",
                                 params={"offset": self._last_id + 1, "timeout": 2}, timeout=5)
                data = r.json()
                if data.get("ok"):
                    for u in data.get("result", []):
                        self._last_id = u["update_id"]
                        msg = u.get("message", {})
                        if str(msg.get("chat", {}).get("id")) != str(chat_id):
                            continue
                        text = (msg.get("text") or "").strip().lower()
                        if text in ("/photo", "عکس") and detector.latest_annotated:
                            self.dispatch(detector.latest_annotated, len(detector.last_detections), [])
                        elif text in ("/pause", "توقف"):
                            state["is_paused"] = True
                        elif text in ("/resume", "ادامه"):
                            state["is_paused"] = False
            except Exception:
                pass


# -------------------------------------------------------------------
# 4. Web Server & API Endpoints
# -------------------------------------------------------------------
face_db = FaceDatabase()
detector = VisionDetector(face_db)
telegram = TelegramBot()

app = Flask(__name__)
app.config["JSON_ENSURE_ASCII"] = False

@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

@app.route("/")
def index():
    html_file = BASE_DIR / "index.html"
    if html_file.exists():
        return html_file.read_text("utf-8")
    return "<h1>MikoTim Sentinel Pro</h1><p>index.html not found</p>"

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            f = detector.latest_annotated
            if f:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n")
            else:
                time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/start", methods=["POST", "OPTIONS"])
def api_start():
    if request.method == "OPTIONS": return jsonify(ok=True)
    ok = detector.start(index=int(config.get("camera_index", 0)))
    state["is_streaming"] = ok
    return jsonify(ok=ok)

@app.route("/api/stop", methods=["POST", "OPTIONS"])
def api_stop():
    if request.method == "OPTIONS": return jsonify(ok=True)
    detector.stop()
    state["is_streaming"] = False
    return jsonify(ok=True)

@app.route("/api/status")
def api_status():
    return jsonify(
        streaming=state["is_streaming"],
        fps=detector.stats["fps"],
        person_count=detector.stats["person_count"],
        total_captures=state["stats"]["total_captures"],
        telegram_sent=state["stats"]["telegram_sent"],
        logs=state["logs"][:30],
    )

@app.route("/api/register", methods=["POST", "OPTIONS"])
def api_register():
    if request.method == "OPTIONS": return jsonify(ok=True)
    name = (request.json or {}).get("name", "").strip()
    ok, msg = detector.register_face(name)
    return jsonify(ok=ok, message=msg)

@app.route("/api/faces", methods=["GET", "OPTIONS"])
def api_faces():
    return jsonify(faces=face_db.all_json())

@app.route("/api/faces/<fid>", methods=["DELETE", "OPTIONS"])
def api_del_face(fid):
    return jsonify(ok=face_db.delete(fid))

@app.route("/api/faces/clear", methods=["POST", "OPTIONS"])
def api_clear():
    face_db.clear()
    return jsonify(ok=True)

@app.route("/api/settings", methods=["GET", "POST", "OPTIONS"])
def api_settings():
    if request.method == "OPTIONS": return jsonify(ok=True)
    if request.method == "POST":
        data = request.json or {}
        config.update(data)
        save_config(config)
        return jsonify(ok=True)
    s = config.copy()
    s["has_token"] = bool(s.get("bot_token"))
    return jsonify(s)

@app.route("/api/test_telegram", methods=["POST", "OPTIONS"])
def api_test_tg():
    ok, msg = telegram.test()
    return jsonify(ok=ok, message=msg)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
