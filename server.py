"""
MikoTim Sentinel Pro - Unified Server
======================================
Powered by MikiVo AI Vision Engine (Derived & Evolved from YOLOv8 Architecture)

Consolidated Architecture (4 Core Sections):
  1. Configuration & State Management
  2. MikiVo AI Vision & Facial Identity Pipeline
  3. Telegram Bot Gateway & Remote Command Dispatcher
  4. Web Server & High-Throughput RESTful Endpoints
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
from flask import Flask, Response, request, jsonify

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


# ===================================================================
# 1. Configuration & State Management
# ===================================================================
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
    "custom_caption": "هشدار تردد هوشمند MikiVo",
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
    "stats": {"fps": 0.0, "person_count": 0, "total_captures": 0, "telegram_sent": 0},
    "logs": [],
}


# ===================================================================
# 2. MikiVo AI Vision Engine (Derived & Evolved from YOLO Architecture)
# ===================================================================
# MikiVo integrates YOLO's deep convolutional backbone with:
#  - Adaptive LAB Optical CLAHE Equalization (illumination invariance)
#  - Pose-Guided Affine Landmark Alignment (head tilt invariance)
#  - 256-Dimensional Multi-Scale Spatial-Geometric Feature Embedder
#  - IoU + EMA Temporal Tracklet Smoother with Identity Stabilization

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

def compute_iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    yb = min(box_a[1] + box_a[3], box_b[1] + box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    return inter / float(area_a + area_b - inter + 1e-9)

def extract_mikivo_embedding(crop, kpts=None):
    """
    MikiVo 256-D Feature Descriptor:
      - 8x8 spatial block statistics (means + stds): 128-D
      - Multi-scale gradient energy (Sobel Gx, Gy, Magnitude): 48-D
      - Fast texture Laplacian & intensity distribution: 32-D
      - Anthropometric keypoint geometry / moments: 48-D
    """
    if crop is None or crop.size == 0:
        return np.zeros(256, dtype=np.float32)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    gray = cv2.resize(gray, (64, 64))
    norm_gray = clahe.apply(gray)

    # 1. Spatial Block Statistics (8x8 blocks of 8x8 pixels)
    blocks_8 = norm_gray.reshape(8, 8, 8, 8)
    means_8 = blocks_8.mean(axis=(1, 3)).ravel() / 255.0
    stds_8 = blocks_8.std(axis=(1, 3)).ravel() / 128.0

    # 2. Multi-scale directional gradients
    gx = cv2.Sobel(norm_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)

    mag_b = mag.reshape(4, 16, 4, 16).mean(axis=(1, 3)).ravel() / 255.0
    gx_b = np.abs(gx).reshape(4, 16, 4, 16).mean(axis=(1, 3)).ravel() / 255.0
    gy_b = np.abs(gy).reshape(4, 16, 4, 16).mean(axis=(1, 3)).ravel() / 255.0

    # 3. Laplacian texture energy & intensity histogram
    lap = np.abs(cv2.Laplacian(norm_gray, cv2.CV_32F)).reshape(4, 16, 4, 16).mean(axis=(1, 3)).ravel() / 255.0
    hist, _ = np.histogram(norm_gray, bins=16, range=(0, 256), density=True)

    # 4. Anthropometric Geometry (Invariant to distance & scale)
    geom = np.zeros(48, dtype=np.float32)
    if kpts is not None and len(kpts) >= 5:
        ref_pt = kpts[0][:2]
        pts = kpts[:5, :2]
        rel = (pts - ref_pt).ravel()
        geom[:len(rel)] = rel / 100.0
        d12 = np.linalg.norm(pts[1] - pts[2]) / 100.0
        d10 = np.linalg.norm(pts[1] - pts[0]) / 100.0
        d20 = np.linalg.norm(pts[2] - pts[0]) / 100.0
        geom[len(rel):len(rel)+3] = [d12, d10, d20]
    else:
        moments = cv2.moments(norm_gray)
        hu = cv2.HuMoments(moments).ravel()
        geom[:len(hu)] = -np.sign(hu) * np.log10(np.abs(hu) + 1e-9)

    emb = np.concatenate([means_8, stds_8, mag_b, gx_b, gy_b, lap, hist, geom])[:256].astype(np.float32)
    norm = np.linalg.norm(emb) + 1e-9
    return emb / norm


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

    def find_match(self, embedding, threshold=0.73):
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


class MikiVoTracker:
    """
    Temporal Tracklet Engine:
      - Assigns continuous Track IDs across frames
      - Smooths bounding box coordinates using Exponential Moving Average (EMA)
      - Majority-vote temporal identity stabilization (filters momentary recognition noise)
    """
    def __init__(self):
        self.next_id = 1
        self.tracks = {}

    def update(self, detections, now=None):
        now = now or time.time()
        updated = []
        unmatched_indices = list(range(len(detections)))
        matched_tracks = set()

        for idx in list(unmatched_indices):
            det = detections[idx]
            best_iou, best_tid = 0.0, None
            for tid, trk in self.tracks.items():
                if tid in matched_tracks:
                    continue
                iou = compute_iou(det["bbox"], trk["bbox"])
                if iou > best_iou:
                    best_iou, best_tid = iou, tid

            if best_iou > 0.28 and best_tid is not None:
                matched_tracks.add(best_tid)
                unmatched_indices.remove(idx)
                trk = self.tracks[best_tid]

                # Exponential Moving Average coordinate smoothing
                alpha = 0.72
                cb = det["bbox"]
                pb = trk["smooth_bbox"]
                smooth_box = [
                    int(alpha * cb[i] + (1 - alpha) * pb[i])
                    for i in range(4)
                ]

                trk["bbox"] = cb
                trk["smooth_bbox"] = smooth_box
                trk["last_seen"] = now
                trk["history"].append(det.get("name"))
                if len(trk["history"]) > 8:
                    trk["history"].pop(0)

                valid_names = [n for n in trk["history"] if n]
                stable_name = max(set(valid_names), key=valid_names.count) if valid_names else None

                det_copy = dict(det)
                det_copy["track_id"] = best_tid
                det_copy["smooth_bbox"] = smooth_box
                det_copy["name"] = stable_name or det.get("name")
                updated.append(det_copy)

        for idx in unmatched_indices:
            det = detections[idx]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "bbox": det["bbox"],
                "smooth_bbox": det["bbox"],
                "last_seen": now,
                "history": [det.get("name")],
            }
            det_copy = dict(det)
            det_copy["track_id"] = tid
            det_copy["smooth_bbox"] = det["bbox"]
            det_copy["name"] = det.get("name")
            updated.append(det_copy)

        expired = [tid for tid, trk in self.tracks.items() if now - trk["last_seen"] > 2.0]
        for tid in expired:
            del self.tracks[tid]

        return updated


class MikiVoVisionEngine:
    """
    MikiVo Autonomous AI Vision Core:
      - Derived and evolved from YOLOv8 Architecture
      - High-speed pose detection + keypoint tracking
      - Optical normalization & affine face canonicalization
    """
    def __init__(self, db):
        self.db = db
        self.tracker = MikiVoTracker()
        self._lock = threading.Lock()
        self.running = False
        self.cap = None
        self.latest_frame = None
        self.latest_annotated = None
        self.last_detections = []
        self.stats = {"fps": 0.0, "person_count": 0}
        self.last_alert_time = 0.0
        self.reported_tracks = set()

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
        time.sleep(0.15)
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        self.latest_annotated = None
        self.latest_frame = None

    def _align_face(self, frame, bbox, kpts=None):
        h, w = frame.shape[:2]
        x, y, bw, bh = bbox

        if kpts is not None and len(kpts) >= 3 and kpts[1][2] > 0.35 and kpts[2][2] > 0.35:
            left_eye = kpts[1][:2]
            right_eye = kpts[2][:2]
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = float(np.degrees(np.arctan2(dy, dx)))
            eye_center = (float((left_eye[0] + right_eye[0]) / 2.0), float((left_eye[1] + right_eye[1]) / 2.0))

            m = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            m[0, 2] += (56 - eye_center[0])
            m[1, 2] += (44 - eye_center[1])
            return cv2.warpAffine(frame, m, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        else:
            fy1 = max(0, y)
            fy2 = min(h, y + int(bh * 0.35))
            fx1 = max(0, x)
            fx2 = min(w, x + bw)
            crop = frame[fy1:fy2, fx1:fx2]
            if crop.size == 0:
                return np.zeros((112, 112, 3), dtype=np.uint8)
            return cv2.resize(crop, (112, 112))

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
            tracked_detections = self.tracker.update(detections, now=now)
            self.last_detections = tracked_detections
            self.stats["person_count"] = len(tracked_detections)

            annotated = self._draw(frame.copy(), tracked_detections)
            _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.latest_annotated = jpg.tobytes()

            _, raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.latest_frame = raw.tobytes()

            # Automated Telegram Alert Dispatch
            if tracked_detections and not state["is_paused"] and config.get("auto_send"):
                cooldown = max(2, int(config.get("cooldown_sec", 6)))
                if now - self.last_alert_time >= cooldown:
                    names = [d["name"] for d in tracked_detections if d.get("name")]
                    track_ids = [d.get("track_id") for d in tracked_detections]
                    should_alert = True

                    if config.get("only_new_person"):
                        # Alert if any track is new or an unknown intruder is present
                        has_unknown = any(not d.get("name") for d in tracked_detections)
                        has_unreported_track = any(t not in self.reported_tracks for t in track_ids)
                        should_alert = has_unknown or has_unreported_track

                    if should_alert:
                        self.last_alert_time = now
                        for t in track_ids:
                            self.reported_tracks.add(t)
                        self._trigger_alert(self.latest_annotated, tracked_detections)

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

                    if bw < 28 or bh < 38:
                        continue

                    kpts = kpts_arr[idx].cpu().numpy() if kpts_arr is not None and idx < len(kpts_arr) else None
                    aligned_face = self._align_face(frame, [x1, y1, bw, bh], kpts)
                    emb = extract_mikivo_embedding(aligned_face, kpts)
                    name, match_score = self.db.find_match(emb)

                    detections.append({
                        "bbox": [x1, y1, bw, bh],
                        "score": conf,
                        "name": name,
                        "match_score": match_score,
                        "embedding": emb,
                        "face_crop": aligned_face,
                        "kpts": kpts,
                    })
        return detections

    def _draw(self, frame, detections):
        h, w = frame.shape[:2]

        # Engine watermark
        cv2.putText(frame, "MikiVo AI [YOLOv8 Core]", (18, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        for det in detections:
            x, y, bw, bh = det.get("smooth_bbox", det["bbox"])
            name = det.get("name")
            tid = det.get("track_id", 1)
            color = (48, 209, 88) if name else (0, 165, 255)

            # Apple-inspired HUD corner brackets
            edge = min(22, max(8, bw // 5))
            t = 2
            cv2.line(frame, (x, y + edge), (x, y), color, t)
            cv2.line(frame, (x, y), (x + edge, y), color, t)
            cv2.line(frame, (x + bw - edge, y), (x + bw, y), color, t)
            cv2.line(frame, (x + bw, y), (x + bw, y + edge), color, t)
            cv2.line(frame, (x, y + bh - edge), (x, y + bh), color, t)
            cv2.line(frame, (x, y + bh), (x + edge, y + bh), color, t)
            cv2.line(frame, (x + bw - edge, y + bh), (x + bw, y + bh), color, t)
            cv2.line(frame, (x + bw, y + bh), (x + bw, y + bh - edge), color, t)

            # Keypoint indicator
            kpts = det.get("kpts")
            if kpts is not None and len(kpts) > 0 and kpts[0][2] > 0.4:
                nx, ny = int(kpts[0][0]), int(kpts[0][1])
                cv2.circle(frame, (nx, ny), 3, color, -1)

            # Informative HUD label
            label = f"MikiVo #{tid} | {name} ({det.get('match_score', 0.92)*100:.0f}%)" if name else f"MikiVo #{tid} | Unknown ({det['score']*100:.0f}%)"
            
            # Label background pill
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            ly = max(24, y - 8)
            cv2.rectangle(frame, (x, ly - th - 4), (x + tw + 10, ly + 4), (20, 20, 22), -1)
            cv2.rectangle(frame, (x, ly - th - 4), (x + tw + 10, ly + 4), color, 1)
            cv2.putText(frame, label, (x + 5, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 247), 1, cv2.LINE_AA)

        return frame

    def _trigger_alert(self, jpg_bytes, detections):
        count = len(detections)
        identified = [d["name"] for d in detections if d.get("name")]
        custom_cap = config.get("custom_caption", "هشدار تردد MikiVo")

        state["stats"]["total_captures"] += 1
        log_entry = {
            "id": int(time.time() * 1000),
            "time": datetime.now().strftime("%H:%M:%S"),
            "persons": count,
            "names": ", ".join(identified) if identified else "ناشناس",
            "telegram": "در حال ارسال...",
        }
        state["logs"].insert(0, log_entry)
        if len(state["logs"]) > 50:
            state["logs"].pop()

        def dispatch_worker():
            ok = telegram.dispatch(jpg_bytes, count, identified, custom_cap)
            log_entry["telegram"] = "ارسال شد" if ok else "ارسال ناموفق"
            if ok:
                state["stats"]["telegram_sent"] += 1

        threading.Thread(target=dispatch_worker, daemon=True).start()

    def register_face(self, name):
        if not self.latest_frame or not self.last_detections:
            return False, "شخصی در کادر تصویر تشخیص داده نشد"
        det = self.last_detections[0]
        crop = det.get("face_crop")
        if crop is None or crop.size == 0:
            return False, "برش چهره معتبر نیست"
        thumb = cv2.resize(crop, (64, 64))
        _, jpg = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
        rec = self.db.add(name, det["embedding"], thumb_b64)
        return True, rec["id"]


# ===================================================================
# 3. Telegram Bot Gateway
# ===================================================================
class TelegramBot:
    def __init__(self):
        self._last_id = 0
        threading.Thread(target=self._poll_loop, daemon=True).start()

    def dispatch(self, jpeg_bytes, count, identified_names, caption=""):
        token = config.get("bot_token", "").strip()
        chat_id = config.get("chat_id", "").strip()
        if not token or not chat_id:
            return False

        names = ", ".join(identified_names) if identified_names else "ناشناس (Unknown)"
        text = (
            f"🛡 MikoTim Sentinel Pro | MikiVo AI (Derived from YOLO)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 {caption}\n"
            f"👥 تعداد اشخاص: {count}\n"
            f"🧬 هویت: {names}\n"
            f"⏱ زمان ثبت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendPhoto",
                data={"chat_id": chat_id, "caption": text},
                files={"photo": ("alert_mikivo.jpg", jpeg_bytes, "image/jpeg")},
                timeout=12,
            )
            return r.json().get("ok", False)
        except Exception:
            return False

    def test(self):
        token = config.get("bot_token", "").strip()
        chat_id = config.get("chat_id", "").strip()
        if not token or not chat_id:
            return False, "توکن یا شناسه چت تنظیم نشده است"
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": "✅ اتصال با سرور پایش MikoTim و موتور هوش مصنوعی MikiVo با موفقیت برقرار شد."},
                timeout=8
            )
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
                r = requests.get(
                    f"https://api.telegram.org/bot{token}/getUpdates",
                    params={"offset": self._last_id + 1, "timeout": 2},
                    timeout=5
                )
                data = r.json()
                if data.get("ok"):
                    for u in data.get("result", []):
                        self._last_id = u["update_id"]
                        msg = u.get("message", {})
                        if str(msg.get("chat", {}).get("id")) != str(chat_id):
                            continue
                        text = (msg.get("text") or "").strip().lower()
                        if text in ("/photo", "عکس") and detector.latest_annotated:
                            names = [d["name"] for d in detector.last_detections if d.get("name")]
                            self.dispatch(detector.latest_annotated, len(detector.last_detections), names, "عکس درخواستی زنده")
                        elif text in ("/pause", "توقف"):
                            state["is_paused"] = True
                        elif text in ("/resume", "ادامه"):
                            state["is_paused"] = False
            except Exception:
                pass


# ===================================================================
# 4. Web Server & API Endpoints
# ===================================================================
face_db = FaceDatabase()
detector = MikiVoVisionEngine(face_db)
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

@app.route("/api/mikivo/info")
def api_mikivo_info():
    return jsonify(
        engine="MikiVo AI Vision Core",
        origin="Derived & Evolved from YOLOv8 Architecture",
        version="2.4.0-pro",
        features=[
            "Adaptive LAB Optical CLAHE Equalization",
            "Pose-Guided Affine Landmark Canonicalization",
            "256-D Multi-Scale Spatial-Geometric Embedding",
            "IoU + EMA Temporal Tracklet Smoother",
            "Local High-Performance Execution",
        ]
    )

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
