from pathlib import Path
import os
import json
import time
import base64
import threading
from datetime import datetime

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

DB_DIR = Path(__file__).parent / "faces"
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "faces_db.json"

POSE_WEIGHTS = Path(__file__).parent / "yolov8n-pose.pt"
BASE_WEIGHTS = Path(__file__).parent / "yolov8n.pt"

SIMILARITY_THRESHOLD = 0.70


class FaceDatabase:
    def __init__(self, db_path=None):
        self._lock = threading.Lock()
        self.db_file = Path(db_path) if db_path else DB_PATH
        self.records = []
        self._load()

    @property
    def faces(self):
        return self.records

    def _load(self):
        if self.db_file.exists() and self.db_file.stat().st_size > 0:
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for item in raw:
                    item["embedding"] = np.array(item["embedding"], dtype=np.float32)
                self.records = raw
            except Exception:
                self.records = []
        else:
            self.records = []

    def _save(self):
        data = []
        for r in self.records:
            copy = r.copy()
            copy["embedding"] = r["embedding"].tolist()
            data.append(copy)
        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add(self, name, embedding, thumbnail_b64=""):
        with self._lock:
            record = {
                "id": f"face_{int(time.time() * 1000)}",
                "name": name.strip(),
                "embedding": embedding.astype(np.float32),
                "thumbnail": thumbnail_b64 or "",
                "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_seen": None,
            }
            self.records.append(record)
            self._save()
            return record

    def find_match(self, embedding):
        with self._lock:
            if not self.records:
                return None, 0.0

            best_score = 0.0
            best_match = None
            best_id = None

            for r in self.records:
                score = cosine_similarity(embedding, r["embedding"])
                if score > best_score:
                    best_score = score
                    best_match = r["name"]
                    best_id = r["id"]

            if best_score >= SIMILARITY_THRESHOLD:
                for r in self.records:
                    if r["id"] == best_id:
                        r["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        break
                self._save()
                return best_match, float(best_score)

            return None, float(best_score)

    def find(self, embedding):
        return self.find_match(embedding)

    def delete(self, face_id):
        if isinstance(face_id, dict):
            face_id = face_id.get("id", "")
        with self._lock:
            prev_len = len(self.records)
            self.records = [r for r in self.records if r["id"] != str(face_id)]
            if len(self.records) < prev_len:
                self._save()
                return True
            return False

    def clear(self):
        with self._lock:
            self.records.clear()
            self._save()

    def all_records_json(self):
        with self._lock:
            return [{k: v for k, v in r.items() if k != "embedding"} for r in self.records]


class MikoDetector:
    def __init__(self, db_path=None):
        self.db = FaceDatabase(db_path=db_path)
        self._lock = threading.Lock()

        self.running = False
        self.cap = None
        self.latest_frame_jpg = None
        self.latest_annotated_jpg = None
        self.last_detections = []

        self.stats = {
            "total_captures": 0,
            "telegram_sent": 0,
            "fps": 0.0,
            "person_count": 0,
            "motion_level": 0,
        }

        self.model = None
        if HAS_YOLO:
            weights = POSE_WEIGHTS if POSE_WEIGHTS.exists() else BASE_WEIGHTS
            try:
                self.model = YOLO(str(weights))
                self.model.fuse()
            except Exception as e:
                print(f"[Detector] Failed to load YOLO weights ({e}), using motion fallback.")
                self.model = None

        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)
        self._fps_count = 0
        self._fps_timestamp = time.time()

    def start(self, device_index=0, width=1280, height=720):
        with self._lock:
            if self.running:
                return True

            cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                cap.release()
                return False

            self.cap = cap
            self.running = True

        t = threading.Thread(target=self._worker, daemon=True)
        t.start()
        return True

    def stop(self):
        self.running = False
        time.sleep(0.2)
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None

        self.latest_frame_jpg = None
        self.latest_annotated_jpg = None
        self.last_detections = []
        self.stats["person_count"] = 0
        self.stats["fps"] = 0.0

    def _worker(self):
        while self.running:
            if not self.cap:
                break

            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            # Update FPS counter
            self._fps_count += 1
            now = time.time()
            elapsed = now - self._fps_timestamp
            if elapsed >= 1.0:
                self.stats["fps"] = round(self._fps_count / elapsed, 1)
                self._fps_count = 0
                self._fps_timestamp = now

            detections = self._detect(frame)
            self.last_detections = detections
            self.stats["person_count"] = len(detections)

            annotated = self._render(frame.copy(), detections)

            _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.latest_annotated_jpg = jpg.tobytes()

            _, raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.latest_frame_jpg = raw.tobytes()

            time.sleep(0.01)

    def _detect(self, frame):
        detections = []
        h, w = frame.shape[:2]

        if self.model is not None:
            results = self.model.predict(
                frame,
                classes=[0],
                conf=0.40,
                iou=0.45,
                verbose=False,
                imgsz=640,
            )

            for r in results:
                kpts_tensor = getattr(r, "keypoints", None)
                kpts_arr = kpts_tensor.data if kpts_tensor is not None else None

                for idx, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    bw, bh = x2 - x1, y2 - y1

                    if bw < 25 or bh < 35:
                        continue

                    kpts = kpts_arr[idx].cpu().numpy() if kpts_arr is not None and idx < len(kpts_arr) else None
                    face_crop = self._crop_face(frame, x1, y1, bw, bh, kpts)
                    emb = extract_embedding(face_crop, kpts)

                    name, match_score = self.db.find_match(emb)
                    detections.append({
                        "bbox": [x1, y1, bw, bh],
                        "score": conf,
                        "name": name,
                        "match_score": match_score,
                        "id": f"det_{x1}_{y1}_{int(conf * 100)}",
                        "embedding": emb,
                        "face_crop": face_crop,
                    })

            self.stats["motion_level"] = min(100, len(detections) * 25)
        else:
            fg = self._bg_subtractor.apply(frame)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < (w * h * 0.005):
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                if not (0.5 <= (bh / max(bw, 1)) <= 4.5):
                    continue

                crop = frame[y:y + int(bh * 0.3), x:x + bw]
                emb = extract_embedding(crop)
                name, match_score = self.db.find_match(emb)

                detections.append({
                    "bbox": [x, y, bw, bh],
                    "score": min(0.95, area / (w * h) * 10),
                    "name": name,
                    "match_score": match_score,
                    "id": f"det_{x}_{y}",
                    "embedding": emb,
                    "face_crop": crop,
                })
            self.stats["motion_level"] = min(100, len(detections) * 30)

        return detections

    def _crop_face(self, frame, x, y, bw, bh, kpts):
        h, w = frame.shape[:2]
        if kpts is not None and len(kpts) >= 5:
            pts = kpts[:5, :2]
            valid = pts[kpts[:5, 2] > 0.3] if kpts.shape[-1] > 2 else pts
            if len(valid) >= 2:
                fx1, fy1 = np.min(valid, axis=0) - 15
                fx2, fy2 = np.max(valid, axis=0) + 15
                fx1, fy1 = max(0, int(fx1)), max(0, int(fy1))
                fx2, fy2 = min(w, int(fx2)), min(h, int(fy2))
                if fx2 > fx1 and fy2 > fy1:
                    return frame[fy1:fy2, fx1:fx2]

        top_y2 = min(h, y + max(25, int(bh * 0.28)))
        left_x = max(0, x + int(bw * 0.12))
        right_x = min(w, x + int(bw * 0.88))
        crop = frame[y:top_y2, left_x:right_x]
        return crop if crop.size > 0 else frame[y:y + bh, x:x + bw]

    def _render(self, frame, detections):
        h, w = frame.shape[:2]

        for det in detections:
            x, y, bw, bh = det["bbox"]
            x2, y2 = x + bw, y + bh
            name = det.get("name")
            score = det["score"]
            match_score = det.get("match_score", 0.0)

            is_known = name is not None
            color = (16, 185, 129) if is_known else (245, 158, 11)  # Emerald green / Amber
            edge = min(20, max(8, bw // 5))

            # Corner markings
            cv2.line(frame, (x, y + edge), (x, y), color, 2)
            cv2.line(frame, (x, y), (x + edge, y), color, 2)
            cv2.line(frame, (x2 - edge, y), (x2, y), color, 2)
            cv2.line(frame, (x2, y), (x2, y + edge), color, 2)
            cv2.line(frame, (x, y2 - edge), (x, y2), color, 2)
            cv2.line(frame, (x, y2), (x + edge, y2), color, 2)
            cv2.line(frame, (x2 - edge, y2), (x2, y2), color, 2)
            cv2.line(frame, (x2, y2), (x2, y2 - edge), color, 2)

            label = f"{name} ({match_score * 100:.0f}%)" if is_known else f"Unknown ({score * 100:.0f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(label, font, 0.52, 1)

            tag_y = y - 7 if y > 24 else y2 + th + 7
            cv2.rectangle(frame, (x, tag_y - th - 4), (x + tw + 8, tag_y + baseline), (17, 24, 39), -1)
            cv2.rectangle(frame, (x, tag_y - th - 4), (x + tw + 8, tag_y + baseline), color, 1)
            cv2.putText(frame, label, (x + 4, tag_y), font, 0.52, color, 1, cv2.LINE_AA)

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"MikoTim Sentinel Pro | {stamp}", (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 116, 139), 1, cv2.LINE_AA)
        return frame

    def register_face_from_frame(self, name):
        if not self.latest_frame_jpg:
            return False, "Camera is not active."

        frame = cv2.imdecode(np.frombuffer(self.latest_frame_jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return False, "Unable to read camera frame."

        detections = self.last_detections or self._detect(frame)
        if not detections:
            return False, "No person detected in frame."

        det = detections[0]
        emb = det.get("embedding")
        if emb is None:
            return False, "Failed to compute face embedding."

        crop = det.get("face_crop")
        if crop is None or crop.size == 0:
            crop = frame

        thumb = cv2.resize(crop, (64, 64))
        _, thumb_jpg = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(thumb_jpg.tobytes()).decode()

        rec = self.db.add(name, emb, thumb_b64)
        return True, rec["id"]

    def take_snapshot(self):
        if not self.latest_annotated_jpg:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(self.latest_annotated_jpg).decode()

    def take_raw_snapshot(self):
        return self.latest_annotated_jpg


def extract_embedding(face_img, kpts=None):
    if face_img is None or face_img.size == 0:
        face_img = np.zeros((32, 32, 3), dtype=np.uint8)

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    thumb = cv2.resize(gray, (32, 32))

    # Grid block statistics
    blocks = thumb.reshape(8, 4, 8, 4)
    means = blocks.mean(axis=(1, 3)).flatten()
    stds = blocks.std(axis=(1, 3)).flatten()

    # Structural gradients
    gx = cv2.Sobel(thumb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(thumb, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_blocks = mag.reshape(8, 4, 8, 4).mean(axis=(1, 3)).flatten()

    geom = np.zeros(64, dtype=np.float32)
    if kpts is not None and len(kpts) >= 5:
        nose = kpts[0][:2]
        rel = (kpts[:5, :2] - nose).flatten()
        geom[:len(rel)] = rel
        if kpts.shape[-1] > 2:
            geom[10:15] = kpts[:5, 2]
        eye_dist = np.linalg.norm(kpts[1][:2] - kpts[2][:2]) + 1e-6
        geom[15] = eye_dist

    emb = np.concatenate([means, stds, mag_blocks, geom])[:128].astype(np.float32)
    norm = np.linalg.norm(emb) + 1e-9
    return emb / norm


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# Backward compatibility aliases
_extract_face_embedding = extract_embedding
_cosine_similarity = cosine_similarity
