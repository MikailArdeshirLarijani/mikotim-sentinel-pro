"""
MikoTim Sentinel Pro - Unified Server
======================================
Powered by MikiVo AI Vision Engine (Derived & Evolved from YOLOv8 Architecture)

Consolidated Architecture (4 Core Sections):
  1. Configuration & State Management (ROI, Tripwire, Media Selector, Loitering)
  2. MikiVo AI Vision Pipeline (Pose, Alignment, 256-D Descriptor, Tracklets, Video Clip Buffer)
  3. Telegram Bot Gateway (Photo & MP4 Video Dispatch, Remote Commands, Daily Summary)
  4. Web Server & High-Throughput RESTful Endpoints
"""

import os
import cv2
import json
import time
import base64
import tempfile
import threading
import collections
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
    "alert_media": "photo",          # "photo" (عکس) or "video" (ویدیو MP4)
    "video_duration_sec": 4,         # Duration of MP4 alert clip
    "roi_enabled": False,            # Virtual ROI security zone
    "roi_polygon": [],               # [[x, y], ...] normalized 0.0-1.0
    "tripwire_enabled": False,       # Directional line crossing
    "tripwire_y": 0.55,              # Normalized Y coordinate for tripwire line
    "loitering_enabled": True,       # Loitering detection flag
    "loitering_sec": 10,             # Loitering time threshold in seconds
    "audio_alarm": False,            # Local sound/speech alert toggle
}

def load_config():
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_PATH.exists():
        try:
            stored = json.loads(CONFIG_PATH.read_text("utf-8"))
            cfg.update(stored)
        except Exception:
            pass
    return cfg

def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")

config = load_config()

state = {
    "is_streaming": False,
    "is_paused": False,
    "stats": {
        "fps": 0.0,
        "person_count": 0,
        "total_captures": 0,
        "telegram_sent": 0,
        "in_count": 0,
        "out_count": 0,
        "loitering_count": 0,
    },
    "hourly_traffic": [0] * 24,
    "logs": [],
}


# ===================================================================
# 2. MikiVo AI Vision Engine (Derived & Evolved from YOLO Architecture)
# ===================================================================
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
    MikiVo 256-D Multi-Scale Hybrid Feature Descriptor:
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

    # 1. Spatial Block Statistics (8x8 blocks)
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

    # 4. Anthropometric Geometry
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
      - Continuous Track IDs across video frames
      - EMA coordinate smoothing to eliminate jitter
      - Temporal majority-vote identity stabilization
      - Directional Tripwire crossing (IN / OUT)
      - Unauthorized Loitering detection timer
    """
    def __init__(self):
        self.next_id = 1
        self.tracks = {}

    def update(self, detections, frame_shape, now=None):
        now = now or time.time()
        fh, fw = frame_shape[:2]
        updated = []
        unmatched_indices = list(range(len(detections)))
        matched_tracks = set()

        line_y = float(config.get("tripwire_y", 0.55)) * fh
        loitering_limit = float(config.get("loitering_sec", 10))

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

                # Feet Y coordinate
                curr_feet_y = smooth_box[1] + smooth_box[3]
                prev_feet_y = trk["feet_y"]

                # Directional Tripwire Crossing
                direction = None
                if config.get("tripwire_enabled") and not trk["tripwire_crossed"]:
                    if prev_feet_y < line_y <= curr_feet_y:
                        direction = "IN"
                        trk["tripwire_crossed"] = True
                        state["stats"]["in_count"] += 1
                    elif prev_feet_y > line_y >= curr_feet_y:
                        direction = "OUT"
                        trk["tripwire_crossed"] = True
                        state["stats"]["out_count"] += 1

                # Loitering Detection
                duration = now - trk["first_seen"]
                is_loitering = False
                if config.get("loitering_enabled") and duration >= loitering_limit:
                    is_loitering = True
                    if not trk["loitering_logged"]:
                        trk["loitering_logged"] = True
                        state["stats"]["loitering_count"] += 1

                trk["bbox"] = cb
                trk["smooth_bbox"] = smooth_box
                trk["last_seen"] = now
                trk["feet_y"] = curr_feet_y
                trk["history"].append(det.get("name"))
                if len(trk["history"]) > 8:
                    trk["history"].pop(0)

                valid_names = [n for n in trk["history"] if n]
                stable_name = max(set(valid_names), key=valid_names.count) if valid_names else None

                det_copy = dict(det)
                det_copy["track_id"] = best_tid
                det_copy["smooth_bbox"] = smooth_box
                det_copy["name"] = stable_name or det.get("name")
                det_copy["direction"] = direction
                det_copy["is_loitering"] = is_loitering
                det_copy["duration"] = round(duration, 1)
                updated.append(det_copy)

        # Allocate new tracks
        for idx in unmatched_indices:
            det = detections[idx]
            tid = self.next_id
            self.next_id += 1
            feet_y = det["bbox"][1] + det["bbox"][3]
            self.tracks[tid] = {
                "bbox": det["bbox"],
                "smooth_bbox": det["bbox"],
                "first_seen": now,
                "last_seen": now,
                "feet_y": feet_y,
                "tripwire_crossed": False,
                "loitering_logged": False,
                "history": [det.get("name")],
            }
            det_copy = dict(det)
            det_copy["track_id"] = tid
            det_copy["smooth_bbox"] = det["bbox"]
            det_copy["name"] = det.get("name")
            det_copy["direction"] = None
            det_copy["is_loitering"] = False
            det_copy["duration"] = 0.0
            updated.append(det_copy)

        # Clean expired tracks
        expired = [tid for tid, trk in self.tracks.items() if now - trk["last_seen"] > 2.0]
        for tid in expired:
            del self.tracks[tid]

        return updated


class MikiVoVisionEngine:
    """
    MikiVo Autonomous AI Vision Core:
      - YOLOv8 Neural Backbone + MikiVo Custom Preprocessing
      - Pose-Guided Affine Face Canonicalization
      - Virtual Tripwire & ROI Spatial Filtering
      - Circular Frame Buffer for Instant Video Alert Dispatch
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

        # Circular buffer for pre-incident video clip (~4 sec @ 15 FPS)
        self.frame_buffer = collections.deque(maxlen=60)
        self.latest_annotated_mat = None

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
        self.latest_annotated_mat = None

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

    def _is_in_roi(self, smooth_box, frame_shape):
        if not config.get("roi_enabled"):
            return True
        poly = config.get("roi_polygon")
        if not poly or len(poly) < 3:
            return True

        fh, fw = frame_shape[:2]
        pts = np.array([[int(p[0] * fw), int(p[1] * fh)] for p in poly], dtype=np.int32)
        # Feet center point
        feet_pt = (smooth_box[0] + smooth_box[2] // 2, smooth_box[1] + smooth_box[3])
        return cv2.pointPolygonTest(pts, feet_pt, False) >= 0

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
            tracked = self.tracker.update(detections, frame.shape, now=now)

            # Apply ROI filter if enabled
            active_detections = []
            for d in tracked:
                in_roi = self._is_in_roi(d.get("smooth_bbox", d["bbox"]), frame.shape)
                d["in_roi"] = in_roi
                if in_roi:
                    active_detections.append(d)

            self.last_detections = tracked
            self.stats["person_count"] = len(active_detections)

            annotated = self._draw(frame.copy(), tracked)
            self.latest_annotated_mat = annotated
            self.frame_buffer.append(annotated.copy())

            _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.latest_annotated = jpg.tobytes()

            _, raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.latest_frame = raw.tobytes()

            # Record hourly traffic statistic
            current_hour = datetime.now().hour
            if active_detections:
                state["hourly_traffic"][current_hour] += 1

            # Automated Telegram Alert Dispatch
            if active_detections and not state["is_paused"] and config.get("auto_send"):
                cooldown = max(2, int(config.get("cooldown_sec", 6)))
                if now - self.last_alert_time >= cooldown:
                    track_ids = [d.get("track_id") for d in active_detections]
                    should_alert = True

                    if config.get("only_new_person"):
                        has_unknown = any(not d.get("name") for d in active_detections)
                        has_unreported_track = any(t not in self.reported_tracks for t in track_ids)
                        has_loitering = any(d.get("is_loitering") for d in active_detections)
                        should_alert = has_unknown or has_unreported_track or has_loitering

                    if should_alert:
                        self.last_alert_time = now
                        for t in track_ids:
                            self.reported_tracks.add(t)
                        self._trigger_alert(active_detections)

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

        # 1. Draw Virtual ROI Zone if enabled
        if config.get("roi_enabled"):
            poly = config.get("roi_polygon")
            if poly and len(poly) >= 3:
                pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in poly], dtype=np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (255, 69, 58))
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                cv2.polylines(frame, [pts], True, (255, 69, 58), 2, cv2.LINE_AA)
                cv2.putText(frame, "RESTRICTED SECURITY ZONE", (pts[0][0] + 5, pts[0][1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 69, 58), 1, cv2.LINE_AA)

        # 2. Draw Tripwire Line if enabled
        if config.get("tripwire_enabled"):
            line_y = int(float(config.get("tripwire_y", 0.55)) * h)
            cv2.line(frame, (0, line_y), (w, line_y), (0, 215, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "TRIPWIRE LINE [ IN / OUT ]", (20, line_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

        # 3. Watermark
        cv2.putText(frame, "MikiVo AI [YOLOv8 Core]", (18, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Annotate Persons
        for det in detections:
            x, y, bw, bh = det.get("smooth_bbox", det["bbox"])
            name = det.get("name")
            tid = det.get("track_id", 1)
            is_loiter = det.get("is_loitering", False)
            in_roi = det.get("in_roi", True)

            if is_loiter:
                color = (0, 0, 255) # Warning Red
            elif not in_roi:
                color = (120, 120, 120) # Outside Zone Grey
            elif name:
                color = (48, 209, 88) # Authorized Green
            else:
                color = (0, 165, 255) # Unknown Amber

            # Apple-inspired corner brackets
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
            prefix = "LOITERING! " if is_loiter else ""
            id_txt = f"{name} ({det.get('match_score', 0.92)*100:.0f}%)" if name else f"Unknown ({det['score']*100:.0f}%)"
            label = f"MikiVo #{tid} | {prefix}{id_txt}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            ly = max(24, y - 8)
            cv2.rectangle(frame, (x, ly - th - 4), (x + tw + 10, ly + 4), (20, 20, 22), -1)
            cv2.rectangle(frame, (x, ly - th - 4), (x + tw + 10, ly + 4), color, 1)
            cv2.putText(frame, label, (x + 5, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 247), 1, cv2.LINE_AA)

        return frame

    def _encode_mp4_clip(self, frames):
        if not frames:
            return b""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tmp_path = tf.name
        try:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_path, fourcc, 15.0, (w, h))
            for f in frames:
                out.write(f)
            out.release()
            with open(tmp_path, "rb") as f:
                data = f.read()
            return data
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _trigger_alert(self, detections):
        count = len(detections)
        identified = [d["name"] for d in detections if d.get("name")]
        has_loitering = any(d.get("is_loitering") for d in detections)
        custom_cap = config.get("custom_caption", "هشدار تردد MikiVo")

        state["stats"]["total_captures"] += 1
        log_entry = {
            "id": int(time.time() * 1000),
            "time": datetime.now().strftime("%H:%M:%S"),
            "persons": count,
            "names": ", ".join(identified) if identified else "ناشناس",
            "type": "ویدیو MP4" if config.get("alert_media") == "video" else "عکس JPG",
            "telegram": "در حال ارسال...",
        }
        state["logs"].insert(0, log_entry)
        if len(state["logs"]) > 50:
            state["logs"].pop()

        def alert_worker():
            media_type = config.get("alert_media", "photo")
            if media_type == "video":
                # Gather pre-incident buffer + record post-incident frames
                clip_frames = list(self.frame_buffer)
                for _ in range(25):
                    time.sleep(0.06)
                    if self.latest_annotated_mat is not None:
                        clip_frames.append(self.latest_annotated_mat.copy())
                mp4_bytes = self._encode_mp4_clip(clip_frames)
                ok = telegram.dispatch_video(mp4_bytes, count, identified, custom_cap, is_loitering=has_loitering)
                if not ok and self.latest_annotated:
                    # Graceful fallback to photo
                    ok = telegram.dispatch_photo(self.latest_annotated, count, identified, custom_cap, is_loitering=has_loitering)
            else:
                ok = telegram.dispatch_photo(self.latest_annotated, count, identified, custom_cap, is_loitering=has_loitering)

            log_entry["telegram"] = "ارسال شد" if ok else "ارسال ناموفق"
            if ok:
                state["stats"]["telegram_sent"] += 1

        threading.Thread(target=alert_worker, daemon=True).start()

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
# 3. Telegram Bot Gateway (Photo, MP4 Video, Remote Commands & Summary)
# ===================================================================
class TelegramBot:
    def __init__(self):
        self._last_id = 0
        threading.Thread(target=self._poll_loop, daemon=True).start()

    def dispatch_photo(self, jpeg_bytes, count, identified_names, caption="", is_loitering=False):
        token = config.get("bot_token", "").strip()
        chat_id = config.get("chat_id", "").strip()
        if not token or not chat_id:
            return False

        names = ", ".join(identified_names) if identified_names else "ناشناس (Unknown)"
        alert_badge = "⚠️ هشدار پرسه زدن فرد ناشناس!\n" if is_loitering else ""
        text = (
            f"🛡 MikoTim Sentinel Pro | MikiVo AI\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{alert_badge}"
            f"📌 {caption}\n"
            f"👥 تعداد اشخاص: {count}\n"
            f"🧬 هویت: {names}\n"
            f"📥 ورود: {state['stats']['in_count']} | 📤 خروج: {state['stats']['out_count']}\n"
            f"⏱ زمان: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
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

    def dispatch_video(self, mp4_bytes, count, identified_names, caption="", is_loitering=False):
        token = config.get("bot_token", "").strip()
        chat_id = config.get("chat_id", "").strip()
        if not token or not chat_id or not mp4_bytes:
            return False

        names = ", ".join(identified_names) if identified_names else "ناشناس (Unknown)"
        alert_badge = "⚠️ هشدار پرسه زدن فرد ناشناس!\n" if is_loitering else ""
        text = (
            f"🎬 ویدیو رخداد امنیتی | MikiVo AI\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{alert_badge}"
            f"📌 {caption}\n"
            f"👥 افراد در صحنه: {count}\n"
            f"🧬 هویت: {names}\n"
            f"📥 ورود: {state['stats']['in_count']} | 📤 خروج: {state['stats']['out_count']}\n"
            f"⏱ زمان ثبت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendVideo",
                data={"chat_id": chat_id, "caption": text, "supports_streaming": True},
                files={"video": ("alert_clip.mp4", mp4_bytes, "video/mp4")},
                timeout=25,
            )
            return r.json().get("ok", False)
        except Exception:
            return False

    def send_summary(self):
        token = config.get("bot_token", "").strip()
        chat_id = config.get("chat_id", "").strip()
        if not token or not chat_id:
            return False

        active_faces = len(face_db.records)
        text = (
            f"📊 گزارش آماری امنیتی MikoTim Sentinel Pro\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 تاریخ: {datetime.now().strftime('%Y/%m/%d')}\n"
            f"👥 کل رخدادهای ثبت‌شده: {state['stats']['total_captures']}\n"
            f"📥 مجموع ورود: {state['stats']['in_count']} نفر\n"
            f"📤 مجموع خروج: {state['stats']['out_count']} نفر\n"
            f"🚨 هشدارهای پرسه زدن: {state['stats']['loitering_count']} مورد\n"
            f"🧬 پروفایل‌های ذخیره‌شده: {active_faces} نفر\n"
            f"📡 وضعیت پایش: {'🟢 فعال' if state['is_streaming'] else '⚪ آماده‌باش'}"
        )
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat_id, "text": text}, timeout=8)
            return True
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
                            self.dispatch_photo(detector.latest_annotated, len(detector.last_detections), names, "عکس درخواستی زنده")
                        elif text in ("/video", "ویدیو"):
                            frames = list(detector.frame_buffer)
                            clip = detector._encode_mp4_clip(frames)
                            names = [d["name"] for d in detector.last_detections if d.get("name")]
                            self.dispatch_video(clip, len(detector.last_detections), names, "ویدیو درخواستی زنده")
                        elif text in ("/summary", "خلاصه", "گزارش"):
                            self.send_summary()
                        elif text in ("/mode photo", "حالت عکس"):
                            config["alert_media"] = "photo"
                            save_config(config)
                            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                                          json={"chat_id": chat_id, "text": "📸 نوع هشدار روی عکس تنظیم شد."}, timeout=5)
                        elif text in ("/mode video", "حالت ویدیو"):
                            config["alert_media"] = "video"
                            save_config(config)
                            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                                          json={"chat_id": chat_id, "text": "🎬 نوع هشدار روی ویدیو تنظیم شد."}, timeout=5)
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
        version="3.0.0-pro",
        features=[
            "Adaptive LAB Optical CLAHE Equalization",
            "Pose-Guided Affine Landmark Canonicalization",
            "256-D Multi-Scale Spatial-Geometric Embedding",
            "IoU + EMA Temporal Tracklet Smoother",
            "Virtual Security Zone (ROI) & Tripwire Line",
            "In/Out Direction Trajectory Counter",
            "Unauthorized Loitering Detection",
            "Dual Media Alert Dispatch (Photo / Short MP4 Video)",
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
        in_count=state["stats"]["in_count"],
        out_count=state["stats"]["out_count"],
        loitering_count=state["stats"]["loitering_count"],
        hourly_traffic=state["hourly_traffic"],
        alert_media=config.get("alert_media", "photo"),
        logs=state["logs"][:35],
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

@app.route("/api/summary", methods=["GET", "POST", "OPTIONS"])
def api_summary():
    if request.method == "POST":
        ok = telegram.send_summary()
        return jsonify(ok=ok)
    return jsonify(
        total_captures=state["stats"]["total_captures"],
        in_count=state["stats"]["in_count"],
        out_count=state["stats"]["out_count"],
        loitering_count=state["stats"]["loitering_count"],
        hourly_traffic=state["hourly_traffic"],
        faces_count=len(face_db.records)
    )

@app.route("/api/test_telegram", methods=["POST", "OPTIONS"])
def api_test_tg():
    ok, msg = telegram.test()
    return jsonify(ok=ok, message=msg)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
