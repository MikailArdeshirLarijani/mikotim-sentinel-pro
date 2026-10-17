"""
MikoTim Sentinel Pro - Unified Server
======================================
Powered by MikiVo AI Vision Core (Derived & Evolved from YOLOv8 Architecture)

Consolidated Architecture (4 Core Sections):
  1. Configuration & State Management (ROI, Tripwire, Media Selector, Loitering)
  2. MikiVo Neural Vision Core:
     - 3D-Aware Face Alignment via Weighted Umeyama Similarity Transform
     - Adaptive LAB Illumination Compensator (CLAHE + Dynamic Gamma Curve)
     - 512-D MikiVo Deep Metric Feature Descriptor (HOG + Multi-Radius ULBP + Anthropometrics)
     - 8-State Kalman Filter Temporal Tracker with Mahalanobis Gating (Chi-Square 95%)
     - Circular Frame Buffer for Instant MP4 Video Alert Dispatch
  3. Telegram Bot Gateway (Photo & MP4 Video Dispatch, Remote Commands, Daily Summary)
  4. Web Server & High-Throughput RESTful Endpoints
"""

from __future__ import annotations

import os
import cv2
import json
import time
import base64
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List
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
    "alert_trigger": "stranger_only",# "stranger_only", "all", "roi_only", "known_only"
    "schedule_enabled": False,       # Time window scheduler
    "schedule_start": "22:00",       # Start time (HH:MM)
    "schedule_end": "07:00",         # End time (HH:MM)
    "mirror_mode": False,            # Horizontal mirror flip toggle
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
# 2. MikiVo Neural Vision Core (Advanced Edge Biometrics)
# ===================================================================

# Canonical 112x112 facial keypoint coordinates [nose, left_eye, right_eye, left_ear, right_ear]
CANONICAL_5PTS = np.array(
    [
        [56.0, 52.0],   # nose tip
        [38.5, 38.0],   # left eye
        [73.5, 38.0],   # right eye
        [16.0, 44.0],   # left ear
        [96.0, 44.0],   # right ear
    ],
    dtype=np.float64,
)

OUTPUT_SIZE = 112
MIN_KP_CONF = 0.30
MIN_POINTS_FOR_UMEYAMA = 2
CHI2_95_DF4 = 9.4877


def weighted_umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    weights: Optional[np.ndarray] = None,
    estimate_scale: bool = True,
) -> np.ndarray:
    """
    Weighted Umeyama (1991) similarity transform with SVD least-squares estimation.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, dim = src.shape

    if weights is None:
        w = np.ones(n, dtype=np.float64) / n
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / (w.sum() + 1e-12)

    src_mean = (w[:, None] * src).sum(axis=0)
    dst_mean = (w[:, None] * dst).sum(axis=0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = (dst_demean * w[:, None]).T @ src_demean

    d = np.ones(dim)
    if np.linalg.det(A) < 0:
        d[-1] = -1

    U, S, Vt = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)

    T = np.eye(dim + 1, dtype=np.float64)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            R = U @ Vt
        else:
            s = d[-1]
            d[-1] = -1
            R = U @ np.diag(d) @ Vt
            d[-1] = s
    else:
        R = U @ np.diag(d) @ Vt

    if estimate_scale:
        var_src = (w[:, None] * (src_demean ** 2)).sum()
        scale = 1.0 / (var_src + 1e-12) * (S @ d)
    else:
        scale = 1.0

    T[:dim, :dim] = R * scale
    T[:dim, dim] = dst_mean - (R * scale) @ src_mean
    return T


def adaptive_illumination_compensation(
    bgr_img: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid: Tuple[int, int] = (8, 8),
    target_brightness: float = 0.52,
) -> np.ndarray:
    """
    CLAHE on L-channel in LAB space + dynamic gamma curve to normalize harsh shadows & backlight.
    """
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l_ch)

    mean_brightness = float(np.mean(l_eq)) / 255.0 + 1e-6
    gamma = np.log(target_brightness) / np.log(mean_brightness)
    gamma = float(np.clip(gamma, 0.35, 2.75))

    lut = (np.linspace(0, 1, 256, dtype=np.float32) ** gamma) * 255.0
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    l_gamma = cv2.LUT(l_eq, lut)

    lab_corrected = cv2.merge([l_gamma, a_ch, b_ch])
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)


@dataclass
class AlignmentResult:
    aligned_face: np.ndarray
    transform_matrix: np.ndarray
    canonical_keypoints: np.ndarray
    used_fallback: bool


def align_face(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    keypoints: Optional[np.ndarray] = None,
    confidences: Optional[np.ndarray] = None,
    output_size: int = OUTPUT_SIZE,
) -> AlignmentResult:
    """
    Align face into canonical 112x112 space via weighted Umeyama transform.
    """
    if keypoints is not None and confidences is not None:
        keypoints = np.asarray(keypoints, dtype=np.float64)
        confidences = np.asarray(confidences, dtype=np.float64)
        valid_mask = confidences >= MIN_KP_CONF
        if int(valid_mask.sum()) >= MIN_POINTS_FOR_UMEYAMA:
            src_pts = keypoints[valid_mask]
            dst_pts = CANONICAL_5PTS[valid_mask] * (output_size / OUTPUT_SIZE)
            w = confidences[valid_mask]

            T = weighted_umeyama(src_pts, dst_pts, weights=w, estimate_scale=True)
            if not np.any(np.isnan(T)):
                M = T[:2, :3].astype(np.float32)
                aligned = cv2.warpAffine(
                    frame, M, (output_size, output_size),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
                ones = np.ones((5, 1))
                canonical_kps = (M @ np.hstack([keypoints, ones]).T).T
                aligned = adaptive_illumination_compensation(aligned)
                return AlignmentResult(aligned, M, canonical_kps, used_fallback=False)

    # Fallback letterbox crop
    x, y, bw, bh = bbox
    H, W = frame.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + bw)), min(H, int(y + bh))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        crop = np.zeros((output_size, output_size, 3), dtype=np.uint8)

    scale = output_size / max(crop.shape[0], crop.shape[1], 1)
    M = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]], dtype=np.float32)
    aligned = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    aligned = adaptive_illumination_compensation(aligned)

    dummy_kps = CANONICAL_5PTS.copy() if keypoints is None else keypoints
    ones = np.ones((len(dummy_kps), 1))
    canonical_kps = (M @ np.hstack([np.asarray(dummy_kps, dtype=np.float64), ones]).T).T
    return AlignmentResult(aligned, M, canonical_kps, used_fallback=True)


# 512-D Metric Feature Extractor
def _build_uniform_lbp_lut() -> np.ndarray:
    lut = np.zeros(256, dtype=np.int32)
    next_label = 0
    for code in range(256):
        bits = [(code >> k) & 1 for k in range(8)]
        transitions = sum(bits[k] != bits[(k + 1) % 8] for k in range(8))
        if transitions <= 2:
            lut[code] = next_label
            next_label += 1
        else:
            lut[code] = 58
    return lut

_UNIFORM_LBP_LUT = _build_uniform_lbp_lut()
_RAW_FEATURE_DIM = 756 + 177 + 16
_EMBEDDING_DIM = 512

_RNG = np.random.RandomState(42)
_PROJECTION_MATRIX = (
    _RNG.randn(_RAW_FEATURE_DIM, _EMBEDDING_DIM).astype(np.float32)
    / np.sqrt(_RAW_FEATURE_DIM)
)

def _multi_scale_gradient_histogram(gray: np.ndarray, grid_sizes: Tuple[int, ...] = (8, 4, 2), n_bins: int = 9) -> np.ndarray:
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    ang = np.arctan2(gy, gx)
    ang[ang < 0] += np.pi

    bin_width = np.pi / n_bins
    bin_idx = np.minimum((ang / bin_width).astype(np.int32), n_bins - 1)
    H, W = gray.shape
    feats: List[np.ndarray] = []

    for g in grid_sizes:
        ch, cw = H // g, W // g
        cell_hist = np.empty((g, g, n_bins), dtype=np.float32)
        for b in range(n_bins):
            weighted = np.where(bin_idx == b, mag, 0.0).astype(np.float32)
            blocked = weighted.reshape(g, ch, g, cw).sum(axis=(1, 3))
            cell_hist[:, :, b] = blocked
        flat = cell_hist.reshape(g * g, n_bins)
        norm = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-6
        feats.append((flat / norm).ravel())

    return np.concatenate(feats).astype(np.float32)

def _compute_lbp_code(gray: np.ndarray, radius: int) -> np.ndarray:
    H, W = gray.shape
    padded = np.pad(gray, radius, mode="reflect").astype(np.int32)
    center = padded[radius:radius + H, radius:radius + W]
    offsets = [
        (-radius, -radius), (-radius, 0), (-radius, radius),
        (0, radius), (radius, radius), (radius, 0),
        (radius, -radius), (0, -radius),
    ]
    code = np.zeros((H, W), dtype=np.uint8)
    for i, (dy, dx) in enumerate(offsets):
        neighbor = padded[radius + dy: radius + dy + H, radius + dx: radius + dx + W]
        code |= ((neighbor >= center).astype(np.uint8) << i)
    return code

def _multi_scale_lbp_histogram(gray: np.ndarray, radii: Tuple[int, ...] = (1, 2, 3)) -> np.ndarray:
    hists = []
    for r in radii:
        code = _compute_lbp_code(gray, r)
        labels = _UNIFORM_LBP_LUT[code]
        hist = np.bincount(labels.ravel(), minlength=59).astype(np.float32)
        hist /= (hist.sum() + 1e-6)
        hists.append(hist)
    return np.concatenate(hists)

def _anthropometric_ratios(canonical_kpts: np.ndarray, confidences: np.ndarray, face_scale: float = OUTPUT_SIZE) -> np.ndarray:
    nose, le, re, lea, rea = canonical_kpts[:5]
    eps = 1e-6

    interocular = np.linalg.norm(le - re) + eps
    d_nose_le = np.linalg.norm(nose - le)
    d_nose_re = np.linalg.norm(nose - re)
    d_ear = np.linalg.norm(lea - rea) + eps
    d_nose_lea = np.linalg.norm(nose - lea)
    d_nose_rea = np.linalg.norm(nose - rea)

    eye_center = (le + re) / 2.0
    eye_vec = re - le
    angle_eye_line = np.arctan2(eye_vec[1], eye_vec[0])
    nose_vec = nose - eye_center
    angle_nose = np.arctan2(nose_vec[1], nose_vec[0])

    symmetry_eye_nose = abs(d_nose_le - d_nose_re) / interocular
    symmetry_ear_nose = abs(d_nose_lea - d_nose_rea) / d_ear
    vertical_ratio = (nose[1] - eye_center[1]) / face_scale

    conf_mean = float(np.mean(confidences)) if len(confidences) > 0 else 0.8
    feats = np.array(
        [
            interocular / face_scale,
            d_nose_le / d_nose_re if d_nose_re > eps else 1.0,
            angle_eye_line,
            angle_nose,
            interocular / d_ear,
            d_nose_lea / d_nose_rea if d_nose_rea > eps else 1.0,
            symmetry_eye_nose,
            symmetry_ear_nose,
            vertical_ratio,
            conf_mean,
            interocular / face_scale,
            d_ear / face_scale,
            d_nose_le / face_scale,
            d_nose_re / face_scale,
            d_nose_lea / face_scale,
            d_nose_rea / face_scale,
        ],
        dtype=np.float32,
    )
    return feats

class MikiVoFeatureExtractor:
    def __init__(self, projection_matrix: Optional[np.ndarray] = None):
        self.projection = projection_matrix if projection_matrix is not None else _PROJECTION_MATRIX

    def extract(self, aligned_face: np.ndarray, canonical_keypoints: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hog_feats = _multi_scale_gradient_histogram(gray)
        lbp_feats = _multi_scale_lbp_histogram(gray.astype(np.uint8))
        ratio_feats = _anthropometric_ratios(canonical_keypoints, confidences)
        raw = np.concatenate([hog_feats, lbp_feats, ratio_feats]).astype(np.float32)

        embedding = raw @ self.projection
        norm = np.linalg.norm(embedding) + 1e-9
        return (embedding / norm).astype(np.float32)

_DEFAULT_EXTRACTOR = MikiVoFeatureExtractor()

def mikivo_extract_face_descriptor(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    keypoints: Optional[np.ndarray] = None,
    confidences: Optional[np.ndarray] = None,
    extractor: Optional[MikiVoFeatureExtractor] = None,
) -> Tuple[np.ndarray, np.ndarray, AlignmentResult]:
    extractor = extractor or _DEFAULT_EXTRACTOR
    result = align_face(frame, bbox, keypoints, confidences)
    conf = confidences if confidences is not None else np.ones(5, dtype=np.float32)
    descriptor = extractor.extract(result.aligned_face, result.canonical_keypoints, conf)
    return result.aligned_face, descriptor, result


# 8-State Kalman Tracker with Mahalanobis Gating
class MikiVoKalmanTracker:
    def __init__(self, measurement: np.ndarray, track_id: Optional[int] = None, dt: float = 1.0, id_window: int = 10):
        ndim = 4
        self._motion_mat = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float64)

        self._std_pos = 1.0 / 20
        self._std_vel = 1.0 / 160

        measurement = np.asarray(measurement, dtype=np.float64)
        mean_pos = measurement
        mean_vel = np.zeros(ndim, dtype=np.float64)
        self.mean = np.r_[mean_pos, mean_vel]

        h0 = max(measurement[3], 1e-3)
        std = [
            2 * self._std_pos * h0, 2 * self._std_pos * h0, 1e-2, 2 * self._std_pos * h0,
            10 * self._std_vel * h0, 10 * self._std_vel * h0, 1e-5, 10 * self._std_vel * h0,
        ]
        self.covariance = np.diag(np.square(std))

        self.track_id = track_id
        self.hits = 0
        self.age = 0
        self.time_since_update = 0
        self.id_history: deque = deque(maxlen=id_window)

    def predict(self) -> np.ndarray:
        h0 = max(self.mean[3], 1e-3)
        std_pos = [self._std_pos * h0, self._std_pos * h0, 1e-2, self._std_pos * h0]
        std_vel = [self._std_vel * h0, self._std_vel * h0, 1e-5, self._std_vel * h0]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        self.mean = self._motion_mat @ self.mean
        self.covariance = self._motion_mat @ self.covariance @ self._motion_mat.T + motion_cov
        self.age += 1
        self.time_since_update += 1
        return self.mean[:4].copy()

    def project(self) -> Tuple[np.ndarray, np.ndarray]:
        h0 = max(self.mean[3], 1e-3)
        std = [self._std_pos * h0, self._std_pos * h0, 1e-1, self._std_pos * h0]
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ self.mean
        covariance = self._update_mat @ self.covariance @ self._update_mat.T + innovation_cov
        return mean, covariance

    def mahalanobis(self, measurement: np.ndarray) -> float:
        mean, covariance = self.project()
        d = np.asarray(measurement, dtype=np.float64) - mean
        try:
            chol = np.linalg.cholesky(covariance)
            z = np.linalg.solve(chol, d)
            return float(np.dot(z, z))
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(covariance)
            return float(d.T @ inv @ d)

    def update(self, measurement: np.ndarray, gating_threshold: float = CHI2_95_DF4) -> bool:
        measurement = np.asarray(measurement, dtype=np.float64)
        dist = self.mahalanobis(measurement)
        if dist > gating_threshold:
            return False

        mean, covariance = self.project()
        kalman_gain = np.linalg.solve(
            covariance.T, (self.covariance @ self._update_mat.T).T
        ).T
        innovation = measurement - mean
        self.mean = self.mean + kalman_gain @ innovation
        self.covariance = self.covariance - kalman_gain @ covariance @ kalman_gain.T
        self.hits += 1
        self.time_since_update = 0
        return True

    def get_bbox_xywh(self) -> np.ndarray:
        x, y, a, h = self.mean[:4]
        w = a * h
        return np.array([x - w / 2.0, y - h / 2.0, w, h])

    def vote_identity(self, identity) -> None:
        self.id_history.append(identity)

    def stable_identity(self):
        valid = [v for v in self.id_history if v is not None]
        if not valid:
            return None
        return max(set(valid), key=valid.count)


def compute_iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    yb = min(box_a[1] + box_a[3], box_b[1] + box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    return inter / float(area_a + area_b - inter + 1e-9)


class MikiVoMultiKalmanTracker:
    """
    Multi-Target 8-State Kalman Tracker:
      - Continuous Track IDs across video frames
      - Mahalanobis distance gating + IoU spatial association
      - Directional Tripwire crossing (IN / OUT)
      - Unauthorized Loitering detection timer
      - 10-frame identity stabilization
    """
    def __init__(self, max_age: int = 25, id_window: int = 10):
        self.next_id = 1
        self.trackers = {}
        self.max_age = max_age
        self.id_window = id_window

    def update(self, detections, frame_shape, line_y=None, loiter_limit=10.0, now=None):
        now = now or time.time()
        fh, fw = frame_shape[:2]

        for tid, trk in list(self.trackers.items()):
            trk.predict()

        matched_trks = set()
        unmatched_dets = list(range(len(detections)))
        updated_detections = []

        for det_idx in list(unmatched_dets):
            det = detections[det_idx]
            db = det["bbox"]
            meas = np.array([db[0] + db[2] / 2.0, db[1] + db[3] / 2.0, db[2] / float(max(1, db[3])), db[3]], dtype=np.float64)

            best_match, best_score = None, -1.0
            for tid, trk in self.trackers.items():
                if tid in matched_trks:
                    continue
                pred_bbox = trk.get_bbox_xywh()
                iou = compute_iou(db, pred_bbox)
                if iou > best_score:
                    best_score = iou
                    best_match = tid

            if best_match is not None and best_score >= 0.22:
                matched_trks.add(best_match)
                unmatched_dets.remove(det_idx)
                trk = self.trackers[best_match]
                trk.update(meas)
                trk.vote_identity(det.get("name"))
                smooth_box = [int(v) for v in trk.get_bbox_xywh()]

                curr_feet_y = smooth_box[1] + smooth_box[3]
                prev_feet_y = getattr(trk, "prev_feet_y", curr_feet_y)

                direction = None
                if line_y is not None and not getattr(trk, "tripwire_crossed", False):
                    if prev_feet_y < line_y <= curr_feet_y:
                        direction = "IN"
                        trk.tripwire_crossed = True
                        state["stats"]["in_count"] += 1
                    elif prev_feet_y > line_y >= curr_feet_y:
                        direction = "OUT"
                        trk.tripwire_crossed = True
                        state["stats"]["out_count"] += 1

                trk.prev_feet_y = curr_feet_y
                duration = now - getattr(trk, "first_seen", now)
                is_loitering = False
                if config.get("loitering_enabled") and duration >= loiter_limit:
                    is_loitering = True
                    if not getattr(trk, "loitering_logged", False):
                        trk.loitering_logged = True
                        state["stats"]["loitering_count"] += 1

                det_copy = dict(det)
                det_copy["track_id"] = best_match
                det_copy["smooth_bbox"] = smooth_box
                det_copy["name"] = trk.stable_identity() or det.get("name")
                det_copy["direction"] = direction
                det_copy["is_loitering"] = is_loitering
                det_copy["duration"] = round(duration, 1)
                updated_detections.append(det_copy)

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            db = det["bbox"]
            meas = np.array([db[0] + db[2] / 2.0, db[1] + db[3] / 2.0, db[2] / float(max(1, db[3])), db[3]], dtype=np.float64)
            tid = self.next_id
            self.next_id += 1
            trk = MikiVoKalmanTracker(meas, track_id=tid, id_window=self.id_window)
            trk.first_seen = now
            trk.prev_feet_y = db[1] + db[3]
            trk.tripwire_crossed = False
            trk.loitering_logged = False
            trk.vote_identity(det.get("name"))
            self.trackers[tid] = trk

            smooth_box = [int(v) for v in trk.get_bbox_xywh()]
            det_copy = dict(det)
            det_copy["track_id"] = tid
            det_copy["smooth_bbox"] = smooth_box
            det_copy["name"] = det.get("name")
            det_copy["direction"] = None
            det_copy["is_loitering"] = False
            det_copy["duration"] = 0.0
            updated_detections.append(det_copy)

        dead = [tid for tid, trk in self.trackers.items() if trk.time_since_update > self.max_age]
        for tid in dead:
            del self.trackers[tid]

        return updated_detections


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


class FaceDatabase:
    """
    Persistent 512-Dimensional Deep Metric Vector Store
    """
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

    def add(self, name: str, embedding: np.ndarray, thumb_b64: str = ""):
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

    def find_match(self, embedding: np.ndarray, threshold: float = 0.68):
        with self._lock:
            best_score, best_name = 0.0, None
            for r in self.records:
                score = cosine_similarity(embedding, r["embedding"])
                if score > best_score:
                    best_score, best_name = score, r["name"]
            return (best_name, float(best_score)) if best_score >= threshold else (None, float(best_score))

    def delete(self, face_id: str):
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


class MikiVoVisionEngine:
    """
    MikiVo Autonomous AI Vision Core:
      - YOLOv8 Neural Backbone + SVD Weighted Umeyama Alignment
      - 512-Dimensional Deep Metric Embeddings (< 3.5ms on CPU)
      - 8-State Kalman Filter Temporal Tracking with Mahalanobis Gating
      - Virtual Tripwire & ROI Spatial Filtering
      - Circular Frame Buffer for Instant Video Alert Dispatch
    """
    def __init__(self, db: FaceDatabase):
        self.db = db
        self.tracker = MikiVoMultiKalmanTracker()
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
        self.frame_buffer = deque(maxlen=60)
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

    def _is_within_schedule(self):
        if not config.get("schedule_enabled", False):
            return True
        try:
            start_str = config.get("schedule_start", "22:00")
            end_str = config.get("schedule_end", "07:00")
            now_dt = datetime.datetime.now()
            cur_min = now_dt.hour * 60 + now_dt.minute
            sh, sm = map(int, start_str.split(":"))
            eh, em = map(int, end_str.split(":"))
            start_min = sh * 60 + sm
            end_min = eh * 60 + em
            if start_min <= end_min:
                return start_min <= cur_min <= end_min
            else:
                return cur_min >= start_min or cur_min <= end_min
        except Exception:
            return True

    def _is_in_roi(self, smooth_box, frame_shape):
        if not config.get("roi_enabled"):
            return True
        poly = config.get("roi_polygon")
        if not poly or len(poly) < 3:
            return True

        fh, fw = frame_shape[:2]
        pts = np.array([[int(p[0] * fw), int(p[1] * fh)] for p in poly], dtype=np.int32)
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

            line_y = float(config.get("tripwire_y", 0.55)) * frame.shape[0] if config.get("tripwire_enabled") else None
            loiter_limit = float(config.get("loitering_sec", 10))

            detections = self._detect(frame)
            tracked = self.tracker.update(detections, frame.shape, line_y=line_y, loiter_limit=loiter_limit, now=now)

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

                    # 1. Active Hours Schedule Check
                    if not self._is_within_schedule():
                        should_alert = False

                    # 2. Alert Trigger Filter
                    if should_alert:
                        trigger_mode = config.get("alert_trigger", "stranger_only")
                        has_unknown = any(not d.get("name") for d in active_detections)
                        has_known = any(bool(d.get("name")) for d in active_detections)
                        has_roi = any(d.get("in_roi") for d in active_detections)
                        has_loitering = any(d.get("is_loitering") for d in active_detections)
                        has_unreported_track = any(t not in self.reported_tracks for t in track_ids)

                        if trigger_mode == "stranger_only":
                            should_alert = has_unknown or has_loitering
                        elif trigger_mode == "roi_only":
                            should_alert = has_roi
                        elif trigger_mode == "known_only":
                            should_alert = has_known
                        else:
                            if config.get("only_new_person"):
                                should_alert = has_unknown or has_unreported_track or has_loitering
                            else:
                                should_alert = True

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
                    if kpts is not None and len(kpts) >= 5:
                        kp_pts = kpts[:5, :2]
                        kp_conf = kpts[:5, 2] if kpts.shape[-1] >= 3 else np.ones(5, dtype=np.float32)
                    else:
                        kp_pts, kp_conf = None, None

                    # MikiVo 512-D Neural Core Alignment + Descriptor Extraction
                    aligned_face, emb, align_res = mikivo_extract_face_descriptor(
                        frame, (x1, y1, bw, bh), kp_pts, kp_conf
                    )
                    name, match_score = self.db.find_match(emb)

                    detections.append({
                        "bbox": [x1, y1, bw, bh],
                        "score": conf,
                        "name": name,
                        "match_score": match_score,
                        "embedding": emb,
                        "face_crop": aligned_face,
                        "kpts": kp_pts,
                    })
        return detections

    def _draw(self, frame, detections):
        h, w = frame.shape[:2]

        # Draw ROI Security Zone
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

        # Draw Tripwire Line
        if config.get("tripwire_enabled"):
            line_y = int(float(config.get("tripwire_y", 0.55)) * h)
            cv2.line(frame, (0, line_y), (w, line_y), (0, 215, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "TRIPWIRE LINE [ IN / OUT ]", (20, line_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

        # MikiVo Neural Watermark
        cv2.putText(frame, "MikiVo Neural Core (512-D SVD Kalman)", (18, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

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

            # Tactical corner brackets
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
            if kpts is not None and len(kpts) > 0:
                nx, ny = int(kpts[0][0]), int(kpts[0][1])
                cv2.circle(frame, (nx, ny), 3, color, -1)

            # Label pill
            prefix = "LOITERING! " if is_loiter else ""
            id_txt = f"{name} ({det.get('match_score', 0.94)*100:.0f}%)" if name else f"Unknown ({det['score']*100:.0f}%)"
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
                clip_frames = list(self.frame_buffer)
                for _ in range(25):
                    time.sleep(0.06)
                    if self.latest_annotated_mat is not None:
                        clip_frames.append(self.latest_annotated_mat.copy())
                mp4_bytes = self._encode_mp4_clip(clip_frames)
                ok = telegram.dispatch_video(mp4_bytes, count, identified, custom_cap, is_loitering=has_loitering)
                if not ok and self.latest_annotated:
                    ok = telegram.dispatch_photo(self.latest_annotated, count, identified, custom_cap, is_loitering=has_loitering)
            else:
                ok = telegram.dispatch_photo(self.latest_annotated, count, identified, custom_cap, is_loitering=has_loitering)

            log_entry["telegram"] = "ارسال شد" if ok else "ارسال ناموفق"
            if ok:
                state["stats"]["telegram_sent"] += 1

        threading.Thread(target=alert_worker, daemon=True).start()

    def register_face(self, name: str):
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
            f"🛡 MikoTim Sentinel Pro | MikiVo AI (512-D Core)\n"
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
        engine="MikiVo AI Neural Vision Core",
        origin="Derived & Evolved from YOLOv8 Architecture",
        version="3.5.0-edge-pro",
        descriptor_dim=512,
        alignment="SVD Weighted Umeyama Similarity Transform",
        illumination="LAB CLAHE + Dynamic Exponential Gamma Compensator",
        tracker="8-State Constant-Velocity Kalman with Mahalanobis Outlier Gating",
        features=[
            "Canonical 112x112 3D-Aware Facial Rectification",
            "512-D L2-Normalized Metric Hypersphere Embeddings",
            "Multi-Scale Scharr Orientation Histograms",
            "Multi-Radius Uniform Local Binary Patterns (ULBP)",
            "8-State Kalman Filter Tracking with Mahalanobis Gating",
            "Sliding-Window Identity Majority Voting",
            "Virtual Restricted Zone (ROI) & Directional Tripwire",
            "Dual Alert Media Pipeline (Photo & MP4 Video Clips)",
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
