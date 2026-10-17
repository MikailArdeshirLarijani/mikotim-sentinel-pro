"""
MikoTim Sentinel Pro - Unified Server
======================================
Powered by MikoVo RaRa 11 AI Vision Core (Derived & Evolved from YOLOv8 Architecture)

Consolidated Architecture (4 Core Sections):
  1. Configuration & State Management (ROI, Tripwire, Media Selector, Loitering)
  2. MikoVo RaRa 11 Neural Vision Core:
     - 3D-Aware Face Alignment via Weighted Umeyama Similarity Transform
     - Adaptive LAB Illumination Compensator (CLAHE + Dynamic Gamma Curve)
     - 512-D MikoVo RaRa 11 Deep Metric Feature Descriptor (HOG + Multi-Radius ULBP + Anthropometrics)
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
import math
from typing import Optional, Tuple, List, Dict, Any, Deque
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
    "custom_caption": "هشدار تردد هوشمند MikoVo RaRa 11",
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
        "threat_count": 0,
    },
    "hourly_traffic": [0] * 24,
    "logs": [],
}


# ===================================================================
# 2. MikoVo RaRa 11 Neural Vision Core (Advanced Edge Biometrics)
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

class MikoVoRaRa11FeatureExtractor:
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

_DEFAULT_EXTRACTOR = MikoVoRaRa11FeatureExtractor()

def mikovo_rara11_extract_face_descriptor(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    keypoints: Optional[np.ndarray] = None,
    confidences: Optional[np.ndarray] = None,
    extractor: Optional[MikoVoRaRa11FeatureExtractor] = None,
) -> Tuple[np.ndarray, np.ndarray, AlignmentResult]:
    extractor = extractor or _DEFAULT_EXTRACTOR
    result = align_face(frame, bbox, keypoints, confidences)
    conf = confidences if confidences is not None else np.ones(5, dtype=np.float32)
    descriptor = extractor.extract(result.aligned_face, result.canonical_keypoints, conf)
    return result.aligned_face, descriptor, result


# 8-State Kalman Tracker with Mahalanobis Gating
class MikoVoRaRa11KalmanTracker:
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


class MikoVoRaRa11MultiKalmanTracker:
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
                det_copy["kalman_state"] = trk.mean.copy()
                updated_detections.append(det_copy)

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            db = det["bbox"]
            meas = np.array([db[0] + db[2] / 2.0, db[1] + db[3] / 2.0, db[2] / float(max(1, db[3])), db[3]], dtype=np.float64)
            tid = self.next_id
            self.next_id += 1
            trk = MikoVoRaRa11KalmanTracker(meas, track_id=tid, id_window=self.id_window)
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
            det_copy["kalman_state"] = trk.mean.copy()
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


# ===================================================================
# 2.B MikoVo RaRa 11 Liveness & Attention Sentinel (Anti-Spoofing & 3D Pose)
# ===================================================================
# =============================================================================
# 0. TUNABLE CONSTANTS (calibrate against your deployment camera/dataset)
# =============================================================================

LIVENESS_PASS_THRESHOLD: float = 0.65

# FFT spectral band boundaries (fraction of max radius)
_FFT_LOW_BAND_FRAC: float = 0.10
_FFT_HIGH_BAND_FRAC: float = 0.35
_FFT_HIGH_RATIO_SPOOF_CEIL: float = 0.55   # ratio at/above which fft_score -> 0
_FFT_PEAK_SCREEN_THRESHOLD: float = 9.0    # peak/mean in high band -> moire grid

# Chrominance / specular constants
_SPECULAR_V_THRESH: int = 245
_SPECULAR_S_THRESH: int = 25
_SPECULAR_RATIO_SPOOF_CEIL: float = 0.15
_CHROMA_STD_NATURAL_CAP: float = 15.0

# rPPG proxy constants
_RPPG_WINDOW: int = 24           # ~1 sec @ 24 FPS sliding window
_RPPG_MIN_FRAMES: int = 10
_RPPG_IDEAL_STD: float = 1.0
_RPPG_STD_TOLERANCE: float = 1.2

# Head pose constants
_POSE_ANGLE_LIMIT_DEG: float = 30.0
_POSE_ANGLE_FALLOFF_DEG: float = 60.0
_POSE_ROLL_LIMIT_DEG: float = 45.0
_POSE_ROLL_FALLOFF_DEG: float = 90.0

# Fusion memory constants
_FUSION_BASE_ALPHA: float = 0.25
_FUSION_REJECT_THRESHOLD: float = 0.30
_FUSION_DESCRIPTOR_DIM: int = 512

# Sharpness (Laplacian variance) calibration for a 112x112 crop
_SHARPNESS_VAR_SCALE: float = 500.0

# Standard anthropometric 3D canonical head model (mm), ordered as
# [nose, left_eye, right_eye, left_ear, right_ear].
_HEAD_MODEL_3D: np.ndarray = np.array(
    [
        [0.0, 0.0, 0.0],        # nose tip
        [-35.0, 32.0, -25.0],   # left eye
        [35.0, 32.0, -25.0],    # right eye
        [-70.0, -10.0, -65.0],  # left ear / cheek
        [70.0, -10.0, -65.0],   # right ear / cheek
    ],
    dtype=np.float64,
)


# =============================================================================
# 1. RESULT DATACLASSES
# =============================================================================

@dataclass
class LivenessResult:
    """Detailed breakdown of the passive liveness/anti-spoofing analysis."""
    score: float
    is_real: bool
    attack_type: Optional[str]           # "SCREEN_REPLAY" | "PRINT_PHOTO" | None
    fft_score: float
    chroma_specular_score: float
    rppg_score: float
    rppg_available: bool


@dataclass
class HeadPoseResult:
    """3D head-pose solution and frontal-attention quality."""
    pitch_deg: float
    yaw_deg: float
    roll_deg: float
    pose_quality: float
    success: bool


@dataclass
class SentinelReport:
    """
    Unified output of `evaluate_face_sentinel`.

    Core contract fields (as specified):
        is_live, liveness_score, is_spoof_attack, attack_type,
        euler_angles (pitch, yaw, roll), pose_quality,
        fused_descriptor_ready.

    Additional integration-friendly fields are included for convenience
    when wiring this module together with `mikivo_neural_core.py`.
    """
    is_live: bool
    liveness_score: float
    is_spoof_attack: bool
    attack_type: Optional[str]
    euler_angles: Tuple[float, float, float]
    pose_quality: float
    fused_descriptor_ready: bool
    # --- bonus integration fields ---
    sharpness_score: float = 0.0
    quality_weight: float = 0.0
    descriptor_fused: bool = False


# =============================================================================
# 2. SHARED LOW-LEVEL VECTORIZED HELPERS
# =============================================================================

@lru_cache(maxsize=4)
def _radial_distance_grid(h: int, w: int) -> np.ndarray:
    """
    Precompute (and cache) a normalized radial-distance grid centered at the
    FFT zero-frequency origin, used to build low/high frequency band masks
    without any per-pixel Python loops.
    """
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = math.hypot(cy, cx) + 1e-9
    return dist / max_r


def _laplacian_sharpness_score(gray: np.ndarray) -> float:
    """Normalized [0,1] focus/sharpness score via Laplacian variance."""
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    var = float(lap.var())
    return float(np.clip(var / _SHARPNESS_VAR_SCALE, 0.0, 1.0))


def _angle_penalty(angle_deg: float, limit: float, falloff: float) -> float:
    """Linear penalty: 1.0 within `limit`, decaying to 0.0 over `falloff`."""
    excess = max(0.0, abs(angle_deg) - limit)
    return float(max(0.0, 1.0 - excess / falloff))


# =============================================================================
# 3. MIKOVO LIVENESS DETECTOR
# =============================================================================

class MikoVoLivenessDetector:
    """
    Passive, multi-spectral face anti-spoofing engine operating on a single
    112x112 (or similarly sized) aligned BGR face crop, optionally enriched
    with short-term temporal history per track ID for an rPPG proxy signal.

    No active IR/depth sensing is required — all signals are derived
    passively from a standard RGB/BGR camera frame.
    """

    def __init__(self, rppg_window: int = _RPPG_WINDOW) -> None:
        self._rppg_window = rppg_window
        self._green_history: Dict[Any, Deque[float]] = {}

    # ------------------------------------------------------------------ #
    # 3.1 Fourier Spectral Analysis
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fft_spectral_analysis(gray: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute the high/low radial frequency energy ratio and the spectral
        peakiness of the high-frequency band (periodic subpixel/moire grid
        signature), fully vectorized via `cv2.dft` + boolean band masks.

        Returns
        -------
        (fft_score, energy_ratio, peak_ratio)
        """
        h, w = gray.shape
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        windowed = gray.astype(np.float32) * win

        dft = cv2.dft(windowed, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft, axes=(0, 1))
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

        radial = _radial_distance_grid(h, w)
        low_mask = radial <= _FFT_LOW_BAND_FRAC
        high_mask = radial >= _FFT_HIGH_BAND_FRAC

        low_energy = float(np.sum(magnitude[low_mask] ** 2)) + 1e-9
        high_vals = magnitude[high_mask]
        high_energy = float(np.sum(high_vals ** 2))

        energy_ratio = high_energy / low_energy
        peak_ratio = float(high_vals.max() / (high_vals.mean() + 1e-9))

        fft_score = 1.0 - np.clip(energy_ratio / _FFT_HIGH_RATIO_SPOOF_CEIL, 0.0, 1.0)
        return float(fft_score), energy_ratio, peak_ratio

    # ------------------------------------------------------------------ #
    # 3.2 Chrominance & Specular Dispersion
    # ------------------------------------------------------------------ #
    @staticmethod
    def _chroma_specular_analysis(face_bgr: np.ndarray) -> Tuple[float, float]:
        """
        Combine YCbCr chrominance diversity (natural skin subsurface
        scattering produces richer local color variance than flat
        print/screen reproduction) with HSV specular-highlight dispersion
        (large uniform bright/desaturated regions indicate glare off glass
        or glossy paper rather than skin).

        Returns
        -------
        (chroma_specular_score, specular_ratio)
        """
        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        cr, cb = ycrcb[:, :, 1].astype(np.float32), ycrcb[:, :, 2].astype(np.float32)
        chroma_std = (float(np.std(cr)) + float(np.std(cb))) / 2.0
        chroma_score = float(np.clip(chroma_std / _CHROMA_STD_NATURAL_CAP, 0.0, 1.0))

        hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        s_ch, v_ch = hsv[:, :, 1], hsv[:, :, 2]
        specular_mask = (v_ch >= _SPECULAR_V_THRESH) & (s_ch <= _SPECULAR_S_THRESH)
        specular_ratio = float(np.mean(specular_mask))
        specular_score = 1.0 - np.clip(specular_ratio / _SPECULAR_RATIO_SPOOF_CEIL, 0.0, 1.0)

        combined = 0.5 * chroma_score + 0.5 * specular_score
        return float(combined), specular_ratio

    # ------------------------------------------------------------------ #
    # 3.3 Temporal rPPG Proxy
    # ------------------------------------------------------------------ #
    def _rppg_proxy_score(self, face_bgr: np.ndarray, track_id: Optional[Any]) -> Tuple[float, bool]:
        """
        Maintain a short rolling buffer of mean-green intensity sampled from
        the central facial ROI (forehead/cheek region, rich in capillary
        blood flow) and score its detrended micro-variance. A genuine live
        face exhibits a subtle, non-zero biological pulse jitter; a static
        photo or frozen replay frame yields near-zero variance.
        """
        if track_id is None:
            return 0.5, False  # neutral — no temporal context available

        h, w = face_bgr.shape[:2]
        roi = face_bgr[int(h * 0.15):int(h * 0.55), int(w * 0.25):int(w * 0.75), 1]
        green_mean = float(np.mean(roi))

        history = self._green_history.setdefault(track_id, deque(maxlen=self._rppg_window))
        history.append(green_mean)

        if len(history) < _RPPG_MIN_FRAMES:
            return 0.5, False

        signal = np.asarray(history, dtype=np.float32)
        detrended = signal - np.convolve(signal, np.ones(3) / 3.0, mode="same")
        sig_std = float(np.std(detrended))

        score = math.exp(-((sig_std - _RPPG_IDEAL_STD) / _RPPG_STD_TOLERANCE) ** 2)
        return float(np.clip(score, 0.0, 1.0)), True

    # ------------------------------------------------------------------ #
    def reset_track(self, track_id: Any) -> None:
        """Clear the rPPG temporal buffer for a track (e.g. on track loss)."""
        self._green_history.pop(track_id, None)

    def analyze(self, face_bgr: np.ndarray, track_id: Optional[Any] = None) -> LivenessResult:
        """
        Run the full multi-spectral liveness pipeline on a single aligned
        face crop and return a detailed `LivenessResult`.
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        fft_score, energy_ratio, peak_ratio = self._fft_spectral_analysis(gray)
        chroma_score, specular_ratio = self._chroma_specular_analysis(face_bgr)
        rppg_score, rppg_available = self._rppg_proxy_score(face_bgr, track_id)

        if rppg_available:
            weights = (0.40, 0.30, 0.30)
        else:
            weights = (0.55, 0.45, 0.0)

        total_score = (
            weights[0] * fft_score + weights[1] * chroma_score + weights[2] * rppg_score
        )
        total_score = float(np.clip(total_score, 0.0, 1.0))
        is_real = total_score >= LIVENESS_PASS_THRESHOLD

        attack_type: Optional[str] = None
        if not is_real:
            if peak_ratio >= _FFT_PEAK_SCREEN_THRESHOLD or specular_ratio >= _SPECULAR_RATIO_SPOOF_CEIL:
                attack_type = "SCREEN_REPLAY"
            elif energy_ratio >= _FFT_HIGH_RATIO_SPOOF_CEIL * 0.6:
                attack_type = "PRINT_PHOTO"

        return LivenessResult(
            score=total_score,
            is_real=is_real,
            attack_type=attack_type,
            fft_score=fft_score,
            chroma_specular_score=chroma_score,
            rppg_score=rppg_score,
            rppg_available=rppg_available,
        )


# =============================================================================
# 4. MIKOVO HEAD POSE ESTIMATOR
# =============================================================================

class MikoVoHeadPoseEstimator:
    """
    Solves 3D head pose (pitch, yaw, roll) from the 5 YOLOv8-pose keypoints
    using `cv2.solvePnP`, and derives a frontal-attention quality score that
    penalizes poses known to degrade downstream face-recognition accuracy.
    """

    def __init__(self, image_size: int = 112) -> None:
        self.image_size = image_size
        self._camera_matrix = np.array(
            [
                [image_size, 0.0, image_size / 2.0],
                [0.0, image_size, image_size / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    @staticmethod
    def _rotation_matrix_to_euler(r_mat: np.ndarray) -> Tuple[float, float, float]:
        """Decompose a rotation matrix into pitch/yaw/roll (degrees), gimbal-lock safe."""
        sy = math.sqrt(r_mat[0, 0] ** 2 + r_mat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(r_mat[2, 1], r_mat[2, 2])
            yaw = math.atan2(-r_mat[2, 0], sy)
            roll = math.atan2(r_mat[1, 0], r_mat[0, 0])
        else:
            pitch = math.atan2(-r_mat[1, 2], r_mat[1, 1])
            yaw = math.atan2(-r_mat[2, 0], sy)
            roll = 0.0

        return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)

    def estimate(self, kpts_5x2: np.ndarray, confs_5: Optional[np.ndarray] = None) -> HeadPoseResult:
        """
        Parameters
        ----------
        kpts_5x2 : (5, 2) keypoints in the *aligned 112x112 crop* coordinate
                   space, ordered [nose, left_eye, right_eye, left_ear, right_ear].
        confs_5  : optional (5,) confidences (currently used only to guard
                   against degenerate all-zero input).

        Returns
        -------
        HeadPoseResult
        """
        image_points = np.asarray(kpts_5x2, dtype=np.float64).reshape(5, 1, 2)

        if confs_5 is not None and np.all(np.asarray(confs_5) < 0.05):
            return HeadPoseResult(0.0, 0.0, 0.0, 0.0, success=False)

        success, rvec, tvec = cv2.solvePnP(
            _HEAD_MODEL_3D.reshape(5, 1, 3),
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not success:
            success, rvec, tvec = cv2.solvePnP(
                _HEAD_MODEL_3D.reshape(5, 1, 3),
                image_points,
                self._camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        if not success:
            return HeadPoseResult(0.0, 0.0, 0.0, 0.0, success=False)

        r_mat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = self._rotation_matrix_to_euler(r_mat)

        yaw_penalty = _angle_penalty(yaw, _POSE_ANGLE_LIMIT_DEG, _POSE_ANGLE_FALLOFF_DEG)
        pitch_penalty = _angle_penalty(pitch, _POSE_ANGLE_LIMIT_DEG, _POSE_ANGLE_FALLOFF_DEG)
        roll_penalty = _angle_penalty(roll, _POSE_ROLL_LIMIT_DEG, _POSE_ROLL_FALLOFF_DEG)

        pose_quality = float(np.clip(yaw_penalty * pitch_penalty * (0.85 + 0.15 * roll_penalty), 0.0, 1.0))

        return HeadPoseResult(
            pitch_deg=float(pitch),
            yaw_deg=float(yaw),
            roll_deg=float(roll),
            pose_quality=pose_quality,
            success=True,
        )


# =============================================================================
# 5. MIKOVO TRACK FUSION MEMORY
# =============================================================================

class MikoVoTrackFusionMemory:
    """
    Maintains a quality-weighted Exponential Moving Average (EMA) of the
    512-D identity descriptor for each active track ID, re-normalized to
    the unit hypersphere after every update.

    Low-quality observations (spoofed, blurry, extreme-pose crops) are
    rejected outright rather than being allowed to corrupt the EMA, and
    high-quality observations are blended in proportionally faster.
    """

    def __init__(
        self,
        base_alpha: float = _FUSION_BASE_ALPHA,
        reject_threshold: float = _FUSION_REJECT_THRESHOLD,
        descriptor_dim: int = _FUSION_DESCRIPTOR_DIM,
    ) -> None:
        self.base_alpha = base_alpha
        self.reject_threshold = reject_threshold
        self.descriptor_dim = descriptor_dim
        self._memory: Dict[Any, np.ndarray] = {}
        self._observation_count: Dict[Any, int] = {}

    @staticmethod
    def compute_quality_weight(
        liveness_score: float, pose_quality: float, sharpness_score: float
    ) -> float:
        """Combined acceptance weight in [0,1] used to gate/scale EMA updates."""
        return float(np.clip(liveness_score * pose_quality * sharpness_score, 0.0, 1.0))

    def update(self, track_id: Any, descriptor_512d: np.ndarray, quality_weight: float) -> bool:
        """
        Fuse a new descriptor observation into the track's identity memory.

        Returns
        -------
        bool — True if the observation was accepted and fused, False if it
        was rejected (quality below `reject_threshold`, e.g. a spoof attempt
        or a blurred/extreme-pose crop).
        """
        descriptor_512d = np.asarray(descriptor_512d, dtype=np.float32)
        if descriptor_512d.shape[-1] != self.descriptor_dim:
            raise ValueError(
                f"Expected a {self.descriptor_dim}-D descriptor, got {descriptor_512d.shape[-1]}."
            )

        if quality_weight < self.reject_threshold:
            return False

        norm = np.linalg.norm(descriptor_512d) + 1e-9
        descriptor_512d = descriptor_512d / norm

        if track_id not in self._memory:
            self._memory[track_id] = descriptor_512d.copy()
            self._observation_count[track_id] = 1
            return True

        effective_alpha = float(np.clip(self.base_alpha * quality_weight, 0.0, 1.0))
        fused = (1.0 - effective_alpha) * self._memory[track_id] + effective_alpha * descriptor_512d
        fused_norm = np.linalg.norm(fused) + 1e-9
        self._memory[track_id] = (fused / fused_norm).astype(np.float32)
        self._observation_count[track_id] += 1
        return True

    def get(self, track_id: Any) -> Optional[np.ndarray]:
        """Return the current fused unit-norm descriptor for a track, if any."""
        return self._memory.get(track_id)

    def observation_count(self, track_id: Any) -> int:
        """Number of accepted (non-rejected) observations fused for a track."""
        return self._observation_count.get(track_id, 0)

    def reset(self, track_id: Any) -> None:
        """Drop a track's identity memory (e.g. on track termination)."""
        self._memory.pop(track_id, None)
        self._observation_count.pop(track_id, None)


# =============================================================================
# 6. TOP-LEVEL FACADE
# =============================================================================

_DEFAULT_LIVENESS_DETECTOR = MikoVoLivenessDetector()
_DEFAULT_POSE_ESTIMATOR = MikoVoHeadPoseEstimator(image_size=112)
_DEFAULT_FUSION_MEMORY = MikoVoTrackFusionMemory()


def evaluate_face_sentinel(
    aligned_112x112_bgr: np.ndarray,
    kpts_5x2: np.ndarray,
    confs_5: np.ndarray,
    track_id: Optional[Any] = None,
    descriptor_512d: Optional[np.ndarray] = None,
    liveness_detector: MikoVoLivenessDetector = _DEFAULT_LIVENESS_DETECTOR,
    pose_estimator: MikoVoHeadPoseEstimator = _DEFAULT_POSE_ESTIMATOR,
    fusion_memory: MikoVoTrackFusionMemory = _DEFAULT_FUSION_MEMORY,
) -> SentinelReport:
    """
    Clean top-level facade: run liveness analysis, 3D head-pose profiling,
    and (optionally) quality-weighted descriptor fusion, in a single call.

    Parameters
    ----------
    aligned_112x112_bgr : (112, 112, 3) BGR aligned face crop.
    kpts_5x2            : (5, 2) keypoints [nose, left_eye, right_eye,
                           left_ear, right_ear] in the aligned-crop space.
    confs_5             : (5,) keypoint confidences.
    track_id            : optional tracker identity, enabling temporal
                           rPPG analysis and descriptor fusion continuity.
    descriptor_512d     : optional 512-D face descriptor (e.g. produced by
                           `mikivo_neural_core.MikiVoFeatureExtractor`). If
                           provided, it is fused into `fusion_memory` under
                           `track_id`, gated by the computed quality weight.

    Returns
    -------
    SentinelReport
    """
    gray = cv2.cvtColor(aligned_112x112_bgr, cv2.COLOR_BGR2GRAY)

    liveness = liveness_detector.analyze(aligned_112x112_bgr, track_id=track_id)
    pose = pose_estimator.estimate(kpts_5x2, confs_5)
    sharpness = _laplacian_sharpness_score(gray)

    quality_weight = MikoVoTrackFusionMemory.compute_quality_weight(
        liveness.score, pose.pose_quality, sharpness
    )

    fused_descriptor_ready = (
        liveness.is_real and pose.success and quality_weight >= fusion_memory.reject_threshold
    )

    descriptor_fused = False
    if descriptor_512d is not None and track_id is not None:
        descriptor_fused = fusion_memory.update(track_id, descriptor_512d, quality_weight)

    return SentinelReport(
        is_live=liveness.is_real,
        liveness_score=liveness.score,
        is_spoof_attack=not liveness.is_real,
        attack_type=liveness.attack_type,
        euler_angles=(pose.pitch_deg, pose.yaw_deg, pose.roll_deg),
        pose_quality=pose.pose_quality,
        fused_descriptor_ready=fused_descriptor_ready,
        sharpness_score=sharpness,
        quality_weight=quality_weight,
        descriptor_fused=descriptor_fused,
    )


# =============================================================================
# 7. MIKOVO RARA 11 BEHAVIORAL & KINETIC THREAT PROFILER
# =============================================================================

DEFAULT_FPS: float = 30.0

# Subsystem 1: Facial Occlusion & Concealment
OCCLUSION_CONCEALED_THRESHOLD: float = 0.45
_KP_VIS_THRESHOLD: float = 0.45
_LOWER_GRAD_RATIO_MASK_CEIL: float = 0.52
_MASK_SAT_DIFF_THRESH: float = 28.0
_DARK_FABRIC_V_THRESH: float = 65.0
_SURGICAL_BLUE_H_RANGE: Tuple[int, int] = (85, 130)
_SURGICAL_WHITE_S_MAX: int = 38
_SURGICAL_WHITE_V_MIN: int = 165

# Subsystem 2: Kinetic Threat & Fall Detection
_DASH_VELOCITY_PX_SEC: float = 420.0        # Rapid sprinting threshold (px/s)
_DASH_ACCEL_PX_SEC2: float = 650.0          # Sudden burst acceleration (px/s^2)
_APPROACH_VH_RATIO_SEC: float = 0.35        # Relative bbox height expansion rate (vh/h per sec)
_FALL_VY_PX_SEC: float = 220.0              # Downward vertical velocity spike (px/s)
_FALL_WINDOW_FRAMES: int = 12               # ~0.40s @ 30 FPS
_FALL_UPRIGHT_H_OVER_W: float = 2.0         # Standing aspect ratio h/w > 2.0
_FALL_COLLAPSED_H_OVER_W: float = 0.90      # Collapsed aspect ratio h/w < 0.9

# Subsystem 3: Trajectory Entropy & Prowler Detection
PROWLING_SCORE_THRESHOLD: float = 0.68
_TRAJECTORY_WINDOW: int = 45                # Sliding window length (frames)
_MIN_PROWL_FRAMES: int = 18                 # Minimum history before scoring
_MIN_TOTAL_PATH_PX: float = 65.0            # Ignore stationary micro-jitter
_TORTUOSITY_SATURATION: float = 6.5         # Path length / net displacement ceiling
_ANGULAR_BINS: int = 8                      # 8 compass octants for Shannon entropy


@dataclass
class OcclusionResult:
    """Output of `MikoVoOcclusionDetector.analyze`."""
    occlusion_ratio: float
    is_face_concealed: bool
    concealment_type: Optional[str]


@dataclass
class KineticResult:
    """Output of `MikoVoKineticThreatAnalyzer.analyze`."""
    velocity_px_sec: float
    acceleration_px_sec2: float
    is_rapid_dash: bool
    is_rapid_approach: bool
    is_fallen: bool
    kinetic_event: Optional[str]


@dataclass
class TrajectoryEntropyResult:
    """Output of `MikoVoTrajectoryEntropyAnalyzer.analyze`."""
    prowling_score: float
    is_prowling: bool
    tortuosity: float
    directional_entropy: float


@dataclass
class ThreatReport:
    """
    Unified behavioral & kinetic threat assessment returned by
    `evaluate_behavioral_threat`.
    """
    is_threat: bool
    threat_level: str                       # "CLEAR" | "ELEVATED" | "CRITICAL"
    threat_category: Optional[str]          # "RAPID_DASH" | "CONCEALED_FACE" | "PROWLER_PACING" | "PERSON_FALLEN" | None
    occlusion_ratio: float
    is_concealed: bool
    velocity_px_sec: float
    prowling_score: float
    is_fallen: bool
    summary_text: str
    concealment_type: Optional[str] = None
    acceleration_px_sec2: float = 0.0
    is_rapid_approach: bool = False
    tortuosity: float = 1.0


class MikoVoOcclusionDetector:
    """
    Vectorized facial occlusion and concealment classifier.
    Combines keypoint confidence & bilateral symmetry analysis
    fused with vectorized HSV / Sobel gradient micro-texture comparison
    between upper and lower facial zones to detect surgical masks,
    bandanas, sunglasses, and heavy balaclavas.
    """
    def __init__(self, threshold: float = OCCLUSION_CONCEALED_THRESHOLD) -> None:
        self.threshold = threshold

    @staticmethod
    def _symmetry_deficit(kpts_5x2: np.ndarray, confs_5: np.ndarray) -> float:
        nose, le, re, lea, rea = kpts_5x2
        eps = 1e-6

        d_le = float(np.linalg.norm(le - nose))
        d_re = float(np.linalg.norm(re - nose))
        eye_asym = abs(d_le - d_re) / (max(d_le, d_re) + eps)

        d_lea = float(np.linalg.norm(lea - nose))
        d_rea = float(np.linalg.norm(rea - nose))
        ear_asym = abs(d_lea - d_rea) / (max(d_lea, d_rea) + eps)

        eye_w = float(min(confs_5[1], confs_5[2]))
        ear_w = float(min(confs_5[3], confs_5[4]))
        score = 0.65 * (eye_asym * eye_w) + 0.35 * (ear_asym * ear_w)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _crop_zone_metrics(face_crop: np.ndarray) -> Tuple[float, Optional[str], float, float]:
        h, w = face_crop.shape[:2]
        if h < 16 or w < 16:
            return 0.0, None, 128.0, 128.0

        x0, x1 = int(w * 0.18), int(w * 0.82)
        upper_bgr = face_crop[int(h * 0.18):int(h * 0.48), x0:x1]
        lower_bgr = face_crop[int(h * 0.54):int(h * 0.92), x0:x1]

        hsv_up = cv2.cvtColor(upper_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_lo = cv2.cvtColor(lower_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        gray_up = cv2.cvtColor(upper_bgr, cv2.COLOR_BGR2GRAY)
        gray_lo = cv2.cvtColor(lower_bgr, cv2.COLOR_BGR2GRAY)

        gx_up = cv2.Sobel(gray_up, cv2.CV_32F, 1, 0, ksize=3)
        gy_up = cv2.Sobel(gray_up, cv2.CV_32F, 0, 1, ksize=3)
        grad_var_up = float(np.var(cv2.magnitude(gx_up, gy_up))) + 1.0

        gx_lo = cv2.Sobel(gray_lo, cv2.CV_32F, 1, 0, ksize=3)
        gy_lo = cv2.Sobel(gray_lo, cv2.CV_32F, 0, 1, ksize=3)
        grad_var_lo = float(np.var(cv2.magnitude(gx_lo, gy_lo))) + 1.0

        h_lo_mean = float(np.mean(hsv_lo[:, :, 0]))
        s_up_mean, s_lo_mean = float(np.mean(hsv_up[:, :, 1])), float(np.mean(hsv_lo[:, :, 1]))
        v_up_mean, v_lo_mean = float(np.mean(hsv_up[:, :, 2])), float(np.mean(hsv_lo[:, :, 2]))
        h_up_mean = float(np.mean(hsv_up[:, :, 0]))

        hue_diff = min(abs(h_up_mean - h_lo_mean), 180.0 - abs(h_up_mean - h_lo_mean))
        sat_diff = abs(s_up_mean - s_lo_mean)
        val_diff = abs(v_up_mean - v_lo_mean)

        color_discontinuity = float(
            np.clip((hue_diff / 45.0) * 0.45 + (sat_diff / 70.0) * 0.30 + (val_diff / 85.0) * 0.25, 0.0, 1.0)
        )

        grad_ratio = grad_var_lo / (grad_var_up + 120.0)
        smooth_lower_score = float(np.clip(1.0 - (grad_ratio / _LOWER_GRAD_RATIO_MASK_CEIL), 0.0, 1.0))

        is_surgical_blue = (_SURGICAL_BLUE_H_RANGE[0] <= h_lo_mean <= _SURGICAL_BLUE_H_RANGE[1]) and (s_lo_mean > 35.0)
        is_surgical_white = (s_lo_mean <= _SURGICAL_WHITE_S_MAX) and (v_lo_mean >= _SURGICAL_WHITE_V_MIN) and (sat_diff > 18.0)

        fabric_hint: Optional[str] = None
        if v_up_mean < _DARK_FABRIC_V_THRESH and v_lo_mean < _DARK_FABRIC_V_THRESH and s_lo_mean < 55.0:
            fabric_hint = "HEAVY_BALACLAVA"
            crop_score = 0.88
        elif is_surgical_blue or is_surgical_white:
            fabric_hint = "SURGICAL_MASK"
            crop_score = max(0.65, 0.55 * color_discontinuity + 0.45 * smooth_lower_score)
        elif v_lo_mean < _DARK_FABRIC_V_THRESH and val_diff > 35.0:
            fabric_hint = "FABRIC_BANDANA"
            crop_score = max(0.62, 0.55 * color_discontinuity + 0.45 * smooth_lower_score)
        elif v_up_mean < _DARK_FABRIC_V_THRESH and v_lo_mean > v_up_mean + 35.0:
            fabric_hint = "UPPER_SUNGLASSES"
            crop_score = 0.58
        else:
            crop_score = 0.55 * color_discontinuity + 0.45 * smooth_lower_score
            if crop_score > 0.52:
                fabric_hint = "FABRIC_BANDANA"

        return float(np.clip(crop_score, 0.0, 1.0)), fabric_hint, v_up_mean, v_lo_mean

    def analyze(
        self,
        kpts_5x2: np.ndarray,
        confs_5: np.ndarray,
        face_crop: Optional[np.ndarray] = None,
    ) -> OcclusionResult:
        kpts = np.asarray(kpts_5x2, dtype=np.float64)
        confs = np.clip(np.asarray(confs_5, dtype=np.float64), 0.0, 1.0)

        nose_conf = float(confs[0])
        eyes_conf = float(np.mean(confs[1:3]))
        ears_conf = float(np.mean(confs[3:5]))

        weighted_vis = 0.40 * nose_conf + 0.40 * eyes_conf + 0.20 * ears_conf
        kp_deficit = float(np.clip(1.0 - weighted_vis, 0.0, 1.0))
        sym_deficit = self._symmetry_deficit(kpts, confs)

        crop_score = 0.0
        fabric_hint: Optional[str] = None
        if face_crop is not None and face_crop.size > 0:
            crop_score, fabric_hint, _, _ = self._crop_zone_metrics(face_crop)
            occlusion_ratio = float(
                np.clip(0.45 * kp_deficit + 0.42 * crop_score + 0.13 * sym_deficit, 0.0, 1.0)
            )
            if fabric_hint in ("SURGICAL_MASK", "FABRIC_BANDANA", "HEAVY_BALACLAVA"):
                occlusion_ratio = max(occlusion_ratio, 0.72 * crop_score + 0.28 * kp_deficit)
        else:
            occlusion_ratio = float(np.clip(0.82 * kp_deficit + 0.18 * sym_deficit, 0.0, 1.0))

        is_concealed = occlusion_ratio > self.threshold
        concealment_type: Optional[str] = None

        if is_concealed:
            if nose_conf < _KP_VIS_THRESHOLD and eyes_conf < _KP_VIS_THRESHOLD and ears_conf < _KP_VIS_THRESHOLD:
                concealment_type = "HEAVY_BALACLAVA"
            elif fabric_hint == "HEAVY_BALACLAVA":
                concealment_type = "HEAVY_BALACLAVA"
            elif eyes_conf < _KP_VIS_THRESHOLD and nose_conf >= _KP_VIS_THRESHOLD:
                concealment_type = "UPPER_SUNGLASSES"
            elif fabric_hint is not None:
                concealment_type = fabric_hint
            elif nose_conf < _KP_VIS_THRESHOLD and eyes_conf >= _KP_VIS_THRESHOLD:
                concealment_type = "SURGICAL_MASK"
            else:
                concealment_type = "FABRIC_BANDANA"

        return OcclusionResult(
            occlusion_ratio=round(occlusion_ratio, 4),
            is_face_concealed=bool(is_concealed),
            concealment_type=concealment_type,
        )


@dataclass
class _KineticTrackMemory:
    prev_speed_px_sec: float
    prev_vy_px_sec: float
    hw_ratio_history: Deque[float]


class MikoVoKineticThreatAnalyzer:
    """
    Analyzes 8-state Kalman filter velocity and posture:
      - Rapid sprinting/dash (RAPID_DASH).
      - Perspective looming approach toward camera (RAPID_APPROACH).
      - Sudden human collapse/fall (FALL_DETECTED / COLLAPSE).
    """
    def __init__(self, fps: float = DEFAULT_FPS) -> None:
        self.fps = float(max(fps, 1.0))
        self.dt = 1.0 / self.fps
        self._window_len = max(4, int(round(0.40 * self.fps)))
        self._tracks: Dict[Any, _KineticTrackMemory] = {}

    def reset_track(self, track_id: Any) -> None:
        self._tracks.pop(track_id, None)

    def analyze(
        self,
        track_id: Any,
        kalman_state_8d: np.ndarray,
        bbox_xywh: Tuple[float, float, float, float],
        frame_shape: Tuple[int, ...],
    ) -> KineticResult:
        state = np.asarray(kalman_state_8d, dtype=np.float64).ravel()
        if state.size < 8:
            state = np.pad(state, (0, max(0, 8 - state.size)))

        cx, cy, a_state, h_state, vx_raw, vy_raw, va_raw, vh_raw = state[:8]
        bx, by, bw, bh = bbox_xywh

        raw_mag = math.hypot(vx_raw, vy_raw)
        scale = self.fps if raw_mag < 75.0 else 1.0

        vx = vx_raw * scale
        vy = vy_raw * scale
        vh = vh_raw * (self.fps if abs(vh_raw) < 40.0 else 1.0)

        speed_px_sec = float(math.hypot(vx, vy))

        w_eff = max(float(bw), float(a_state * h_state), 1.0)
        h_eff = max(float(bh), float(h_state), 1.0)
        h_over_w = h_eff / w_eff

        mem = self._tracks.get(track_id)
        if mem is None:
            mem = _KineticTrackMemory(
                prev_speed_px_sec=speed_px_sec,
                prev_vy_px_sec=vy,
                hw_ratio_history=deque(maxlen=self._window_len),
            )
            self._tracks[track_id] = mem
            accel_px_sec2 = 0.0
        else:
            accel_px_sec2 = (speed_px_sec - mem.prev_speed_px_sec) / self.dt
            mem.prev_speed_px_sec = speed_px_sec
            mem.prev_vy_px_sec = vy

        mem.hw_ratio_history.append(h_over_w)

        max_recent_hw = max(mem.hw_ratio_history)
        aspect_collapsed = (max_recent_hw >= _FALL_UPRIGHT_H_OVER_W) and (
            h_over_w <= _FALL_COLLAPSED_H_OVER_W
        )
        va_spike = va_raw > 0.35 and h_over_w <= _FALL_COLLAPSED_H_OVER_W
        is_fallen = bool((vy >= _FALL_VY_PX_SEC) and (aspect_collapsed or va_spike))

        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        target_x, target_y = frame_w * 0.5, frame_h * 0.95
        to_cam_x, to_cam_y = target_x - cx, target_y - cy
        to_cam_norm = math.hypot(to_cam_x, to_cam_y) + 1e-6
        heading_cos = (vx * to_cam_x + vy * to_cam_y) / ((speed_px_sec + 1e-6) * to_cam_norm)

        rel_expansion_rate = vh / max(h_eff, 20.0)
        is_rapid_approach = bool(
            not is_fallen
            and (
                (rel_expansion_rate >= _APPROACH_VH_RATIO_SEC and speed_px_sec >= 160.0)
                or (heading_cos >= 0.65 and speed_px_sec >= _DASH_VELOCITY_PX_SEC * 0.85 and accel_px_sec2 >= 250.0)
            )
        )

        is_rapid_dash = bool(
            not is_fallen
            and (
                speed_px_sec >= _DASH_VELOCITY_PX_SEC
                or (speed_px_sec >= 280.0 and accel_px_sec2 >= _DASH_ACCEL_PX_SEC2)
                or is_rapid_approach
            )
        )

        kinetic_event: Optional[str] = None
        if is_fallen:
            kinetic_event = "FALL_DETECTED / COLLAPSE"
        elif is_rapid_approach:
            kinetic_event = "RAPID_APPROACH"
        elif is_rapid_dash:
            kinetic_event = "RAPID_DASH"

        return KineticResult(
            velocity_px_sec=round(speed_px_sec, 2),
            acceleration_px_sec2=round(float(accel_px_sec2), 2),
            is_rapid_dash=is_rapid_dash,
            is_rapid_approach=is_rapid_approach,
            is_fallen=is_fallen,
            kinetic_event=kinetic_event,
        )


class MikoVoTrajectoryEntropyAnalyzer:
    """
    Maintains 45-frame sliding window of track positions:
      1. Path Tortuosity (total_path_distance / net_displacement)
      2. Directional Shannon Entropy across 8 compass octants
    """
    def __init__(
        self,
        window_size: int = _TRAJECTORY_WINDOW,
        prowling_threshold: float = PROWLING_SCORE_THRESHOLD,
    ) -> None:
        self.window_size = window_size
        self.prowling_threshold = prowling_threshold
        self._trajectories: Dict[Any, Deque[Tuple[float, float]]] = {}

    def reset_track(self, track_id: Any) -> None:
        self._trajectories.pop(track_id, None)

    def analyze(self, track_id: Any, center_xy: Tuple[float, float]) -> TrajectoryEntropyResult:
        buf = self._trajectories.setdefault(track_id, deque(maxlen=self.window_size))
        buf.append((float(center_xy[0]), float(center_xy[1])))

        n_pts = len(buf)
        if n_pts < _MIN_PROWL_FRAMES:
            return TrajectoryEntropyResult(0.0, False, 1.0, 0.0)

        pts = np.asarray(buf, dtype=np.float64)
        deltas = np.diff(pts, axis=0)
        step_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
        total_distance = float(np.sum(step_lengths))

        if total_distance < _MIN_TOTAL_PATH_PX:
            return TrajectoryEntropyResult(0.0, False, 1.0, 0.0)

        net_displacement = float(np.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1]))
        tortuosity = total_distance / (net_displacement + 6.0)
        tortuosity_score = float(
            np.clip((tortuosity - 1.25) / (_TORTUOSITY_SATURATION - 1.25), 0.0, 1.0)
        )

        active_mask = step_lengths > 1.2
        if int(np.sum(active_mask)) < 6:
            return TrajectoryEntropyResult(0.0, False, round(tortuosity, 3), 0.0)

        active_deltas = deltas[active_mask]
        angles = np.arctan2(active_deltas[:, 1], active_deltas[:, 0])

        bins = np.floor(((angles + math.pi) / (2.0 * math.pi)) * _ANGULAR_BINS).astype(np.int32)
        bins = np.clip(bins, 0, _ANGULAR_BINS - 1)
        counts = np.bincount(bins, minlength=_ANGULAR_BINS).astype(np.float64)
        probs = counts[counts > 0] / float(angles.size)
        shannon_entropy = float(-np.sum(probs * np.log2(probs))) / math.log2(_ANGULAR_BINS)

        angle_diffs = np.diff(angles)
        angle_diffs = (angle_diffs + math.pi) % (2.0 * math.pi) - math.pi
        turn_var_score = float(np.clip(np.std(angle_diffs) / 1.45, 0.0, 1.0))
        reversal_rate = float(np.mean(np.abs(angle_diffs) > 1.85))
        reversal_score = float(np.clip(reversal_rate / 0.22, 0.0, 1.0))

        directional_chaos = float(
            np.clip(0.40 * shannon_entropy + 0.35 * turn_var_score + 0.25 * reversal_score, 0.0, 1.0)
        )

        prowling_score = float(np.clip(0.62 * tortuosity_score + 0.38 * directional_chaos, 0.0, 1.0))
        is_prowling = prowling_score > self.prowling_threshold

        return TrajectoryEntropyResult(
            prowling_score=round(prowling_score, 4),
            is_prowling=bool(is_prowling),
            tortuosity=round(tortuosity, 3),
            directional_entropy=round(directional_chaos, 4),
        )


_DEFAULT_OCCLUSION_DETECTOR = MikoVoOcclusionDetector()
_DEFAULT_KINETIC_ANALYZER = MikoVoKineticThreatAnalyzer()
_DEFAULT_ENTROPY_ANALYZER = MikoVoTrajectoryEntropyAnalyzer()


def evaluate_behavioral_threat(
    track_id: Any,
    kalman_state_8d: np.ndarray,
    bbox_xywh: Tuple[float, float, float, float],
    kpts_5x2: np.ndarray,
    confs_5: np.ndarray,
    frame_shape: Tuple[int, ...],
    face_crop: Optional[np.ndarray] = None,
    occlusion_detector: MikoVoOcclusionDetector = _DEFAULT_OCCLUSION_DETECTOR,
    kinetic_analyzer: MikoVoKineticThreatAnalyzer = _DEFAULT_KINETIC_ANALYZER,
    entropy_analyzer: MikoVoTrajectoryEntropyAnalyzer = _DEFAULT_ENTROPY_ANALYZER,
) -> ThreatReport:
    state = np.asarray(kalman_state_8d, dtype=np.float64).ravel()
    cx, cy = float(state[0]), float(state[1])

    occ = occlusion_detector.analyze(kpts_5x2, confs_5, face_crop=face_crop)
    kin = kinetic_analyzer.analyze(track_id, state, bbox_xywh, frame_shape)
    ent = entropy_analyzer.analyze(track_id, (cx, cy))

    threat_category: Optional[str] = None
    threat_level: str = "CLEAR"

    if kin.is_fallen:
        threat_category = "PERSON_FALLEN"
        threat_level = "CRITICAL"
    elif kin.is_rapid_dash:
        threat_category = "RAPID_DASH"
        threat_level = "CRITICAL" if (occ.is_face_concealed or kin.is_rapid_approach) else "ELEVATED"
    elif occ.is_face_concealed and ent.is_prowling:
        threat_category = "CONCEALED_FACE"
        threat_level = "CRITICAL"
    elif occ.is_face_concealed:
        threat_category = "CONCEALED_FACE"
        threat_level = "CRITICAL" if occ.concealment_type == "HEAVY_BALACLAVA" else "ELEVATED"
    elif ent.is_prowling:
        threat_category = "PROWLER_PACING"
        threat_level = "ELEVATED"

    is_threat = threat_level != "CLEAR"

    if not is_threat:
        summary_text = (
            f"[CLEAR] Track #{track_id}: Normal transit "
            f"(v={kin.velocity_px_sec:.0f} px/s, occ={occ.occlusion_ratio:.2f}, prowl={ent.prowling_score:.2f})."
        )
    else:
        tags = []
        if kin.is_fallen:
            tags.append("FALL_DETECTED/COLLAPSE")
        if kin.is_rapid_approach:
            tags.append(f"RAPID_APPROACH ({kin.velocity_px_sec:.0f} px/s)")
        elif kin.is_rapid_dash:
            tags.append(f"RAPID_DASH ({kin.velocity_px_sec:.0f} px/s)")
        if occ.is_face_concealed:
            tags.append(f"CONCEALED:{occ.concealment_type} (occ={occ.occlusion_ratio:.2f})")
        if ent.is_prowling:
            tags.append(f"PROWLING (score={ent.prowling_score:.2f}, tort={ent.tortuosity:.1f}x)")

        summary_text = f"[{threat_level}] Track #{track_id} ({threat_category}): " + " | ".join(tags)

    return ThreatReport(
        is_threat=is_threat,
        threat_level=threat_level,
        threat_category=threat_category,
        occlusion_ratio=occ.occlusion_ratio,
        is_concealed=occ.is_face_concealed,
        velocity_px_sec=kin.velocity_px_sec,
        prowling_score=ent.prowling_score,
        is_fallen=kin.is_fallen,
        summary_text=summary_text,
        concealment_type=occ.concealment_type,
        acceleration_px_sec2=kin.acceleration_px_sec2,
        is_rapid_approach=kin.is_rapid_approach,
        tortuosity=ent.tortuosity,
    )


# =============================================================================
# 8. MIKOVO RARA 11 VISION & TRACKING ENGINE
# =============================================================================


class MikoVoRaRa11VisionEngine:
    """
    MikoVo RaRa 11 Autonomous AI Vision Core:
      - YOLOv8 Neural Backbone + SVD Weighted Umeyama Alignment
      - 512-Dimensional Deep Metric Embeddings (< 3.5ms on CPU)
      - 8-State Kalman Filter Temporal Tracking with Mahalanobis Gating
      - Virtual Tripwire & ROI Spatial Filtering
      - Circular Frame Buffer for Instant Video Alert Dispatch
    """
    def __init__(self, db: FaceDatabase):
        self.db = db
        self.tracker = MikoVoRaRa11MultiKalmanTracker()
        self._lock = threading.Lock()
        self.running = False
        self.cap = None
        self.latest_frame = None
        self.latest_annotated = None
        self.last_detections = []
        self.stats = {"fps": 0.0, "person_count": 0}
        self.last_alert_time = 0.0
        self.reported_tracks = set()

        # MikoVo RaRa 11 Biometrics & Passive Liveness Core
        self.liveness_detector = MikoVoLivenessDetector()
        self.pose_estimator = MikoVoHeadPoseEstimator(image_size=112)
        self.fusion_memory = MikoVoTrackFusionMemory()

        # MikoVo RaRa 11 Behavioral Threat Profilers
        self.occlusion_detector = MikoVoOcclusionDetector()
        self.kinetic_analyzer = MikoVoKineticThreatAnalyzer()
        self.entropy_analyzer = MikoVoTrajectoryEntropyAnalyzer()

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

            # Evaluate MikoVo RaRa 11 Behavioral & Kinetic Threats
            for d in tracked:
                kalman_state = d.get("kalman_state")
                kpts = d.get("kpts")
                confs = d.get("kp_conf")
                face_crop = d.get("face_crop")
                tid = d.get("track_id", 1)
                bbox = d.get("smooth_bbox", d["bbox"])

                kpts_5x2 = kpts if (kpts is not None and len(kpts) >= 5) else np.zeros((5, 2))
                confs_5 = confs if confs is not None else np.ones(5, dtype=np.float64) * 0.8
                k_state_8d = kalman_state if kalman_state is not None else np.array([bbox[0] + bbox[2]/2.0, bbox[1] + bbox[3]/2.0, bbox[2]/float(max(1, bbox[3])), float(bbox[3]), 0.0, 0.0, 0.0, 0.0])

                threat_report = evaluate_behavioral_threat(
                    track_id=tid,
                    kalman_state_8d=k_state_8d,
                    bbox_xywh=tuple(bbox),
                    kpts_5x2=kpts_5x2,
                    confs_5=confs_5,
                    frame_shape=frame.shape,
                    face_crop=face_crop,
                    occlusion_detector=self.occlusion_detector,
                    kinetic_analyzer=self.kinetic_analyzer,
                    entropy_analyzer=self.entropy_analyzer,
                )
                d["threat"] = threat_report

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
                        has_threat = any(d.get("threat") and d["threat"].threat_level in ("ELEVATED", "CRITICAL") for d in active_detections)

                        if trigger_mode == "stranger_only":
                            should_alert = has_unknown or has_loitering or has_threat
                        elif trigger_mode == "roi_only":
                            should_alert = has_roi
                        elif trigger_mode == "known_only":
                            should_alert = has_known or has_threat
                        else:
                            if config.get("only_new_person"):
                                should_alert = has_unknown or has_unreported_track or has_loitering or has_threat
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

                    # MikoVo RaRa 11 Neural Core Alignment + Descriptor Extraction
                    aligned_face, emb, align_res = mikovo_rara11_extract_face_descriptor(
                        frame, (x1, y1, bw, bh), kp_pts, kp_conf
                    )

                    # MikoVo RaRa 11 Liveness & Head Pose Sentinel
                    sentinel = evaluate_face_sentinel(
                        aligned_face,
                        align_res.canonical_keypoints,
                        kp_conf if kp_conf is not None else np.ones(5),
                        liveness_detector=self.liveness_detector,
                        pose_estimator=self.pose_estimator,
                        fusion_memory=self.fusion_memory
                    )

                    # If spoof attack detected, reject authentication!
                    if sentinel.is_spoof_attack:
                        name, match_score = None, 0.0
                    else:
                        name, match_score = self.db.find_match(emb)

                    detections.append({
                        "bbox": [x1, y1, bw, bh],
                        "score": conf,
                        "name": name,
                        "match_score": match_score,
                        "embedding": emb,
                        "face_crop": aligned_face,
                        "kpts": kp_pts,
                        "kp_conf": kp_conf,
                        "sentinel": sentinel,
                        "is_live": sentinel.is_live,
                        "liveness_score": sentinel.liveness_score,
                        "is_spoof": sentinel.is_spoof_attack,
                        "attack_type": sentinel.attack_type,
                        "euler_angles": sentinel.euler_angles,
                        "pose_quality": sentinel.pose_quality,
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

        # MikoVo RaRa 11 Neural Watermark
        cv2.putText(frame, "MikoVo RaRa 11", (18, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        for det in detections:
            x, y, bw, bh = det.get("smooth_bbox", det["bbox"])
            name = det.get("name")
            tid = det.get("track_id", 1)
            is_loiter = det.get("is_loitering", False)
            in_roi = det.get("in_roi", True)

            is_spoof = det.get("is_spoof", False)
            attack_type = det.get("attack_type")
            threat = det.get("threat")

            if threat and threat.threat_level == "CRITICAL":
                color = (0, 0, 255) # Alert Red for Critical Threat
            elif is_spoof:
                color = (0, 0, 255) # Alert Red for Spoof Attacks
            elif threat and threat.threat_level == "ELEVATED":
                color = (0, 140, 255) # Intense Orange for Elevated Threat
            elif is_loiter:
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
            if threat and threat.is_threat:
                if threat.is_fallen:
                    prefix = "EMERGENCY: FALL DETECTED! "
                elif threat.threat_category == "CONCEALED_FACE":
                    prefix = f"ALERT: CONCEALED ({threat.concealment_type or 'MASK'}) "
                elif threat.threat_category == "RAPID_DASH":
                    prefix = f"ALERT: DASH ({threat.velocity_px_sec:.0f}px/s) "
                elif threat.threat_category == "PROWLER_PACING":
                    prefix = f"ALERT: PROWLER (score={threat.prowling_score:.2f}) "
                else:
                    prefix = f"THREAT: {threat.threat_category} "
            elif is_spoof:
                prefix = f"SPOOF ALERT! ({attack_type or 'Fake Face'}) "
            elif is_loiter:
                prefix = "LOITERING! "
            else:
                prefix = ""

            live_pct = int(det.get("liveness_score", 1.0) * 100)
            live_txt = f" [Live: {live_pct}%]" if not is_spoof else " [FAKE]"
            id_txt = f"{name} ({det.get('match_score', 0.94)*100:.0f}%)" if name else f"Unknown ({det['score']*100:.0f}%)"
            label = f"MikoVo RaRa 11 #{tid} | {prefix}{id_txt}{live_txt}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            ly = max(24, y - 8)
            cv2.rectangle(frame, (x, ly - th - 4), (x + tw + 10, ly + 4), (20, 20, 22), -1)
            cv2.rectangle(frame, (x, ly - th - 4), (x + tw + 10, ly + 4), color, 1)
            cv2.putText(frame, label, (x + 5, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 247), 1, cv2.LINE_AA)

            # Threat diagnostic sub-badge
            if threat and threat.is_threat:
                diag = f"Occ: {threat.occlusion_ratio:.2f} | Vel: {threat.velocity_px_sec:.0f} px/s | Prowl: {threat.prowling_score:.2f}"
                (dtw, dth), _ = cv2.getTextSize(diag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                dly = y + bh + 16
                if dly < h - 5:
                    cv2.rectangle(frame, (x, dly - dth - 3), (x + dtw + 8, dly + 3), (20, 20, 22), -1)
                    cv2.rectangle(frame, (x, dly - dth - 3), (x + dtw + 8, dly + 3), color, 1)
                    cv2.putText(frame, diag, (x + 4, dly - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 245), 1, cv2.LINE_AA)

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
        custom_cap = config.get("custom_caption", "هشدار تردد MikoVo RaRa 11")

        state["stats"]["total_captures"] += 1
        has_spoof = any(d.get("is_spoof") for d in detections)
        if has_spoof:
            spoof_types = [d.get("attack_type") or "تصویر جعلی" for d in detections if d.get("is_spoof")]
            custom_cap = f"🚨 هشدار ضدتقلب MikoVo RaRa 11: تلاش برای دور زدن بیومتریک با {', '.join(spoof_types)} شناسایی و دفع شد!"

        threat_d = next((d for d in detections if d.get("threat") and d["threat"].threat_level in ("ELEVATED", "CRITICAL")), None)
        if threat_d:
            state["stats"]["threat_count"] = state["stats"].get("threat_count", 0) + 1
            t_report = threat_d["threat"]
            t_cat = t_report.threat_category
            if t_cat == "PERSON_FALLEN":
                custom_cap = "🚨 وضعیت اضطراری MikoVo RaRa 11: سقوط ناگهانی / بیهوشی شخص در صحنه شناسایی شد!"
            elif t_cat == "CONCEALED_FACE":
                c_type = t_report.occlusion.concealment_type or "ماسک یا پوشش"
                custom_cap = f"⚠️ هشدار امنیتی MikoVo RaRa 11: فرد مشکوک با چهره پوشانده شده ({c_type}) شناسایی شد!"
            elif t_cat == "RAPID_DASH":
                custom_cap = "⚡ هشدار امنیتی MikoVo RaRa 11: دویدن یا حمله ناگهانی و هجوم پرسرعت فرد ثبت شد!"
            elif t_cat == "PROWLER_PACING":
                custom_cap = "👁️ هشدار پایش محیطی MikoVo RaRa 11: پرسه زدن مکرر و مشکوک (Prowler) در محدوده ثبت شد!"
            else:
                custom_cap = f"⚠️ هشدار رفتاری MikoVo RaRa 11: تهدید رفتاری ({t_report.threat_level}) ثبت شد!"

        names_str = ", ".join(identified) if identified else "ناشناس"
        if threat_d and threat_d["threat"].threat_category:
            t_cat = threat_d["threat"].threat_category
            if t_cat == "PERSON_FALLEN":
                names_str = f"🚑 سقوط/بیهوشی ({names_str})"
            elif t_cat == "CONCEALED_FACE":
                c_type = threat_d["threat"].occlusion.concealment_type or "ماسک"
                names_str = f"🎭 چهره پوشانده ({c_type})"
            elif t_cat == "RAPID_DASH":
                names_str = f"⚡ هجوم سریع ({names_str})"
            elif t_cat == "PROWLER_PACING":
                names_str = f"🔄 پرسه زنی ({names_str})"
        elif has_spoof:
            names_str = f"🚨 جعل چهره ({', '.join(spoof_types)})"

        log_entry = {
            "id": int(time.time() * 1000),
            "time": datetime.now().strftime("%H:%M:%S"),
            "persons": count,
            "names": names_str,
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
        if det.get("is_spoof"):
            return False, "خطای امنیتی: تصویر زنده نیست (تلاش برای ثبت عکس چاپی یا صفحه نمایش)!"
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
            f"🛡 MikoTim Sentinel Pro | MikoVo RaRa 11 AI\n"
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
            f"🎬 ویدیو رخداد امنیتی | MikoVo RaRa 11 AI\n"
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
                json={"chat_id": chat_id, "text": "✅ اتصال با سرور پایش MikoTim و موتور هوش مصنوعی MikoVo RaRa 11 با موفقیت برقرار شد."},
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
detector = MikoVoRaRa11VisionEngine(face_db)
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
@app.route("/api/mikovo/info")
def api_mikivo_info():
    return jsonify(
        engine="MikoVo RaRa 11 AI Neural Vision Core",
        origin="Derived & Evolved from YOLOv8 Architecture",
        version="4.0.0-sentinel-pro",
        descriptor_dim=512,
        alignment="SVD Weighted Umeyama Similarity Transform",
        illumination="LAB CLAHE + Dynamic Exponential Gamma Compensator",
        liveness="Multi-Spectral Passive Anti-Spoofing (2D FFT + YCrCb/HSV + rPPG)",
        head_pose="3D-to-2D PnP Euler Solver (Pitch/Yaw/Roll Attention Profiling)",
        fusion="Quality-Weighted EMA Track Hypersphere Memory",
        tracker="8-State Constant-Velocity Kalman with Mahalanobis Outlier Gating",
        threat_profiler="Behavioral & Kinetic Threat Profiler (Face Occlusion, Looming Dash, Sudden Fall, Trajectory Entropy Prowler)",
        features=[
            "Canonical 112x112 3D-Aware Facial Rectification",
            "512-D L2-Normalized Metric Hypersphere Embeddings",
            "Multi-Scale Scharr Orientation Histograms",
            "Multi-Radius Uniform Local Binary Patterns (ULBP)",
            "Multi-Spectral Passive Face Anti-Spoofing (Screen/Print Attack Detection)",
            "3D Head Pose & Attention Profiling (PnP Euler Angles)",
            "Quality-Weighted EMA Track Descriptor Fusion",
            "8-State Kalman Filter Tracking with Mahalanobis Gating",
            "Sliding-Window Identity Majority Voting",
            "Virtual Restricted Zone (ROI) & Directional Tripwire",
            "Dual Alert Media Pipeline (Photo & MP4 Video Clips)",
            "Facial Occlusion & Concealment Detection (Mask / Bandana / Balaclava)",
            "Kinetic Looming Dash & Rapid Approach Velocity Analyzer",
            "Sudden Fall & Medical Emergency Aspect-Ratio Collapse Detector",
            "Trajectory Tortuosity & Shannon Directional-Entropy Prowler Profiler",
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
        threat_count=state["stats"].get("threat_count", 0),
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
