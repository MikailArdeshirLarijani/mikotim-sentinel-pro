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
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import math
from typing import Optional, Tuple, List, Dict, Any, Deque, Hashable
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
MIKOVO_WEIGHTS = BASE_DIR / "mikovo_rara11.pt"
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
    "vision_mode": "server_vision",   # "server_vision" (Mode B) or "client_edge" (Mode A)
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
    "vision_mode": config.get("vision_mode", "server_vision"),
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
    "client_detections": [],
    "client_fps": 0.0,
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
        self.kpts_history: deque = deque(maxlen=25)
        self.bbox_history: deque = deque(maxlen=25)

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
                det_kpts = det.get("full_kpts")
                if det_kpts is not None:
                    trk.kpts_history.append(det_kpts)
                else:
                    trk.kpts_history.append(np.zeros((17, 3), dtype=np.float64))
                trk.bbox_history.append(smooth_box)

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
            smooth_box = [int(v) for v in trk.get_bbox_xywh()]
            det_kpts = det.get("full_kpts")
            if det_kpts is not None:
                trk.kpts_history.append(det_kpts)
            else:
                trk.kpts_history.append(np.zeros((17, 3), dtype=np.float64))
            trk.bbox_history.append(smooth_box)
            self.trackers[tid] = trk
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
# 7.2. MIKOVO RARA 11 PHYSICAL VIOLENCE, STRIKE & GRAPPLE SENTINEL
# =============================================================================
# =============================================================================
# 0. COCO 17-KEYPOINT INDICES & CALIBRATION CONSTANTS
# =============================================================================

# Standard YOLOv8-pose / COCO 17-keypoint layout:
# 0:nose, 1:l_eye, 2:r_eye, 3:l_ear, 4:r_ear,
# 5:l_shoulder, 6:r_shoulder, 7:l_elbow, 8:r_elbow,
# 9:l_wrist, 10:r_wrist, 11:l_hip, 12:r_hip,
# 13:l_knee, 14:r_knee, 15:l_ankle, 16:r_ankle
KP_NOSE: int = 0
KP_L_EYE: int = 1
KP_R_EYE: int = 2
KP_L_EAR: int = 3
KP_R_EAR: int = 4
KP_L_SHOULDER: int = 5
KP_R_SHOULDER: int = 6
KP_L_ELBOW: int = 7
KP_R_ELBOW: int = 8
KP_L_WRIST: int = 9
KP_R_WRIST: int = 10
KP_L_HIP: int = 11
KP_R_HIP: int = 12
KP_L_KNEE: int = 13
KP_R_KNEE: int = 14
KP_L_ANKLE: int = 15
KP_R_ANKLE: int = 16

DEFAULT_FPS: float = 30.0
MIN_KP_CONF: float = 0.30

# Subsystem 1: Strike Velocity & Jerk Thresholds
_STRIKE_WRIST_VEL_PX_SEC: float = 520.0         # Absolute wrist speed threshold (px/s)
_STRIKE_EXTENSION_VEL_PX_SEC: float = 340.0     # Outward wrist-from-shoulder radial speed (px/s)
_STRIKE_JERK_PX_SEC3: float = 18000.0           # 3rd-derivative kinematic jerk ceiling (px/s^3)
_STRIKE_ANGULAR_ACCEL_RAD_SEC2: float = 85.0    # Elbow snap angular acceleration (rad/s^2)

# Subsystem 2: Pairwise Interlock / Grapple Thresholds
_GRAPPLE_IOU_THRESHOLD: float = 0.22            # Bounding-box intersection-over-union
_GRAPPLE_OVERLAP_RATIO_THRESH: float = 0.38     # Intersection over smaller bbox area
_GRAPPLE_CLOSING_SPEED_PX_SEC: float = 160.0    # Mutual closing rate (px/s)
_GRAPPLE_REL_ACCEL_PX_SEC2: float = 380.0       # Relative convergence acceleration (px/s^2)

# Subsystem 3: Chaotic Struggle Entropy Thresholds
_STRUGGLE_WINDOW: int = 20                      # 20-frame sliding window
_STRUGGLE_MIN_FRAMES: int = 6
_JITTER_RMS_SATURATION_PX: float = 14.0         # High-pass centroid residual RMS (px)
_TURBULENCE_ACCEL_STD_PX_SEC2: float = 550.0    # Acceleration standard deviation ceiling
_VIOLENCE_SCORE_THRESHOLD: float = 0.52         # Unified aggression trigger threshold


# =============================================================================
# 1. REQUIRED DATACLASSES
# =============================================================================

@dataclass
class StrikeEvent:
    """Represents a detected limb strike (punch, slap, elbow, melee swing)."""
    is_strike: bool
    limb: Optional[str]                 # "LEFT_ARM" | "RIGHT_ARM" | "DUAL_ARMS" | None
    velocity_px_sec: float
    confidence: float
    # Diagnostic metadata
    track_id: Optional[Any] = None
    jerk_px_sec3: float = 0.0
    angular_accel_rad_sec2: float = 0.0


@dataclass
class GrappleEvent:
    """Represents a multi-person proxemic collapse / physical grapple state."""
    is_grappling: bool
    person_ids: Tuple[Any, Any]
    overlap_ratio: float
    relative_accel: float
    # Diagnostic metadata
    closing_speed_px_sec: float = 0.0
    iou: float = 0.0


@dataclass
class AggressionReport:
    """
    Unified output of `evaluate_aggression_dynamics`.

    Core required attributes:
        is_violent_threat : bool
        threat_level      : str ("CLEAR", "ELEVATED", "CRITICAL")
        threat_type       : Optional[str] ("MELEE_STRIKE", "PHYSICAL_GRAPPLE",
                            "CHAOTIC_BRAWL", None)
        aggression_score  : float in [0.0, 1.0]
        summary_text      : str (operator dispatch summary)
    """
    is_violent_threat: bool
    threat_level: str
    threat_type: Optional[str]
    aggression_score: float
    summary_text: str
    # Extended telemetry for downstream fusion
    top_strike: Optional[StrikeEvent] = None
    top_grapple: Optional[GrappleEvent] = None
    struggle_entropy_score: float = 0.0


# =============================================================================
# 2. INPUT NORMALIZATION HELPERS
# =============================================================================

@dataclass
class _NormalizedTrackSequence:
    """Internal standardized representation of a single track's temporal history."""
    track_id: Any
    bboxes: np.ndarray          # (T, 4) float64 [x, y, w, h]
    keypoints: np.ndarray       # (T, 17, 3) float64 [x, y, conf]


def _standardize_keypoints_array(kpts: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    """
    Coerce an input keypoint history of shape `(T, K, C)` into a canonical
    `(T, 17, 3)` array `[x, y, conf]`. Supports:
      - Standard `(T, 17, 3)` YOLOv8-pose keypoints.
      - `(T, 17, 2)` keypoints (assigns default confidence 0.85 where non-zero).
      - Compact 9-point or 11-point upper-body subsets `[nose, le, re, lea, rea,
        l_sh, r_sh, l_wr, r_wr, (l_hip, r_hip)]`.
      - 5-point head keypoints `(T, 5, 2|3)` (synthesizes approximate shoulder/hip
        anchors from bbox geometry with low confidence so centroid tracking works).
    """
    arr = np.asarray(kpts, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    T, K, C = arr.shape
    out = np.zeros((T, 17, 3), dtype=np.float64)

    if C == 2:
        valid = (np.abs(arr[:, :, 0]) + np.abs(arr[:, :, 1])) > 1e-3
        conf = np.where(valid, 0.85, 0.0)[:, :, None]
        arr = np.concatenate([arr, conf], axis=2)

    if K >= 17:
        out[:, :17, :] = arr[:, :17, :3]
    elif K == 11:
        # [0..4: head, 5..6: shoulders, 7..8: wrists, 9..10: hips]
        out[:, :7, :] = arr[:, :7, :3]
        out[:, KP_L_WRIST, :] = arr[:, 7, :3]
        out[:, KP_R_WRIST, :] = arr[:, 8, :3]
        out[:, KP_L_HIP, :] = arr[:, 9, :3]
        out[:, KP_R_HIP, :] = arr[:, 10, :3]
    elif K == 9:
        # [0..4: head, 5..6: shoulders, 7..8: wrists]
        out[:, :7, :] = arr[:, :7, :3]
        out[:, KP_L_WRIST, :] = arr[:, 7, :3]
        out[:, KP_R_WRIST, :] = arr[:, 8, :3]
    else:
        k_copy = min(K, 17)
        out[:, :k_copy, :] = arr[:, :k_copy, :3]
        # Populate approximate shoulder centers from bbox for upper-body centroid stability
        bx, by, bw, bh = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        out[:, KP_L_SHOULDER, 0] = bx + 0.30 * bw
        out[:, KP_L_SHOULDER, 1] = by + 0.24 * bh
        out[:, KP_L_SHOULDER, 2] = 0.40
        out[:, KP_R_SHOULDER, 0] = bx + 0.70 * bw
        out[:, KP_R_SHOULDER, 1] = by + 0.24 * bh
        out[:, KP_R_SHOULDER, 2] = 0.40

    return out


def _parse_tracks_history(tracks_history: Any) -> List[_NormalizedTrackSequence]:
    """
    Flexibly parse `tracks_history` passed to `evaluate_aggression_dynamics`.
    """
    normalized: List[_NormalizedTrackSequence] = []
    if not tracks_history:
        return normalized

    items: List[Tuple[Any, Any]] = []
    if isinstance(tracks_history, Mapping):
        items = list(tracks_history.items())
    elif isinstance(tracks_history, Sequence):
        for idx, entry in enumerate(tracks_history):
            if isinstance(entry, Mapping):
                tid = entry.get("track_id", entry.get("id", idx))
                items.append((tid, entry))

    for track_id, data in items:
        if isinstance(data, Mapping):
            raw_bboxes = data.get("bboxes", data.get("bbox"))
            raw_kpts = data.get("keypoints", data.get("kpts"))
            if raw_bboxes is None or raw_kpts is None:
                continue
            bboxes = np.asarray(raw_bboxes, dtype=np.float64)
            if bboxes.ndim == 1:
                bboxes = bboxes[None, :]
            kpts = _standardize_keypoints_array(np.asarray(raw_kpts, dtype=np.float64), bboxes)
            T = min(bboxes.shape[0], kpts.shape[0])
            if T > 0:
                normalized.append(_NormalizedTrackSequence(track_id, bboxes[-T:], kpts[-T:]))
        elif isinstance(data, Sequence) and len(data) > 0 and isinstance(data[0], Mapping):
            bboxes_list = [frame["bbox"] for frame in data if "bbox" in frame and "keypoints" in frame]
            kpts_list = [frame["keypoints"] for frame in data if "bbox" in frame and "keypoints" in frame]
            if not bboxes_list:
                continue
            bboxes = np.asarray(bboxes_list, dtype=np.float64)
            kpts = _standardize_keypoints_array(np.asarray(kpts_list, dtype=np.float64), bboxes)
            normalized.append(_NormalizedTrackSequence(track_id, bboxes, kpts))

    return normalized


# =============================================================================
# 3. SUBSYSTEM 1: STRIKE VELOCITY & KINEMATIC JERK ANALYZER
# =============================================================================

class MikoVoStrikeVelocityAnalyzer:
    """
    Analyzes upper-limb (shoulder -> elbow -> wrist) kinematic chains across
    consecutive frames to detect rapid punches, slaps, and melee strikes.
    """

    def __init__(
        self,
        wrist_vel_threshold: float = _STRIKE_WRIST_VEL_PX_SEC,
        extension_vel_threshold: float = _STRIKE_EXTENSION_VEL_PX_SEC,
        jerk_threshold: float = _STRIKE_JERK_PX_SEC3,
        angular_accel_threshold: float = _STRIKE_ANGULAR_ACCEL_RAD_SEC2,
    ) -> None:
        self.wrist_vel_threshold = wrist_vel_threshold
        self.extension_vel_threshold = extension_vel_threshold
        self.jerk_threshold = jerk_threshold
        self.angular_accel_threshold = angular_accel_threshold

    def _analyze_single_arm(
        self,
        sh_xyz: np.ndarray,
        el_xyz: np.ndarray,
        wr_xyz: np.ndarray,
        dt: float,
    ) -> Tuple[float, float, float, float, float]:
        T = wr_xyz.shape[0]
        if T < 3:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        sh_conf = sh_xyz[:, 2]
        el_conf = el_xyz[:, 2]
        wr_conf = wr_xyz[:, 2]

        valid_sw = (sh_conf >= MIN_KP_CONF) & (wr_conf >= MIN_KP_CONF)
        if int(np.sum(valid_sw[-min(T, 6):])) < 3:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        rel_wr = wr_xyz[:, :2] - sh_xyz[:, :2]
        rel_vel = np.diff(rel_wr, axis=0) / dt
        wrist_speeds = np.hypot(rel_vel[:, 0], rel_vel[:, 1])

        radial_dist = np.hypot(rel_wr[:, 0], rel_wr[:, 1])
        extension_vel = np.diff(radial_dist) / dt

        pair_valid = valid_sw[:-1] & valid_sw[1:]
        if not np.any(pair_valid):
            return 0.0, 0.0, 0.0, 0.0, 0.0

        masked_speeds = np.where(pair_valid, wrist_speeds, 0.0)
        masked_ext_vel = np.where(pair_valid, np.maximum(0.0, extension_vel), 0.0)

        peak_wrist_vel = float(np.max(masked_speeds[-min(len(masked_speeds), 6):]))
        peak_ext_vel = float(np.max(masked_ext_vel[-min(len(masked_ext_vel), 6):]))

        peak_jerk = 0.0
        if T >= 4:
            rel_acc = np.diff(rel_vel, axis=0) / dt
            rel_jerk = np.diff(rel_acc, axis=0) / dt
            jerk_mag = np.hypot(rel_jerk[:, 0], rel_jerk[:, 1])
            jerk_valid = valid_sw[:-3] & valid_sw[1:-2] & valid_sw[2:-1] & valid_sw[3:]
            if np.any(jerk_valid):
                peak_jerk = float(np.max(np.where(jerk_valid, jerk_mag, 0.0)[-min(len(jerk_mag), 5):]))

        peak_ang_accel = 0.0
        valid_all = valid_sw & (el_conf >= MIN_KP_CONF)
        if T >= 4 and int(np.sum(valid_all)) >= 4:
            v_upper = sh_xyz[:, :2] - el_xyz[:, :2]
            v_fore = wr_xyz[:, :2] - el_xyz[:, :2]
            dot = np.sum(v_upper * v_fore, axis=1)
            norm_prod = (np.hypot(v_upper[:, 0], v_upper[:, 1]) * np.hypot(v_fore[:, 0], v_fore[:, 1])) + 1e-6
            cos_theta = np.clip(dot / norm_prod, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            ang_vel = np.diff(theta) / dt
            ang_acc = np.abs(np.diff(ang_vel) / dt)
            ang_valid = valid_all[:-2] & valid_all[1:-1] & valid_all[2:]
            if np.any(ang_valid):
                peak_ang_accel = float(np.max(np.where(ang_valid, ang_acc, 0.0)[-min(len(ang_acc), 5):]))

        vel_score = float(np.clip(peak_wrist_vel / (self.wrist_vel_threshold * 1.35), 0.0, 1.0))
        ext_score = float(np.clip(peak_ext_vel / (self.extension_vel_threshold * 1.25), 0.0, 1.0))
        jerk_score = float(np.clip(peak_jerk / (self.jerk_threshold * 1.40), 0.0, 1.0))
        ang_score = float(np.clip(peak_ang_accel / (self.angular_accel_threshold * 1.35), 0.0, 1.0))

        if peak_ang_accel > 0.0:
            strike_score = 0.38 * vel_score + 0.28 * ext_score + 0.20 * jerk_score + 0.14 * ang_score
        else:
            strike_score = 0.44 * vel_score + 0.34 * ext_score + 0.22 * jerk_score

        mean_conf = float(np.mean(wr_conf[valid_sw][-min(int(np.sum(valid_sw)), 5):]))
        return float(np.clip(strike_score, 0.0, 1.0)), peak_wrist_vel, peak_jerk, peak_ang_accel, mean_conf

    def analyze_track(
        self,
        kpts_seq: np.ndarray,
        fps: float = DEFAULT_FPS,
        track_id: Optional[Any] = None,
    ) -> StrikeEvent:
        dt = 1.0 / max(float(fps), 1.0)
        kpts = np.asarray(kpts_seq, dtype=np.float64)
        if kpts.ndim != 3 or kpts.shape[0] < 3:
            return StrikeEvent(False, None, 0.0, 0.0, track_id=track_id)

        l_score, l_vel, l_jerk, l_ang, l_conf = self._analyze_single_arm(
            kpts[:, KP_L_SHOULDER], kpts[:, KP_L_ELBOW], kpts[:, KP_L_WRIST], dt
        )
        r_score, r_vel, r_jerk, r_ang, r_conf = self._analyze_single_arm(
            kpts[:, KP_R_SHOULDER], kpts[:, KP_R_ELBOW], kpts[:, KP_R_WRIST], dt
        )

        l_trig = (l_vel >= self.wrist_vel_threshold and l_score >= 0.52) or (l_score >= 0.62)
        r_trig = (r_vel >= self.wrist_vel_threshold and r_score >= 0.52) or (r_score >= 0.62)

        if l_trig and r_trig:
            limb = "DUAL_ARMS"
            best_vel = max(l_vel, r_vel)
            best_jerk = max(l_jerk, r_jerk)
            best_ang = max(l_ang, r_ang)
            conf = float(np.clip(max(l_score, r_score) * max(l_conf, r_conf), 0.0, 1.0))
            is_strike = True
        elif l_score >= r_score:
            limb = "LEFT_ARM" if l_trig else None
            best_vel, best_jerk, best_ang = l_vel, l_jerk, l_ang
            conf = float(np.clip(l_score * max(l_conf, 0.5), 0.0, 1.0))
            is_strike = bool(l_trig)
        else:
            limb = "RIGHT_ARM" if r_trig else None
            best_vel, best_jerk, best_ang = r_vel, r_jerk, r_ang
            conf = float(np.clip(r_score * max(r_conf, 0.5), 0.0, 1.0))
            is_strike = bool(r_trig)

        return StrikeEvent(
            is_strike=is_strike,
            limb=limb,
            velocity_px_sec=round(best_vel, 2),
            confidence=round(conf, 4),
            track_id=track_id,
            jerk_px_sec3=round(best_jerk, 1),
            angular_accel_rad_sec2=round(best_ang, 2),
        )


# =============================================================================
# 4. SUBSYSTEM 2: PAIRWISE INTERLOCK & GRAPPLE ANALYZER
# =============================================================================

class MikoVoPairwiseInterlockAnalyzer:
    """
    Analyzes multi-person proxemic collapse across all active pairs `(i, j)`.
    """

    def __init__(
        self,
        overlap_threshold: float = _GRAPPLE_OVERLAP_RATIO_THRESH,
        closing_speed_threshold: float = _GRAPPLE_CLOSING_SPEED_PX_SEC,
        rel_accel_threshold: float = _GRAPPLE_REL_ACCEL_PX_SEC2,
    ) -> None:
        self.overlap_threshold = overlap_threshold
        self.closing_speed_threshold = closing_speed_threshold
        self.rel_accel_threshold = rel_accel_threshold

    @staticmethod
    def _pairwise_overlap_series(b1: np.ndarray, b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x1_min, y1_min = b1[:, 0], b1[:, 1]
        x1_max, y1_max = x1_min + np.maximum(b1[:, 2], 1.0), y1_min + np.maximum(b1[:, 3], 1.0)

        x2_min, y2_min = b2[:, 0], b2[:, 1]
        x2_max, y2_max = x2_min + np.maximum(b2[:, 2], 1.0), y2_min + np.maximum(b2[:, 3], 1.0)

        inter_w = np.maximum(0.0, np.minimum(x1_max, x2_max) - np.maximum(x1_min, x2_min))
        inter_h = np.maximum(0.0, np.minimum(y1_max, y2_max) - np.maximum(y1_min, y2_min))
        inter_area = inter_w * inter_h

        area1 = np.maximum(b1[:, 2], 1.0) * np.maximum(b1[:, 3], 1.0)
        area2 = np.maximum(b2[:, 2], 1.0) * np.maximum(b2[:, 3], 1.0)
        union_area = area1 + area2 - inter_area + 1e-6
        min_area = np.minimum(area1, area2) + 1e-6

        iou = inter_area / union_area
        overlap_ratio = inter_area / min_area
        return iou, overlap_ratio

    def analyze_tracks(
        self,
        tracks: Sequence[_NormalizedTrackSequence],
        fps: float = DEFAULT_FPS,
    ) -> List[GrappleEvent]:
        n_tracks = len(tracks)
        if n_tracks < 2:
            return []

        dt = 1.0 / max(float(fps), 1.0)
        events: List[GrappleEvent] = []

        for i in range(n_tracks):
            for j in range(i + 1, n_tracks):
                tr_a, tr_b = tracks[i], tracks[j]
                T = min(tr_a.bboxes.shape[0], tr_b.bboxes.shape[0], 15)
                if T < 2:
                    continue

                ba = tr_a.bboxes[-T:]
                bb = tr_b.bboxes[-T:]

                iou_series, ov_series = self._pairwise_overlap_series(ba, bb)
                curr_overlap = float(np.max(ov_series[-min(T, 3):]))
                curr_iou = float(np.max(iou_series[-min(T, 3):]))
                overlap_surge = float(curr_overlap - ov_series[0])

                ca = np.column_stack([ba[:, 0] + 0.5 * ba[:, 2], ba[:, 1] + 0.5 * ba[:, 3]])
                cb = np.column_stack([bb[:, 0] + 0.5 * bb[:, 2], bb[:, 1] + 0.5 * bb[:, 3]])
                dist_series = np.hypot(ca[:, 0] - cb[:, 0], ca[:, 1] - cb[:, 1])

                closing_speeds = -np.diff(dist_series) / dt
                peak_closing_speed = float(np.max(np.maximum(0.0, closing_speeds)))

                va = np.diff(ca, axis=0) / dt
                vb = np.diff(cb, axis=0) / dt
                rel_vel = va - vb

                if T >= 3:
                    rel_acc = np.diff(rel_vel, axis=0) / dt
                    rel_acc_mag = np.hypot(rel_acc[:, 0], rel_acc[:, 1])
                    peak_rel_accel = float(np.max(rel_acc_mag))
                else:
                    peak_rel_accel = float(np.hypot(rel_vel[-1, 0], rel_vel[-1, 1]) / dt)

                is_grappling = bool(
                    (
                        curr_overlap >= self.overlap_threshold
                        and (
                            peak_closing_speed >= self.closing_speed_threshold
                            or peak_rel_accel >= self.rel_accel_threshold
                            or overlap_surge >= 0.20
                        )
                    )
                    or (curr_iou >= _GRAPPLE_IOU_THRESHOLD * 1.45 and peak_rel_accel >= self.rel_accel_threshold * 0.75)
                )

                events.append(
                    GrappleEvent(
                        is_grappling=is_grappling,
                        person_ids=(tr_a.track_id, tr_b.track_id),
                        overlap_ratio=round(curr_overlap, 4),
                        relative_accel=round(peak_rel_accel, 2),
                        closing_speed_px_sec=round(peak_closing_speed, 2),
                        iou=round(curr_iou, 4),
                    )
                )

        events.sort(
            key=lambda e: (e.is_grappling, e.overlap_ratio * 0.6 + min(e.relative_accel / 1000.0, 1.0) * 0.4),
            reverse=True,
        )
        return events


# =============================================================================
# 5. SUBSYSTEM 3: CHAOTIC STRUGGLE ENTROPY ANALYZER
# =============================================================================

class MikoVoChaoticStruggleEntropy:
    """
    Measures kinetic turbulent energy and high-frequency spatial jitter of
    upper-body centroids across a 20-frame sliding window.
    """

    def __init__(self, window_size: int = _STRUGGLE_WINDOW) -> None:
        self.window_size = window_size

    @staticmethod
    def _extract_upper_body_centroids(track: _NormalizedTrackSequence, window: int) -> np.ndarray:
        bboxes = track.bboxes[-window:]
        kpts = track.keypoints[-window:]
        T = bboxes.shape[0]

        torso_indices = [KP_NOSE, KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP]
        sub_kpts = kpts[:, torso_indices, :2]
        sub_confs = kpts[:, torso_indices, 2]
        sub_confs = np.where(sub_confs >= MIN_KP_CONF, sub_confs, 0.0)

        conf_sum = np.sum(sub_confs, axis=1, keepdims=True)
        weighted_xy = np.sum(sub_kpts * sub_confs[:, :, None], axis=1) / np.maximum(conf_sum, 1e-6)

        bbox_upper_xy = np.column_stack([
            bboxes[:, 0] + 0.50 * bboxes[:, 2],
            bboxes[:, 1] + 0.35 * bboxes[:, 3],
        ])

        has_kpts = (conf_sum[:, 0] > 0.35)[:, None]
        centroids = np.where(has_kpts, weighted_xy, bbox_upper_xy)
        return centroids

    def analyze_track(self, track: _NormalizedTrackSequence, fps: float = DEFAULT_FPS) -> float:
        T = min(track.bboxes.shape[0], self.window_size)
        if T < _STRUGGLE_MIN_FRAMES:
            return 0.0

        dt = 1.0 / max(float(fps), 1.0)
        centroids = self._extract_upper_body_centroids(track, self.window_size)

        kernel = np.array([0.25, 0.50, 0.25], dtype=np.float64)
        smooth_x = np.convolve(centroids[:, 0], kernel, mode="valid")
        smooth_y = np.convolve(centroids[:, 1], kernel, mode="valid")
        res_x = centroids[1:-1, 0] - smooth_x
        res_y = centroids[1:-1, 1] - smooth_y
        jitter_rms = float(np.sqrt(np.mean(res_x ** 2 + res_y ** 2)))
        jitter_score = float(np.clip(jitter_rms / _JITTER_RMS_SATURATION_PX, 0.0, 1.0))

        vel = np.diff(centroids, axis=0) / dt
        speeds = np.hypot(vel[:, 0], vel[:, 1])
        if float(np.max(speeds)) < 25.0:
            return 0.0

        v1, v2 = vel[:-1], vel[1:]
        dot = np.sum(v1 * v2, axis=1)
        denom = (np.hypot(v1[:, 0], v1[:, 1]) * np.hypot(v2[:, 0], v2[:, 1])) + 1e-6
        cos_angles = dot / denom
        active_steps = (speeds[:-1] > 18.0) & (speeds[1:] > 18.0)
        if np.any(active_steps):
            reversals = float(np.mean((cos_angles[active_steps] < -0.10).astype(np.float64)))
        else:
            reversals = 0.0
        reversal_score = float(np.clip(reversals / 0.42, 0.0, 1.0))

        acc = np.diff(vel, axis=0) / dt
        acc_mag = np.hypot(acc[:, 0], acc[:, 1])
        turb_std = float(np.std(acc_mag))
        turb_score = float(np.clip(turb_std / _TURBULENCE_ACCEL_STD_PX_SEC2, 0.0, 1.0))

        struggle_score = 0.42 * jitter_score + 0.33 * reversal_score + 0.25 * turb_score
        return float(np.clip(struggle_score, 0.0, 1.0))


# =============================================================================
# 6. TOP-LEVEL FACADE FUNCTION
# =============================================================================

_DEFAULT_STRIKE_ANALYZER = MikoVoStrikeVelocityAnalyzer()
_DEFAULT_INTERLOCK_ANALYZER = MikoVoPairwiseInterlockAnalyzer()
_DEFAULT_ENTROPY_ANALYZER = MikoVoChaoticStruggleEntropy()


def evaluate_aggression_dynamics(
    tracks_history: Any,
    current_frame_shape: Tuple[int, ...],
    fps: float = DEFAULT_FPS,
    strike_analyzer: MikoVoStrikeVelocityAnalyzer = _DEFAULT_STRIKE_ANALYZER,
    interlock_analyzer: MikoVoPairwiseInterlockAnalyzer = _DEFAULT_INTERLOCK_ANALYZER,
    entropy_analyzer: MikoVoChaoticStruggleEntropy = _DEFAULT_ENTROPY_ANALYZER,
) -> AggressionReport:
    _ = current_frame_shape
    tracks = _parse_tracks_history(tracks_history)

    if not tracks:
        return AggressionReport(
            is_violent_threat=False,
            threat_level="CLEAR",
            threat_type=None,
            aggression_score=0.0,
            summary_text="[CLEAR] No active tracks in scene.",
        )

    strike_events: List[StrikeEvent] = []
    struggle_scores: List[float] = []

    for tr in tracks:
        s_ev = strike_analyzer.analyze_track(tr.keypoints, fps=fps, track_id=tr.track_id)
        strike_events.append(s_ev)
        struggle_scores.append(entropy_analyzer.analyze_track(tr, fps=fps))

    strike_events.sort(key=lambda e: (e.is_strike, e.confidence, e.velocity_px_sec), reverse=True)
    top_strike = strike_events[0] if strike_events else None
    max_struggle = float(max(struggle_scores)) if struggle_scores else 0.0

    grapple_events = interlock_analyzer.analyze_tracks(tracks, fps=fps)
    top_grapple = grapple_events[0] if grapple_events else None

    strike_component = top_strike.confidence if top_strike is not None else 0.0
    if top_strike is not None and top_strike.is_strike:
        strike_component = max(strike_component, 0.72)

    grapple_component = 0.0
    if top_grapple is not None:
        grapple_component = float(
            np.clip(
                0.55 * (top_grapple.overlap_ratio / _GRAPPLE_OVERLAP_RATIO_THRESH)
                + 0.45 * (top_grapple.relative_accel / (_GRAPPLE_REL_ACCEL_PX_SEC2 * 1.5)),
                0.0,
                1.0,
            )
        )
        if top_grapple.is_grappling:
            grapple_component = max(grapple_component, 0.74)

    peak_signal = max(strike_component, grapple_component, max_struggle)
    synergy_boost = 0.15 * min(strike_component, max_struggle) + 0.18 * min(grapple_component, max_struggle)
    aggression_score = float(np.clip(peak_signal + synergy_boost, 0.0, 1.0))

    has_strike = bool(top_strike is not None and top_strike.is_strike)
    has_grapple = bool(top_grapple is not None and top_grapple.is_grappling)
    has_struggle = max_struggle >= 0.60

    is_violent_threat = bool(
        has_strike or has_grapple or has_struggle or (aggression_score >= _VIOLENCE_SCORE_THRESHOLD)
    )

    threat_type: Optional[str] = None
    threat_level: str = "CLEAR"

    if is_violent_threat:
        if (has_grapple and has_strike) or (has_grapple and has_struggle) or (has_strike and has_struggle):
            threat_type = "CHAOTIC_BRAWL"
            threat_level = "CRITICAL"
        elif has_grapple:
            threat_type = "PHYSICAL_GRAPPLE"
            threat_level = "CRITICAL" if aggression_score >= 0.78 else "ELEVATED"
        elif has_strike:
            threat_type = "MELEE_STRIKE"
            threat_level = "CRITICAL" if (top_strike and top_strike.velocity_px_sec >= 720.0) else "ELEVATED"
        else:
            threat_type = "CHAOTIC_BRAWL"
            threat_level = "ELEVATED"

    if not is_violent_threat:
        summary_text = (
            f"[CLEAR] Scene nominal across {len(tracks)} track(s) "
            f"(aggression={aggression_score:.2f}, struggle_entropy={max_struggle:.2f})."
        )
    else:
        details: List[str] = []
        if has_strike and top_strike is not None:
            details.append(
                f"STRIKE[Track #{top_strike.track_id} {top_strike.limb} "
                f"v={top_strike.velocity_px_sec:.0f}px/s conf={top_strike.confidence:.2f}]"
            )
        if has_grapple and top_grapple is not None:
            p1, p2 = top_grapple.person_ids
            details.append(
                f"GRAPPLE[Tracks #{p1}<->#{p2} ov={top_grapple.overlap_ratio:.2f} "
                f"rel_acc={top_grapple.relative_accel:.0f}px/s^2]"
            )
        if max_struggle >= 0.45:
            details.append(f"TURBULENCE[entropy={max_struggle:.2f}]")

        summary_text = (
            f"[{threat_level}] {threat_type} (score={aggression_score:.2f}): "
            + " | ".join(details)
        )

    return AggressionReport(
        is_violent_threat=is_violent_threat,
        threat_level=threat_level,
        threat_type=threat_type,
        aggression_score=round(aggression_score, 4),
        summary_text=summary_text,
        top_strike=top_strike,
        top_grapple=top_grapple,
        struggle_entropy_score=round(max_struggle, 4),
    )


# =============================================================================
# 8. MIKOVO RARA 11 SHRINK-WRAPPED ANATOMICAL BODY & HEAD FITTER
# =============================================================================

# Anthropometric head proportions relative to Inter-Ocular Distance (IOD)
_HEAD_WIDTH_PER_IOD: float = 2.60
_HEAD_HEIGHT_PER_IOD: float = 3.40
_EYE_LINE_FROM_CROWN_FRAC: float = 0.42
_HEAD_WIDTH_PER_EAR_SPAN: float = 1.08

_SEGMENT_HALF_WIDTH_RATIOS: Dict[str, float] = {
    "torso": 0.60,
    "upper_arm": 0.22,
    "forearm": 0.18,
    "thigh": 0.32,
    "calf": 0.23,
}

_KNEE_STRAIGHT_DEG: float = 155.0
_KNEE_BENT_SITTING_DEG: Tuple[float, float] = (65.0, 125.0)
_KNEE_BENT_CROUCH_DEG: float = 95.0
_HIP_BENT_CROUCH_DEG: float = 110.0
_TORSO_LEAN_DEG: float = 22.0
_ARM_EXTEND_RATIO: float = 1.35

_EMA_ALPHA_MIN: float = 0.25
_EMA_ALPHA_MAX: float = 0.92
_EMA_JUMP_NORM_SATURATION: float = 0.55

_SHRINKWRAP_SEARCH_FRAC: float = 0.14
_SHRINKWRAP_DOWNSAMPLE_MAX_DIM: int = 160
_SHRINKWRAP_MIN_ROI_PX: int = 18


@dataclass
class AnatomicalBoundsReport:
    """
    Unified output of `fit_anatomical_person_bounds`.
    """
    body_bbox: Tuple[float, float, float, float]
    head_bbox: Tuple[float, float, float, float]
    confidence: float
    posture_type: str
    tight_aspect_ratio: float


def _clip_box_to_frame(
    box: Tuple[float, float, float, float], frame_shape: Optional[Tuple[int, ...]]
) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    if frame_shape is None:
        return (max(0.0, x), max(0.0, y), max(w, 1.0), max(h, 1.0))

    frame_h, frame_w = frame_shape[0], frame_shape[1]
    x0 = float(np.clip(x, 0.0, max(frame_w - 1.0, 0.0)))
    y0 = float(np.clip(y, 0.0, max(frame_h - 1.0, 0.0)))
    x1 = float(np.clip(x + w, 1.0, float(frame_w)))
    y1 = float(np.clip(y + h, 1.0, float(frame_h)))
    return (x0, y0, max(x1 - x0, 1.0), max(y1 - y0, 1.0))


def _bbox_from_points(points: np.ndarray) -> Tuple[float, float, float, float]:
    x_min, y_min = float(np.min(points[:, 0])), float(np.min(points[:, 1]))
    x_max, y_max = float(np.max(points[:, 0])), float(np.max(points[:, 1]))
    return (x_min, y_min, max(x_max - x_min, 1.0), max(y_max - y_min, 1.0))


def _segment_square_points(p1: np.ndarray, p2: np.ndarray, half_width: float) -> np.ndarray:
    offsets = np.array([[half_width, half_width], [half_width, -half_width],
                         [-half_width, half_width], [-half_width, -half_width]])
    return np.vstack([p1 + offsets, p2 + offsets])


class MikoVoSkeletonFittedBox:
    """
    Reconstructs a tight anatomical bounding rectangle directly from COCO
    17-keypoint body geometry using Dempster-style anthropometric volumetric
    bone-segment projection (head oval, torso capsule, limb cylinders).
    """

    def __init__(self, min_kp_conf: float = MIN_KP_CONF) -> None:
        self.min_kp_conf = min_kp_conf

    @staticmethod
    def _compute_reference_scale(kpts: np.ndarray, raw_bbox: Tuple[float, float, float, float]) -> float:
        l_sh, r_sh = kpts[KP_L_SHOULDER], kpts[KP_R_SHOULDER]
        if l_sh[2] >= MIN_KP_CONF and r_sh[2] >= MIN_KP_CONF:
            d = float(np.linalg.norm(l_sh[:2] - r_sh[:2]))
            if d > 4.0:
                return d

        l_hip, r_hip = kpts[KP_L_HIP], kpts[KP_R_HIP]
        if l_hip[2] >= MIN_KP_CONF and r_hip[2] >= MIN_KP_CONF:
            d = float(np.linalg.norm(l_hip[:2] - r_hip[:2])) * 1.08
            if d > 4.0:
                return d

        return max(raw_bbox[2] * 0.42, 10.0)

    @staticmethod
    def _estimate_head_box(kpts: np.ndarray, reference_scale: float) -> Optional[Tuple[float, float, float, float]]:
        nose, l_eye, r_eye = kpts[KP_NOSE], kpts[KP_L_EYE], kpts[KP_R_EYE]
        l_ear, r_ear = kpts[KP_L_EAR], kpts[KP_R_EAR]

        eyes_valid = l_eye[2] >= MIN_KP_CONF and r_eye[2] >= MIN_KP_CONF
        iod = float(np.linalg.norm(l_eye[:2] - r_eye[:2])) if eyes_valid else 0.0

        if eyes_valid and iod > 3.0:
            eye_center = (l_eye[:2] + r_eye[:2]) / 2.0
            head_width_iod = iod * _HEAD_WIDTH_PER_IOD
            head_height = iod * _HEAD_HEIGHT_PER_IOD
        elif nose[2] >= MIN_KP_CONF:
            eye_center = nose[:2].copy()
            head_width_iod = reference_scale * 0.78
            head_height = reference_scale * 1.02
        else:
            return None

        ears_valid = l_ear[2] >= MIN_KP_CONF and r_ear[2] >= MIN_KP_CONF
        if ears_valid:
            ear_span = float(np.linalg.norm(l_ear[:2] - r_ear[:2])) * _HEAD_WIDTH_PER_EAR_SPAN
            head_width = (head_width_iod + ear_span) / 2.0 if eyes_valid else ear_span
        else:
            head_width = head_width_iod

        crown_y = eye_center[1] - _EYE_LINE_FROM_CROWN_FRAC * head_height
        chin_y = crown_y + head_height
        left_x = eye_center[0] - head_width / 2.0
        right_x = eye_center[0] + head_width / 2.0

        return (float(left_x), float(crown_y), float(right_x - left_x), float(chin_y - crown_y))

    def fit(
        self,
        keypoints_17x3: np.ndarray,
        raw_bbox: Tuple[float, float, float, float],
    ) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], float]:
        kpts = np.asarray(keypoints_17x3, dtype=np.float64)
        if kpts.shape != (17, 3):
            raise ValueError(f"Expected keypoints shape (17, 3), got {kpts.shape}")

        reference_scale = self._compute_reference_scale(kpts, raw_bbox)
        all_points = []
        used_confs = []

        # --- Torso capsule: shoulder-mid -> hip-mid ---
        l_sh, r_sh = kpts[KP_L_SHOULDER], kpts[KP_R_SHOULDER]
        l_hip, r_hip = kpts[KP_L_HIP], kpts[KP_R_HIP]
        shoulder_mid = None
        hip_mid = None
        if l_sh[2] >= MIN_KP_CONF or r_sh[2] >= MIN_KP_CONF:
            valid = [p for p in (l_sh, r_sh) if p[2] >= MIN_KP_CONF]
            shoulder_mid = np.mean([p[:2] for p in valid], axis=0)
            used_confs.extend([p[2] for p in valid])
        if l_hip[2] >= MIN_KP_CONF or r_hip[2] >= MIN_KP_CONF:
            valid = [p for p in (l_hip, r_hip) if p[2] >= MIN_KP_CONF]
            hip_mid = np.mean([p[:2] for p in valid], axis=0)
            used_confs.extend([p[2] for p in valid])

        if shoulder_mid is not None and hip_mid is not None:
            hw = reference_scale * _SEGMENT_HALF_WIDTH_RATIOS["torso"] / 2.0
            all_points.append(_segment_square_points(shoulder_mid, hip_mid, hw))
            if l_sh[2] >= MIN_KP_CONF and r_sh[2] >= MIN_KP_CONF:
                all_points.append(np.vstack([l_sh[:2], r_sh[:2]]))
            if l_hip[2] >= MIN_KP_CONF and r_hip[2] >= MIN_KP_CONF:
                all_points.append(np.vstack([l_hip[:2], r_hip[:2]]))

        # --- Limb segments ---
        limb_segments = (
            (KP_L_SHOULDER, KP_L_ELBOW, "upper_arm"),
            (KP_R_SHOULDER, KP_R_ELBOW, "upper_arm"),
            (KP_L_ELBOW, KP_L_WRIST, "forearm"),
            (KP_R_ELBOW, KP_R_WRIST, "forearm"),
            (KP_L_HIP, KP_L_KNEE, "thigh"),
            (KP_R_HIP, KP_R_KNEE, "thigh"),
            (KP_L_KNEE, KP_L_ANKLE, "calf"),
            (KP_R_KNEE, KP_R_ANKLE, "calf"),
        )
        for idx_a, idx_b, seg_key in limb_segments:
            pa, pb = kpts[idx_a], kpts[idx_b]
            if pa[2] >= MIN_KP_CONF and pb[2] >= MIN_KP_CONF:
                hw = reference_scale * _SEGMENT_HALF_WIDTH_RATIOS[seg_key] / 2.0
                all_points.append(_segment_square_points(pa[:2], pb[:2], hw))
                used_confs.extend([pa[2], pb[2]])

        # --- Head oval ---
        head_box = self._estimate_head_box(kpts, reference_scale)
        if head_box is not None:
            hx, hy, hw_, hh_ = head_box
            all_points.append(np.array([[hx, hy], [hx + hw_, hy + hh_]]))
            used_confs.append(float(max(kpts[KP_NOSE, 2], kpts[KP_L_EYE, 2], kpts[KP_R_EYE, 2])))
        else:
            rx, ry, rw, rh = raw_bbox
            head_box = (rx + rw * 0.28, ry, rw * 0.44, rh * 0.22)

        if not all_points:
            return raw_bbox, head_box, 0.05

        stacked = np.vstack(all_points)
        body_box = _bbox_from_points(stacked)
        confidence = float(np.clip(np.mean(used_confs) if used_confs else 0.1, 0.0, 1.0))

        return body_box, head_box, confidence


class MikoVoSilhouetteShrinkWrap:
    """
    Refines a skeleton-fitted bounding box by snapping its borders onto the
    true outer silhouette/clothing edges using Sobel gradient + Otsu threshold.
    """

    def __init__(
        self,
        search_frac: float = _SHRINKWRAP_SEARCH_FRAC,
        downsample_max_dim: int = _SHRINKWRAP_DOWNSAMPLE_MAX_DIM,
    ) -> None:
        self.search_frac = search_frac
        self.downsample_max_dim = downsample_max_dim

    def refine(
        self,
        frame: np.ndarray,
        box: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        frame_h, frame_w = frame.shape[0], frame.shape[1]
        x, y, w, h = box
        margin_x = w * self.search_frac
        margin_y = h * self.search_frac

        rx0 = int(np.clip(x - margin_x, 0, frame_w - 1))
        ry0 = int(np.clip(y - margin_y, 0, frame_h - 1))
        rx1 = int(np.clip(x + w + margin_x, 1, frame_w))
        ry1 = int(np.clip(y + h + margin_y, 1, frame_h))

        roi = frame[ry0:ry1, rx0:rx1]
        if roi.size == 0 or roi.shape[0] < _SHRINKWRAP_MIN_ROI_PX or roi.shape[1] < _SHRINKWRAP_MIN_ROI_PX:
            return box

        roi_h, roi_w = roi.shape[0], roi.shape[1]
        scale = min(1.0, self.downsample_max_dim / max(roi_h, roi_w))
        if scale < 1.0:
            small = cv2.resize(roi, (max(1, int(roi_w * scale)), max(1, int(roi_h * scale))), interpolation=cv2.INTER_AREA)
        else:
            small = roi

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edge_mask = cv2.threshold(mag_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edge_bin = (edge_mask > 0).astype(np.float32)

        s_h, s_w = edge_bin.shape[0], edge_bin.shape[1]

        sx0 = (x - rx0) * scale
        sy0 = (y - ry0) * scale
        sx1 = (x + w - rx0) * scale
        sy1 = (y + h - ry0) * scale
        s_margin_x = margin_x * scale
        s_margin_y = margin_y * scale

        col_density = edge_bin.sum(axis=0)
        row_density = edge_bin.sum(axis=1)

        def _snap_1d(density: np.ndarray, center: float, search_margin: float, length: int) -> float:
            lo = int(np.clip(center - search_margin, 0, length - 1))
            hi = int(np.clip(center + search_margin, 1, length))
            if hi <= lo:
                return center
            band = density[lo:hi]
            if band.size == 0 or float(np.max(band)) < 1e-3:
                return center
            best_idx = lo + int(np.argmax(band))
            return float(best_idx)

        new_sx0 = _snap_1d(col_density, sx0, s_margin_x, s_w)
        new_sx1 = _snap_1d(col_density, sx1, s_margin_x, s_w)
        new_sy0 = _snap_1d(row_density, sy0, s_margin_y, s_h)
        new_sy1 = _snap_1d(row_density, sy1, s_margin_y, s_h)

        if new_sx1 <= new_sx0:
            new_sx0, new_sx1 = sx0, sx1
        if new_sy1 <= new_sy0:
            new_sy0, new_sy1 = sy0, sy1

        inv_scale = 1.0 / scale
        new_x = rx0 + new_sx0 * inv_scale
        new_y = ry0 + new_sy0 * inv_scale
        new_w = (new_sx1 - new_sx0) * inv_scale
        new_h = (new_sy1 - new_sy0) * inv_scale

        if new_w < 4.0 or new_h < 4.0:
            return box
        return (float(new_x), float(new_y), float(new_w), float(new_h))


@dataclass
class _ZoneSmootherState:
    body_box: np.ndarray
    head_box: np.ndarray


class MikoVoDualZoneFitter:
    """
    Produces two locked anatomical zones (`head_bbox`, `body_bbox`) and
    applies an adaptive alpha-beta/EMA temporal smoother per track ID.
    """

    def __init__(
        self,
        skeleton_fitter: Optional[MikoVoSkeletonFittedBox] = None,
        silhouette_refiner: Optional[MikoVoSilhouetteShrinkWrap] = None,
        alpha_min: float = _EMA_ALPHA_MIN,
        alpha_max: float = _EMA_ALPHA_MAX,
    ) -> None:
        self.skeleton_fitter = skeleton_fitter or MikoVoSkeletonFittedBox()
        self.silhouette_refiner = silhouette_refiner or MikoVoSilhouetteShrinkWrap()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self._state: Dict[Any, _ZoneSmootherState] = {}

    @staticmethod
    def _adaptive_alpha(prev_box: np.ndarray, new_box: np.ndarray, alpha_min: float, alpha_max: float) -> float:
        prev_cx, prev_cy = prev_box[0] + prev_box[2] / 2.0, prev_box[1] + prev_box[3] / 2.0
        new_cx, new_cy = new_box[0] + new_box[2] / 2.0, new_box[1] + new_box[3] / 2.0
        center_jump = float(np.hypot(new_cx - prev_cx, new_cy - prev_cy))
        ref_dim = max(float(prev_box[2]), float(prev_box[3]), 1.0)
        normalized_jump = np.clip(center_jump / (ref_dim * _EMA_JUMP_NORM_SATURATION), 0.0, 1.0)
        return float(alpha_min + normalized_jump * (alpha_max - alpha_min))

    def _smooth(self, track_id: Any, body_box: np.ndarray, head_box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        state = self._state.get(track_id)
        if state is None:
            self._state[track_id] = _ZoneSmootherState(body_box=body_box.copy(), head_box=head_box.copy())
            return body_box, head_box

        alpha_body = self._adaptive_alpha(state.body_box, body_box, self.alpha_min, self.alpha_max)
        alpha_head = self._adaptive_alpha(state.head_box, head_box, self.alpha_min, self.alpha_max)

        smoothed_body = alpha_body * body_box + (1.0 - alpha_body) * state.body_box
        smoothed_head = alpha_head * head_box + (1.0 - alpha_head) * state.head_box

        state.body_box = smoothed_body
        state.head_box = smoothed_head
        return smoothed_body, smoothed_head

    def reset_track(self, track_id: Any) -> None:
        self._state.pop(track_id, None)

    def fit(
        self,
        keypoints_17x3: np.ndarray,
        raw_bbox: Tuple[float, float, float, float],
        frame: Optional[np.ndarray] = None,
        track_id: Optional[Any] = None,
        frame_shape: Optional[Tuple[int, ...]] = None,
    ) -> AnatomicalBoundsReport:
        body_box, head_box, confidence = self.skeleton_fitter.fit(keypoints_17x3, raw_bbox)

        if frame is not None and frame.size > 0:
            body_box = self.silhouette_refiner.refine(frame, body_box)
            head_refiner = MikoVoSilhouetteShrinkWrap(search_frac=0.08, downsample_max_dim=96)
            head_box = head_refiner.refine(frame, head_box)

        resolved_frame_shape = frame_shape if frame_shape is not None else (frame.shape if frame is not None else None)

        body_arr = np.array(body_box, dtype=np.float64)
        head_arr = np.array(head_box, dtype=np.float64)

        if track_id is not None:
            body_arr, head_arr = self._smooth(track_id, body_arr, head_arr)

        body_final = _clip_box_to_frame(tuple(body_arr.tolist()), resolved_frame_shape)
        head_final = _clip_box_to_frame(tuple(head_arr.tolist()), resolved_frame_shape)

        posture_type = _classify_posture(np.asarray(keypoints_17x3, dtype=np.float64), body_final)
        aspect_ratio = float(body_final[3] / max(body_final[2], 1e-6))

        return AnatomicalBoundsReport(
            body_bbox=body_final,
            head_bbox=head_final,
            confidence=round(confidence, 4),
            posture_type=posture_type,
            tight_aspect_ratio=round(aspect_ratio, 4),
        )


def _joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    v1 = a - b
    v2 = c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cos_theta = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _classify_posture(kpts: np.ndarray, body_box: Tuple[float, float, float, float]) -> str:
    def conf(idx: int) -> float:
        return float(kpts[idx, 2])

    def visible(idx: int) -> bool:
        return conf(idx) >= MIN_KP_CONF

    core_visible = sum(
        visible(i) for i in (KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP)
    )
    if core_visible < 2:
        return "PARTIAL_VISIBILITY"

    torso_angle_deg = 0.0
    sh_pts = [kpts[i, :2] for i in (KP_L_SHOULDER, KP_R_SHOULDER) if visible(i)]
    hip_pts = [kpts[i, :2] for i in (KP_L_HIP, KP_R_HIP) if visible(i)]
    if sh_pts and hip_pts:
        shoulder_mid = np.mean(sh_pts, axis=0)
        hip_mid = np.mean(hip_pts, axis=0)
        dx, dy = shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1]
        torso_angle_deg = float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))

    knee_angles = []
    for hip_i, knee_i, ankle_i in ((KP_L_HIP, KP_L_KNEE, KP_L_ANKLE), (KP_R_HIP, KP_R_KNEE, KP_R_ANKLE)):
        if visible(hip_i) and visible(knee_i) and visible(ankle_i):
            ang = _joint_angle_deg(kpts[hip_i, :2], kpts[knee_i, :2], kpts[ankle_i, :2])
            if ang is not None:
                knee_angles.append(ang)

    hip_angles = []
    for sh_i, hip_i, knee_i in ((KP_L_SHOULDER, KP_L_HIP, KP_L_KNEE), (KP_R_SHOULDER, KP_R_HIP, KP_R_KNEE)):
        if visible(sh_i) and visible(hip_i) and visible(knee_i):
            ang = _joint_angle_deg(kpts[sh_i, :2], kpts[hip_i, :2], kpts[knee_i, :2])
            if ang is not None:
                hip_angles.append(ang)

    avg_knee_angle = float(np.mean(knee_angles)) if knee_angles else None
    avg_hip_angle = float(np.mean(hip_angles)) if hip_angles else None

    arms_extended = False
    if sh_pts:
        reference_scale = float(np.linalg.norm(sh_pts[0] - sh_pts[-1])) if len(sh_pts) == 2 else None
        if reference_scale is None or reference_scale < 4.0:
            _, _, body_w, _ = body_box
            reference_scale = body_w * 0.55
        for sh_i, wr_i in ((KP_L_SHOULDER, KP_L_WRIST), (KP_R_SHOULDER, KP_R_WRIST)):
            if visible(sh_i) and visible(wr_i):
                horiz_dist = abs(float(kpts[wr_i, 0] - kpts[sh_i, 0]))
                if horiz_dist >= reference_scale * _ARM_EXTEND_RATIO:
                    arms_extended = True
                    break

    if avg_knee_angle is not None and avg_hip_angle is not None:
        if avg_knee_angle < _KNEE_BENT_CROUCH_DEG and avg_hip_angle < _HIP_BENT_CROUCH_DEG:
            return "CROUCHING"
        if (_KNEE_BENT_SITTING_DEG[0] <= avg_knee_angle <= _KNEE_BENT_SITTING_DEG[1]
                and avg_hip_angle < 150.0):
            return "SITTING"

    if torso_angle_deg >= _TORSO_LEAN_DEG:
        return "LEANING"

    if avg_knee_angle is not None and avg_knee_angle < _KNEE_STRAIGHT_DEG and avg_hip_angle is not None and avg_hip_angle < 150.0:
        return "SITTING"

    if arms_extended:
        return "STANDING_ARMS_EXTENDED"

    if avg_knee_angle is None and avg_hip_angle is None:
        return "PARTIAL_VISIBILITY"

    return "STANDING"


_DEFAULT_DUAL_ZONE_FITTER = MikoVoDualZoneFitter()


def fit_anatomical_person_bounds(
    keypoints_17x3: np.ndarray,
    raw_bbox: Tuple[float, float, float, float],
    frame: Optional[np.ndarray] = None,
    track_id: Optional[Any] = None,
    frame_shape: Optional[Tuple[int, ...]] = None,
    fitter: MikoVoDualZoneFitter = _DEFAULT_DUAL_ZONE_FITTER,
) -> AnatomicalBoundsReport:
    return fitter.fit(
        keypoints_17x3=keypoints_17x3,
        raw_bbox=raw_bbox,
        frame=frame,
        track_id=track_id,
        frame_shape=frame_shape,
    )


# =============================================================================
# 12. SUBSYSTEM 12: TACTICAL BODY-CAM MESH & DYNAMIC WIREFRAME HUD
# =============================================================================

BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]
Vertices = List[Tuple[int, int]]

TACTICAL_CYAN: Color = (255, 215, 0)
SECURITY_EMERALD: Color = (88, 209, 48)
THREAT_AMBER: Color = (10, 159, 255)

# COCO limb connectivity.
_BONES = (
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
_RADIUS_RATIOS = np.array(
    [0.15, 0.12, 0.15, 0.12, 0.22, 0.16, 0.22, 0.16],
    dtype=np.float32,
)


@dataclass(frozen=True)
class HullResult:
    """Ordered source-frame perimeter plus uncalibrated geometry telemetry."""
    vertices: np.ndarray
    confidence: float
    posture: str
    height_px: float
    fallback: bool


def _validate_frame(frame: np.ndarray) -> None:
    """Require writable uint8 BGR input without silently copying it."""
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or min(frame.shape[:2]) < 2
    ):
        raise ValueError("frame must be uint8 BGR with shape (H, W, 3).")
    if not frame.flags.writeable:
        raise ValueError("frame must be writable.")


def _prepare(
    keypoints: np.ndarray,
    bbox: BBox,
    shape: Tuple[int, ...],
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sanitize missing landmarks and intersect the candidate box with frame."""
    kpts = np.asarray(keypoints, dtype=np.float32)
    if kpts.shape != (17, 3):
        raise ValueError("keypoints must have shape (17, 3).")

    box = np.asarray(bbox, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("bbox must contain four finite values.")
    if np.any(box[2:] <= 0):
        raise ValueError("bbox width and height must be positive.")

    height, width = shape[:2]
    lo = np.maximum(box[:2], 0.0)
    hi = np.minimum(box[:2] + box[2:], (width, height))
    if np.any(hi <= lo):
        raise ValueError("bbox does not intersect the frame.")

    clean = np.where(np.isfinite(kpts), kpts, 0.0).copy()
    clean[:, 2] = np.clip(clean[:, 2], 0.0, 1.0)

    valid = (
        np.isfinite(kpts).all(axis=1)
        & (clean[:, 2] >= threshold)
        & (clean[:, 0] >= 0)
        & (clean[:, 0] < width)
        & (clean[:, 1] >= 0)
        & (clean[:, 1] < height)
    )

    tolerance = 0.30 * box[2:]
    valid &= (
        (clean[:, :2] >= box[:2] - tolerance)
        & (clean[:, :2] <= box[:2] + box[2:] + tolerance)
    ).all(axis=1)

    return clean, valid, np.concatenate((lo, hi - lo))


def _resample_closed(contour: np.ndarray, count: int) -> np.ndarray:
    """Uniform arc-length resampling with deterministic winding/start vertex."""
    points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    keep = lengths > 1e-6
    points = points[keep]
    lengths = lengths[keep]

    if len(points) < 3:
        raise ValueError("Contour has fewer than three distinct vertices.")

    signed_area = np.sum(
        points[:, 0] * np.roll(points[:, 1], -1)
        - points[:, 1] * np.roll(points[:, 0], -1)
    )
    if signed_area < 0:
        points = points[::-1].copy()

    points = np.roll(points, -int(np.argmin(points[:, 1])), axis=0)
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    samples = np.arange(count) * cumulative[-1] / count
    indices = np.searchsorted(cumulative, samples, side="right") - 1
    indices = np.clip(indices, 0, len(points) - 1)
    fraction = (samples - cumulative[indices]) / np.maximum(
        lengths[indices], 1e-6
    )
    return (
        points[indices]
        + fraction[:, None] * (following[indices] - points[indices])
    ).astype(np.float32)


class MikoVoAnatomicalContourHull:
    def __init__(
        self,
        vertex_count: int = 20,
        mask_size: int = 160,
        min_confidence: float = 0.30,
    ) -> None:
        if not 12 <= vertex_count <= 24:
            raise ValueError("vertex_count must be between 12 and 24.")
        if not 64 <= mask_size <= 512:
            raise ValueError("mask_size must be between 64 and 512.")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1].")
        self.vertex_count = vertex_count
        self.mask_size = mask_size
        self.min_confidence = min_confidence

    @staticmethod
    def _posture(kpts: np.ndarray, valid: np.ndarray) -> str:
        if not valid[[5, 6, 11, 12]].all():
            return "PARTIAL"

        shoulder = kpts[[5, 6], :2].mean(axis=0)
        hip = kpts[[11, 12], :2].mean(axis=0)
        torso = hip - shoulder
        if abs(float(torso[0])) > 0.55 * max(abs(float(torso[1])), 1.0):
            return "LEANING"

        if valid[[13, 14, 15, 16]].all():
            knees = kpts[[13, 14], :2]
            thighs = kpts[[11, 12], :2] - knees
            calves = kpts[[15, 16], :2] - knees
            denominator = (
                np.linalg.norm(thighs, axis=1)
                * np.linalg.norm(calves, axis=1)
            )
            if np.all(denominator > 1.0):
                cosine = np.sum(thighs * calves, axis=1) / denominator
                angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
                if float(np.mean(angles)) < 125.0:
                    return "BENT_LEGS"

        return "UPRIGHT"

    def compute(
        self,
        keypoints: np.ndarray,
        bbox: BBox,
        frame_shape: Tuple[int, ...],
    ) -> HullResult:
        kpts, valid, box = _prepare(
            keypoints, bbox, frame_shape, self.min_confidence
        )
        x, y, width, height = box

        def fallback() -> HullResult:
            rectangle = np.array(
                [[x, y], [x + width, y],
                 [x + width, y + height], [x, y + height]],
                dtype=np.float32,
            )
            return HullResult(
                _resample_closed(rectangle, self.vertex_count),
                0.0, "UNKNOWN", float(height), True,
            )

        if int(valid[5:].sum()) < 4:
            return fallback()

        scales: List[float] = []
        for left, right, multiplier in ((5, 6, 1.0), (11, 12, 1.25)):
            if valid[left] and valid[right]:
                distance = float(np.linalg.norm(
                    kpts[left, :2] - kpts[right, :2]
                ))
                if distance > 2.0:
                    scales.append(distance * multiplier)

        if valid[[5, 6, 11, 12]].all():
            torso_length = float(np.linalg.norm(
                kpts[[5, 6], :2].mean(axis=0)
                - kpts[[11, 12], :2].mean(axis=0)
            ))
            if torso_length > 2.0:
                scales.append(0.65 * torso_length)

        scale = float(np.median(scales)) if scales else 0.42 * width
        scale = float(np.clip(scale, 3.0, max(width, height) * 0.65))

        observed = kpts[valid, :2]
        low = observed.min(axis=0) - scale * 0.95
        high = observed.max(axis=0) + scale * 0.95
        low = np.maximum(low, 0.0)
        high = np.minimum(high, frame_shape[1::-1])
        extent = np.maximum(high - low, 1.0)
        raster_scale = (self.mask_size - 3.0) / float(extent.max())
        dimensions = np.maximum(
            4, np.ceil(extent * raster_scale).astype(np.int32) + 3
        )
        mask = np.zeros((int(dimensions[1]), int(dimensions[0])), np.uint8)
        mapped = np.rint((kpts[:, :2] - low) * raster_scale + 1).astype(
            np.int32
        )

        def point(index: int) -> Tuple[int, int]:
            return int(mapped[index, 0]), int(mapped[index, 1])

        torso_indices = np.array([5, 6, 12, 11])
        torso_visible = torso_indices[valid[torso_indices]]
        if len(torso_visible) >= 3:
            torso_hull = cv2.convexHull(mapped[torso_visible])
            cv2.fillConvexPoly(mask, torso_hull, 255)
            radius = max(1, int(round(scale * 0.08 * raster_scale)))
            cv2.polylines(
                mask, [torso_hull], True, 255, 2 * radius + 1
            )
        elif len(torso_visible) == 2:
            cv2.line(
                mask, point(int(torso_visible[0])),
                point(int(torso_visible[1])), 255,
                max(2, int(scale * 0.30 * raster_scale)),
            )

        for (start, end), ratio in zip(_BONES, _RADIUS_RATIOS):
            if valid[start] and valid[end]:
                radius = max(1, int(round(
                    scale * float(ratio) * raster_scale
                )))
                cv2.line(mask, point(start), point(end), 255, 2 * radius)
                cv2.circle(mask, point(start), radius, 255, -1)
                cv2.circle(mask, point(end), radius, 255, -1)

        for index in np.flatnonzero(valid[5:]) + 5:
            radius = max(1, int(scale * 0.10 * raster_scale))
            cv2.circle(mask, point(int(index)), radius, 255, -1)

        head_visible = np.flatnonzero(valid[:5])
        if len(head_visible):
            center = kpts[head_visible, :2].mean(axis=0)
            head_width = scale * 0.54
            head_height = scale * 0.70
            angle = 0.0

            if valid[1] and valid[2]:
                eye_vector = kpts[2, :2] - kpts[1, :2]
                iod = float(np.linalg.norm(eye_vector))
                if iod > 1.0:
                    head_width = float(np.clip(
                        iod * 2.6, scale * 0.35, scale * 0.90
                    ))
                    head_height = head_width * 1.30
                    center = kpts[[1, 2], :2].mean(axis=0)
                    angle = math.degrees(math.atan2(
                        float(eye_vector[1]), float(eye_vector[0])
                    ))
                    down = np.array(
                        [-eye_vector[1], eye_vector[0]], dtype=np.float32
                    ) / iod
                    if down[1] < 0:
                        down = -down
                    center = center + down * head_height * 0.08

            mapped_center = np.rint(
                (center - low) * raster_scale + 1
            ).astype(np.int32)
            axes = (
                max(1, int(head_width * raster_scale * 0.5)),
                max(1, int(head_height * raster_scale * 0.5)),
            )
            cv2.ellipse(
                mask, tuple(map(int, mapped_center)), axes,
                angle, 0, 360, 255, -1,
            )

            shoulders = np.array([5, 6])[valid[[5, 6]]]
            if len(shoulders):
                neck = mapped[shoulders].mean(axis=0).astype(np.int32)
                cv2.line(
                    mask, tuple(map(int, mapped_center)),
                    tuple(map(int, neck)), 255,
                    max(2, int(scale * 0.23 * raster_scale)),
                )

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [c for c in contours if cv2.contourArea(c) > 1.0]
        if not contours:
            return fallback()

        disconnected = len(contours) > 1
        if disconnected:
            contour = cv2.convexHull(np.concatenate(contours, axis=0))
        else:
            contour = contours[0]

        if len(contour) < 3:
            return fallback()

        vertices = _resample_closed(contour, self.vertex_count)
        vertices = (vertices - 1.0) / raster_scale + low
        coverage = float(valid.sum()) / 17.0
        confidence = float(kpts[valid, 2].mean()) * math.sqrt(coverage)
        if disconnected:
            confidence *= 0.75

        return HullResult(
            vertices.astype(np.float32),
            float(np.clip(confidence, 0.0, 1.0)),
            self._posture(kpts, valid),
            float(np.ptp(vertices[:, 1])),
            disconnected,
        )


@dataclass
class _MeshState:
    vertices: np.ndarray
    timestamp: float


class MikoVoTemporalMeshStabilizer:
    def __init__(
        self,
        min_alpha: float = 0.25,
        deadband_px: float = 0.35,
        ttl_seconds: float = 2.0,
        max_tracks: int = 256,
    ) -> None:
        self.min_alpha = min_alpha
        self.deadband_px = deadband_px
        self.ttl_seconds = ttl_seconds
        self.max_tracks = max_tracks
        self._states: dict[Hashable, _MeshState] = {}
        self._calls = 0

    def reset_track(self, track_id: Hashable) -> None:
        self._states.pop(track_id, None)

    def clear(self) -> None:
        self._states.clear()

    def update(
        self,
        track_id: Hashable,
        vertices: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> np.ndarray:
        now = time.monotonic() if timestamp is None else float(timestamp)
        current = np.asarray(vertices, dtype=np.float32)
        if (
            current.ndim != 2
            or current.shape[1] != 2
            or len(current) < 3
            or not np.isfinite(current).all()
            or not math.isfinite(now)
        ):
            raise ValueError("Invalid vertices or timestamp.")

        self._calls += 1
        if self._calls % 128 == 0:
            stale = [
                key for key, state in self._states.items()
                if now - state.timestamp > self.ttl_seconds
            ]
            for key in stale:
                del self._states[key]

        state = self._states.get(track_id)
        if (
            state is None
            or state.vertices.shape != current.shape
            or not 0 < now - state.timestamp <= self.ttl_seconds
        ):
            result = current.copy()
        else:
            previous = state.vertices
            count = len(current)

            indices = (
                np.arange(count)[None, :] + np.arange(count)[:, None]
            ) % count
            candidates = current[indices]
            centered = candidates - candidates.mean(axis=1, keepdims=True)
            old_centered = previous - previous.mean(axis=0)
            costs = np.mean((centered - old_centered) ** 2, axis=(1, 2))
            aligned = candidates[int(np.argmin(costs))]

            residual = aligned - previous
            distance = np.linalg.norm(residual, axis=1)
            scale = max(float(np.ptp(aligned, axis=0).max()), 16.0)
            dt = min(max(now - state.timestamp, 1e-3), 0.2)

            fast = distance > max(4.0, scale * 0.045)
            alpha = self.min_alpha + (1.0 - self.min_alpha) * np.clip(
                distance / max(scale * 0.045, 4.0), 0.0, 1.0
            )
            step = residual * alpha[:, None]
            step_length = np.linalg.norm(step, axis=1)
            max_step = max(0.75, scale * 2.0 * dt)
            step *= np.minimum(
                1.0, max_step / np.maximum(step_length, 1e-6)
            )[:, None]
            step[fast] = residual[fast]
            step[distance <= self.deadband_px] = 0.0
            result = previous + step

        if track_id not in self._states and len(self._states) >= self.max_tracks:
            oldest = min(
                self._states, key=lambda key: self._states[key].timestamp
            )
            del self._states[oldest]

        self._states[track_id] = _MeshState(result.copy(), now)
        return result


class MikoVoTacticalHUDPainter:
    def __init__(self, node_radius: int = 3, crosshairs: bool = False) -> None:
        self.node_radius = node_radius
        self.crosshairs = crosshairs

    def paint(
        self,
        frame: np.ndarray,
        vertices: np.ndarray,
        track_id: Hashable,
        confidence: float,
        posture: str,
        height_px: float,
        color: Color,
        show_nodes: bool,
        show_mesh: bool,
        show_telemetry: bool,
        alpha: float,
    ) -> np.ndarray:
        if alpha <= 0 or not (show_nodes or show_mesh or show_telemetry):
            return frame

        points = np.rint(vertices).astype(np.int32)
        frame_h, frame_w = frame.shape[:2]
        points[:, 0] = np.clip(points[:, 0], 0, frame_w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, frame_h - 1)

        label_id = str(track_id).encode("ascii", "replace").decode("ascii")[:20]
        label = (
            f"ID {label_id} | {posture} | "
            f"H {height_px:.0f}px | Q {confidence:.2f}"
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, 1
        )
        text_x = int(np.clip(
            points[:, 0].min(), 0, max(frame_w - text_w - 8, 0)
        ))
        apex = int(points[:, 1].min())
        text_y = apex - 12
        if text_y < text_h + 6:
            text_y = min(frame_h - baseline - 4, apex + text_h + 12)
        text_y = max(text_h + 2, text_y)

        padding = self.node_radius + 7
        lo = points.min(axis=0) - padding
        hi = points.max(axis=0) + padding + 1
        if show_telemetry:
            lo = np.minimum(lo, (text_x, text_y - text_h - 5))
            hi = np.maximum(hi, (
                text_x + text_w + 8, text_y + baseline + 5
            ))
        x0, y0 = np.maximum(lo, 0).astype(int)
        x1, y1 = np.minimum(hi, (frame_w, frame_h)).astype(int)
        if x1 <= x0 or y1 <= y0:
            return frame

        roi = frame[y0:y1, x0:x1]
        overlay = roi.copy()
        local = points - np.array([x0, y0], dtype=np.int32)
        line = local.reshape(-1, 1, 2)
        dim = tuple(int(channel * 0.28) for channel in color)
        mid = tuple(int(channel * 0.58) for channel in color)
        core = tuple(min(255, int(channel * 0.35 + 165)) for channel in color)

        if show_mesh:
            cv2.polylines(overlay, [line], True, dim, 4, cv2.LINE_AA)
            cv2.polylines(overlay, [line], True, color, 1, cv2.LINE_AA)

        if show_nodes:
            for px, py in local:
                center = (int(px), int(py))
                cv2.circle(
                    overlay, center, self.node_radius + 3,
                    dim, -1, cv2.LINE_AA,
                )
                cv2.circle(
                    overlay, center, self.node_radius + 1,
                    mid, -1, cv2.LINE_AA,
                )
                cv2.circle(
                    overlay, center, self.node_radius,
                    color, -1, cv2.LINE_AA,
                )
                cv2.circle(overlay, center, 1, core, -1, cv2.LINE_AA)
                if self.crosshairs:
                    radius = self.node_radius + 5
                    cv2.line(
                        overlay, (int(px - radius), int(py)),
                        (int(px + radius), int(py)), color, 1, cv2.LINE_AA,
                    )
                    cv2.line(
                        overlay, (int(px), int(py - radius)),
                        (int(px), int(py + radius)), color, 1, cv2.LINE_AA,
                    )

        if show_telemetry:
            left, top = text_x - x0, text_y - y0
            cv2.rectangle(
                overlay, (left, top - text_h - 4),
                (left + text_w + 7, top + baseline + 3),
                (14, 19, 22), -1,
            )
            cv2.line(
                overlay, (left, top - text_h - 4),
                (left + text_w + 7, top - text_h - 4),
                color, 1, cv2.LINE_AA,
            )
            cv2.putText(
                overlay, label, (left + 3, top),
                font, font_scale, color, 1, cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)
        return frame


class TacticalBodycamMesh:
    def __init__(self, vertex_count: int = 20) -> None:
        self.hull = MikoVoAnatomicalContourHull(vertex_count=vertex_count)
        self.stabilizer = MikoVoTemporalMeshStabilizer()
        self.painter = MikoVoTacticalHUDPainter()
        self._lock = threading.RLock()

    def reset_track(self, track_id: Hashable) -> None:
        with self._lock:
            self.stabilizer.reset_track(track_id)

    def render(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        bbox: BBox,
        track_id: Hashable,
        color: Color = (0, 215, 255),
        show_nodes: bool = True,
        show_mesh: bool = True,
        show_telemetry: bool = True,
        alpha: float = 0.75,
        timestamp: Optional[float] = None,
    ) -> Tuple[np.ndarray, Vertices]:
        _validate_frame(frame)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        channels = np.asarray(color, dtype=np.float64)
        bgr = tuple(int(np.clip(value, 0, 255)) for value in channels)

        with self._lock:
            try:
                geometry = self.hull.compute(keypoints, bbox, frame.shape)
            except Exception:
                x, y, width, height = bbox
                rect = np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float32)
                geometry = HullResult(_resample_closed(rect, self.hull.vertex_count), 0.0, "UNKNOWN", float(height), True)

            vertices = self.stabilizer.update(
                track_id, geometry.vertices, timestamp
            )
            vertices[:, 0] = np.clip(
                vertices[:, 0], 0, frame.shape[1] - 1
            )
            vertices[:, 1] = np.clip(
                vertices[:, 1], 0, frame.shape[0] - 1
            )
            self.painter.paint(
                frame, vertices, track_id, geometry.confidence,
                geometry.posture, geometry.height_px, bgr,
                show_nodes, show_mesh, show_telemetry, alpha,
            )

        rounded = np.rint(vertices).astype(np.int32)
        result = [(int(x), int(y)) for x, y in rounded]
        return frame, result


_DEFAULT_TACTICAL_MESH = TacticalBodycamMesh()


def render_tactical_bodycam_mesh(
    frame: np.ndarray,
    keypoints: np.ndarray,
    bbox: Tuple[int, int, int, int],
    track_id: int,
    color: Color = (0, 215, 255),
    show_nodes: bool = True,
    show_mesh: bool = True,
    show_telemetry: bool = True,
    alpha: float = 0.75,
) -> Tuple[np.ndarray, Vertices]:
    return _DEFAULT_TACTICAL_MESH.render(
        frame, keypoints, bbox, track_id, color,
        show_nodes, show_mesh, show_telemetry, alpha,
    )


# =============================================================================
# 9. MIKOVO RARA 11 VISION & TRACKING ENGINE
# =============================================================================


class MikoVoRaRa11VisionEngine:
    """
    MikoVo RaRa 11 Autonomous AI Vision Core:
      - YOLOv8 Neural Backbone + SVD Weighted Umeyama Alignment
      - 512-Dimensional Deep Metric Embeddings (< 3.5ms on CPU)
      - 8-State Kalman Filter Temporal Tracking with Mahalanobis Gating
      - Virtual Tripwire & ROI Spatial Filtering
      - Circular Frame Buffer for Instant Video Alert Dispatch
      - Thread-Safe RLock Frame Publishing & Camera Fallback Test Pattern Stream
    """
    def __init__(self, db: FaceDatabase):
        self.db = db
        self.tracker = MikoVoRaRa11MultiKalmanTracker()
        self._lock = threading.RLock()
        self.running = False
        self.cap = None
        self.is_simulated = False
        self._probe_thread = None
        self.feature_extractor = _DEFAULT_EXTRACTOR
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
        self.aggression_strike_analyzer = MikoVoStrikeVelocityAnalyzer()
        self.aggression_interlock_analyzer = MikoVoPairwiseInterlockAnalyzer()
        self.aggression_entropy_analyzer = MikoVoChaoticStruggleEntropy()
        self.latest_aggression_report = None

        # MikoVo RaRa 11 Anatomical Body & Head Shrink-Wrap Fitter
        self.anatomical_fitter = MikoVoDualZoneFitter()

        # MikoVo RaRa 11 Tactical Body-Cam Mesh & Connected Nodes Core
        self.tactical_mesh = TacticalBodycamMesh(vertex_count=20)

        # Circular buffer for pre-incident video clip (~4 sec @ 15 FPS)
        self.frame_buffer = deque(maxlen=60)
        self.latest_annotated_mat = None

        self.model = None
        if HAS_YOLO:
            weights = (
                MIKOVO_WEIGHTS
                if MIKOVO_WEIGHTS.exists()
                else (POSE_WEIGHTS if POSE_WEIGHTS.exists() else BASE_WEIGHTS)
            )
            if not weights.exists():
                try:
                    tmp_m = YOLO("yolov8n-pose.pt")
                    if POSE_WEIGHTS.exists():
                        import shutil
                        shutil.move(str(POSE_WEIGHTS), str(MIKOVO_WEIGHTS))
                    weights = MIKOVO_WEIGHTS
                except Exception:
                    pass
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
            cap = None
            # Bounded retry: up to 3 attempts with 150ms delay
            for attempt in range(3):
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                if cap.isOpened():
                    break
                cap.release()
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    break
                cap.release()
                time.sleep(0.15)

            if cap and cap.isOpened():
                self.cap = cap
                self.is_simulated = False
            else:
                self.cap = None
                self.is_simulated = True
                print(f"[MikoVo RaRa 11] Camera #{index} unavailable. Initializing Animated Tactical Test-Pattern Stream.")
                if not self._probe_thread or not self._probe_thread.is_alive():
                    self._probe_thread = threading.Thread(target=self._camera_probe_loop, args=(index,), daemon=True)
                    self._probe_thread.start()

            self.running = True

        threading.Thread(target=self._loop, daemon=True).start()
        return True

    def _camera_probe_loop(self, index):
        """Background thread auto-probing for hardware camera re-connection."""
        while self.running and self.is_simulated:
            time.sleep(3.0)
            if not self.running or not self.is_simulated:
                break
            try:
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    ret, test_f = cap.read()
                    if ret and test_f is not None:
                        with self._lock:
                            self.cap = cap
                            self.is_simulated = False
                            print(f"[MikoVo RaRa 11] Hardware Camera #{index} Auto-Recovered Successfully.")
                            break
                    cap.release()
            except Exception:
                pass

    def _generate_test_pattern_frame(self, w=1280, h=720):
        """Generates dynamic tactical test pattern when hardware sensor is offline."""
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Background dark tactical grid
        grid_size = 40
        frame[::grid_size, :, :] = (25, 28, 34)
        frame[:, ::grid_size, :] = (25, 28, 34)

        # Dynamic radar sweep line
        now_t = time.time()
        sweep_x = int((now_t * 220) % w)
        cv2.line(frame, (sweep_x, 0), (sweep_x, h), (0, 180, 255), 1, cv2.LINE_AA)

        # Central HUD crosshairs & concentric target circles
        cx, cy = w // 2, h // 2
        cv2.drawMarker(frame, (cx, cy), (70, 75, 110), cv2.MARKER_CROSS, 40, 1)
        cv2.circle(frame, (cx, cy), 130, (45, 50, 75), 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 260, (30, 35, 55), 1, cv2.LINE_AA)

        # Status badge
        cv2.rectangle(frame, (cx - 330, cy - 65), (cx + 330, cy + 65), (14, 16, 24), -1)
        cv2.rectangle(frame, (cx - 330, cy - 65), (cx + 330, cy + 65), (0, 165, 255), 2)
        cv2.putText(frame, "CAM_OFFLINE: SIMULATED SENSOR PATTERN", (cx - 305, cy - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "MikoVo RaRa 11 AI Neural Vision Core [STANDBY / PROBING]", (cx - 280, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 180, 205), 1, cv2.LINE_AA)
        cv2.putText(frame, f"AUTO-PROBING HARDWARE SENSOR...  {datetime.now().strftime('%H:%M:%S.%f')[:-4]}", (cx - 270, cy + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (88, 209, 48), 1, cv2.LINE_AA)

        # Bottom telemetry
        cv2.putText(frame, f"FPS: {self.stats.get('fps', 30.0):.1f} | SENSOR: VIRTUAL_SYNTH | RES: {w}x{h}", (20, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 150, 175), 1, cv2.LINE_AA)
        return frame

    def get_snapshot(self):
        """Thread-safe snapshot getter for frame publishing."""
        with self._lock:
            raw_f = self.latest_frame
            ann_f = self.latest_annotated
            dets = list(self.last_detections) if self.last_detections else []
            st = dict(self.stats)
            sim = self.is_simulated
            return raw_f, ann_f, dets, st, sim

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
            frame = None
            with self._lock:
                cap = self.cap
                sim = self.is_simulated

            if sim or cap is None:
                time.sleep(0.04)
                frame = self._generate_test_pattern_frame()
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    with self._lock:
                        self.is_simulated = True
                        if self.cap:
                            try: self.cap.release()
                            except Exception: pass
                            self.cap = None
                        if not self._probe_thread or not self._probe_thread.is_alive():
                            self._probe_thread = threading.Thread(
                                target=self._camera_probe_loop,
                                args=(int(config.get("camera_index", 0)),),
                                daemon=True
                            )
                            self._probe_thread.start()
                    frame = self._generate_test_pattern_frame()

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

            # Evaluate MikoVo RaRa 11 Behavioral & Kinetic Threats + Anatomical Fitter
            for d in tracked:
                kalman_state = d.get("kalman_state")
                kpts = d.get("kpts")
                confs = d.get("kp_conf")
                full_kpts = d.get("full_kpts")
                face_crop = d.get("face_crop")
                tid = d.get("track_id", 1)
                bbox = d.get("smooth_bbox", d["bbox"])

                # MikoVo RaRa 11 Shrink-Wrapped Anatomical Bounds (Body & Head Envelope)
                raw_box_tuple = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                if full_kpts is not None and len(full_kpts) >= 5:
                    k17 = full_kpts if len(full_kpts) == 17 else _standardize_keypoints_array(full_kpts, np.array([raw_box_tuple]))[0]
                elif kpts is not None and len(kpts) >= 5:
                    k5 = np.column_stack([kpts[:5], confs[:5] if confs is not None else np.ones(5)])
                    k17 = _standardize_keypoints_array(k5, np.array([raw_box_tuple]))[0]
                else:
                    k17 = np.zeros((17, 3), dtype=np.float64)

                d["k17"] = k17

                anat_rep = self.anatomical_fitter.fit(
                    keypoints_17x3=k17,
                    raw_bbox=raw_box_tuple,
                    frame=frame,
                    track_id=tid,
                    frame_shape=frame.shape
                )
                d["anatomical"] = anat_rep
                d["body_bbox"] = [int(v) for v in anat_rep.body_bbox]
                d["head_bbox"] = [int(v) for v in anat_rep.head_bbox]
                d["posture"] = anat_rep.posture_type

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

            # Evaluate MikoVo RaRa 11 Multi-Person Physical Aggression & Struggle Sentinel
            tracks_hist = {}
            for trk_id, trk in self.tracker.trackers.items():
                if len(trk.bbox_history) >= 2:
                    tracks_hist[trk_id] = {
                        "bboxes": np.array(trk.bbox_history, dtype=np.float64),
                        "keypoints": np.array(trk.kpts_history, dtype=np.float64)
                    }
            agg_report = evaluate_aggression_dynamics(
                tracks_hist,
                frame.shape,
                fps=float(self.stats.get("fps", 30.0)) or 30.0,
                strike_analyzer=self.aggression_strike_analyzer,
                interlock_analyzer=self.aggression_interlock_analyzer,
                entropy_analyzer=self.aggression_entropy_analyzer
            )
            self.latest_aggression_report = agg_report

            active_detections = []
            for d in tracked:
                in_roi = self._is_in_roi(d.get("smooth_bbox", d["bbox"]), frame.shape)
                d["in_roi"] = in_roi
                if in_roi:
                    active_detections.append(d)

            annotated = self._draw(frame.copy(), tracked)
            _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            _, raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            with self._lock:
                self.last_detections = tracked
                self.stats["person_count"] = len(active_detections)
                self.latest_annotated_mat = annotated
                self.frame_buffer.append(annotated.copy())
                self.latest_annotated = jpg.tobytes()
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
                        "full_kpts": kpts,
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

        if self.latest_aggression_report and self.latest_aggression_report.is_violent_threat:
            rep = self.latest_aggression_report
            cv2.rectangle(frame, (16, 38), (min(w - 16, 560), 72), (0, 0, 180), -1)
            cv2.putText(frame, f"! VIOLENCE ALERT: {rep.threat_type} ({rep.aggression_score:.2f}) !",
                        (24, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

        for det in detections:
            # Get bounding box
            x, y, bw, bh = det.get("smooth_bbox", det["bbox"])

            name = det.get("name")
            tid = det.get("track_id", 1)
            is_spoof = det.get("is_spoof", False)
            in_roi = det.get("in_roi", True)

            # Simple color logic
            if is_spoof:
                color = (0, 0, 255)       # Red
            elif name:
                color = (48, 209, 88)     # Green
            else:
                color = (0, 165, 255)     # Orange/Amber

            # Simple rectangle
            ix, iy = int(x), int(y)
            ix2, iy2 = int(x + bw), int(y + bh)
            cv2.rectangle(frame, (ix, iy), (ix2, iy2), color, 2, cv2.LINE_AA)

            # Name label
            if name:
                label = name
            else:
                label = "Unknown"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(th + 4, iy - 6)
            cv2.rectangle(frame, (ix, ly - th - 4), (ix + tw + 8, ly + 2), (0, 0, 0), -1)
            cv2.putText(frame, label, (ix + 4, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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

        if self.latest_aggression_report and self.latest_aggression_report.is_violent_threat:
            state["stats"]["threat_count"] = state["stats"].get("threat_count", 0) + 1
            agg = self.latest_aggression_report
            if agg.threat_type == "CHAOTIC_BRAWL":
                custom_cap = f"🚨 هشدار اضطراری MikoVo RaRa 11: نزاع و درگیری فیزیکی شدید در صحنه شناسایی شد! (امتیاز: {agg.aggression_score:.2f})"
            elif agg.threat_type == "PHYSICAL_GRAPPLE":
                custom_cap = f"🚨 هشدار امنیتی MikoVo RaRa 11: گلاویز شدن و درگیری نزدیک دو نفر ثبت شد! (امتیاز: {agg.aggression_score:.2f})"
            elif agg.threat_type == "MELEE_STRIKE":
                custom_cap = f"⚡ هشدار امنیتی MikoVo RaRa 11: پرتاب ضربه سریع فیزیکی یا مشت ثبت شد! (امتیاز: {agg.aggression_score:.2f})"

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

    def register_face(self, name: str, image_b64: Optional[str] = None):
        if not name or not name.strip():
            return False, "نام شخص الزامی است"
        name = name.strip()

        frame = None
        if image_b64:
            try:
                b64_clean = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
                img_data = base64.b64decode(b64_clean)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                return False, f"خطا در پردازش تصویر ارسالی: {e}"

        if frame is None:
            with self._lock:
                raw_bytes = self.latest_frame
            if not raw_bytes:
                return False, "هیچ فریمی از دوربین در دسترس نیست. لطفاً عکس آپلود کنید یا دوربین را روشن نمایید"
            nparr = np.frombuffer(raw_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return False, "فریم تصویر معتبر نیست"

        # Detect face & extract embedding
        detections = self._detect(frame)
        if not detections:
            return False, "شخصی یا چهره‌ای در تصویر تشخیص داده نشد"

        det = detections[0]
        if det.get("is_spoof"):
            return False, "خطای امنیتی: تصویر زنده نیست (تلاش برای ثبت عکس چاپی یا صفحه نمایش)!"

        crop = det.get("face_crop")
        if crop is None or crop.size == 0:
            bbox = det.get("bbox", [0, 0, frame.shape[1], frame.shape[0]])
            x, y, w, h = [int(v) for v in bbox]
            x, y = max(0, x), max(0, y)
            w = max(1, min(frame.shape[1] - x, w))
            h = max(1, min(frame.shape[0] - y, h))
            crop = frame[y:y+h, x:x+w]

        thumb = cv2.resize(crop, (64, 64))
        _, jpg = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()

        emb = det.get("embedding")
        if emb is None:
            _, emb, _ = mikovo_rara11_extract_face_descriptor(frame, (0, 0, frame.shape[1], frame.shape[0]))

        rec = self.db.add(name, emb, thumb_b64)
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

    def test(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        tok = (token or config.get("bot_token", "")).strip()
        cid = (chat_id or config.get("chat_id", "")).strip()
        if not tok or not cid:
            return False, "توکن یا شناسه چت تنظیم نشده است"
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{tok}/sendMessage",
                json={"chat_id": cid, "text": "✅ اتصال با سرور پایش MikoTim و موتور هوش مصنوعی MikoVo RaRa 11 با موفقیت برقرار شد."},
                timeout=8
            )
            res = r.json()
            if res.get("ok"):
                return True, "ارتباط با بات تلگرام با موفقیت تأیید شد."
            return False, res.get("description", "خطا در تلگرام")
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
            "Physical Strike Velocity & Arm Jerk Kinematic Analyzer (Punch/Melee Detection)",
            "Pairwise Proxemic Collapse & Clinch/Grapple Interlock Profiler",
            "Upper-Body Centroid Chaotic Struggle Entropy & Turbulence Detector",
        ]
    )

def mask_secret(secret: str) -> str:
    """Masks secret tokens (e.g. 1234••••abcd or ••••••••)."""
    if not secret:
        return ""
    if len(secret) <= 8:
        return "••••••••"
    return secret[:4] + "••••••••" + secret[-4:]

@app.route("/api/mode", methods=["GET", "POST", "OPTIONS"])
def api_mode():
    if request.method == "OPTIONS": return jsonify(ok=True)
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        mode = data.get("mode")
        if mode not in ("client_edge", "server_vision"):
            return jsonify(ok=False, error="Invalid mode. Must be 'client_edge' or 'server_vision'"), 400
        state["vision_mode"] = mode
        config["vision_mode"] = mode
        save_config(config)

        # Release server camera in Mode A to avoid competing with browser getUserMedia()
        if mode == "client_edge":
            detector.stop()
            state["is_streaming"] = False
        else:
            detector.start(index=int(config.get("camera_index", 0)))
            state["is_streaming"] = True
        return jsonify(ok=True, mode=state["vision_mode"])
    return jsonify(mode=state.get("vision_mode", "server_vision"))

@app.route("/api/detections", methods=["GET", "POST", "OPTIONS"])
def api_detections():
    if request.method == "OPTIONS": return jsonify(ok=True)
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        dets = data.get("detections", [])
        fps = float(data.get("fps", 0.0))
        count = int(data.get("count", len(dets)))
        state["client_detections"] = dets
        state["client_fps"] = fps
        if count > 0:
            state["stats"]["total_captures"] += 1
            current_hour = datetime.now().hour
            state["hourly_traffic"][current_hour] += 1
        return jsonify(ok=True)

    mode = state.get("vision_mode", "server_vision")
    if mode == "server_vision":
        dets = []
        for d in detector.last_detections:
            dets.append({
                "track_id": d.get("track_id"),
                "bbox": d.get("smooth_bbox", d.get("bbox")),
                "body_bbox": d.get("body_bbox"),
                "head_bbox": d.get("head_bbox"),
                "name": d.get("name"),
                "confidence": float(d.get("score", 0.8)),
                "posture": d.get("posture", "UPRIGHT_STANDING"),
                "in_roi": d.get("in_roi", False),
                "is_loitering": d.get("is_loitering", False),
            })
        return jsonify(
            mode="server_vision",
            count=detector.stats["person_count"],
            fps=detector.stats["fps"],
            is_simulated=detector.is_simulated,
            detections=dets,
            timestamp=time.time()
        )
    else:
        return jsonify(
            mode="client_edge",
            count=len(state.get("client_detections", [])),
            fps=state.get("client_fps", 0.0),
            detections=state.get("client_detections", []),
            timestamp=time.time()
        )

@app.route("/api/analytics", methods=["GET", "OPTIONS"])
def api_analytics():
    if request.method == "OPTIONS": return jsonify(ok=True)
    traffic = state["hourly_traffic"]
    max_idx = int(np.argmax(traffic)) if any(traffic) else datetime.now().hour
    max_val = int(traffic[max_idx]) if any(traffic) else 0
    return jsonify(
        hourly_traffic=traffic,
        total_captures=state["stats"]["total_captures"],
        in_count=state["stats"]["in_count"],
        out_count=state["stats"]["out_count"],
        loitering_count=state["stats"]["loitering_count"],
        threat_count=state["stats"].get("threat_count", 0),
        faces_count=len(face_db.records),
        peak_hour=max_idx,
        peak_count=max_val,
    )

@app.route("/api/telegram/send", methods=["POST", "OPTIONS"])
def api_telegram_send():
    if request.method == "OPTIONS": return jsonify(ok=True)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(ok=False, error="Invalid JSON payload"), 400

    photo_b64 = data.get("photo")
    if not photo_b64:
        return jsonify(ok=False, error="Field 'photo' (base64) is required"), 400

    caption = str(data.get("caption", ""))
    count = int(data.get("count", 1))
    names = data.get("names", [])
    if not isinstance(names, list):
        names = [str(names)]
    is_loitering = bool(data.get("is_loitering", False))

    try:
        b64_clean = photo_b64.split(",", 1)[1] if "," in photo_b64 else photo_b64
        jpeg_bytes = base64.b64decode(b64_clean)
    except Exception as e:
        return jsonify(ok=False, error=f"Invalid image base64 data: {e}"), 400

    ok = telegram.dispatch_photo(
        jpeg_bytes=jpeg_bytes,
        count=count,
        identified_names=names,
        caption=caption,
        is_loitering=is_loitering
    )
    if ok:
        state["stats"]["telegram_sent"] += 1
        return jsonify(ok=True, message="Alert successfully sent to Telegram")
    else:
        return jsonify(ok=False, error="Telegram dispatch failed. Verify bot token and chat ID."), 502

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
        vision_mode=state.get("vision_mode", "server_vision"),
        is_simulated=detector.is_simulated,
        fps=detector.stats["fps"],
        person_count=detector.stats["person_count"],
        total_captures=state["stats"]["total_captures"],
        telegram_sent=state["stats"]["telegram_sent"],
        in_count=state["stats"]["in_count"],
        out_count=state["stats"]["out_count"],
        loitering_count=state["stats"]["loitering_count"],
        threat_count=state["stats"].get("threat_count", 0),
        aggression={
            "is_violent_threat": detector.latest_aggression_report.is_violent_threat if detector.latest_aggression_report else False,
            "threat_level": detector.latest_aggression_report.threat_level if detector.latest_aggression_report else "CLEAR",
            "threat_type": detector.latest_aggression_report.threat_type if detector.latest_aggression_report else None,
            "aggression_score": detector.latest_aggression_report.aggression_score if detector.latest_aggression_report else 0.0,
            "summary": detector.latest_aggression_report.summary_text if detector.latest_aggression_report else ""
        },
        hourly_traffic=state["hourly_traffic"],
        alert_media=config.get("alert_media", "photo"),
        logs=state["logs"][:35],
    )

@app.route("/api/register", methods=["POST", "OPTIONS"])
def api_register():
    if request.method == "OPTIONS": return jsonify(ok=True)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(ok=False, error="Payload must be a JSON object"), 400

    name = str(data.get("name", "")).strip()
    if not name:
        return jsonify(ok=False, error="Name is required"), 400

    photo_b64 = data.get("photo") or data.get("image")
    ok, msg = detector.register_face(name, image_b64=photo_b64)
    if not ok:
        return jsonify(ok=False, message=msg), 400
    return jsonify(ok=True, message=msg)

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
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(ok=False, error="Payload must be a JSON object"), 400

        # Validate types and preserve masked secrets
        if "bot_token" in data:
            tok = str(data["bot_token"]).strip()
            if "••••" in tok or (not tok and config.get("bot_token")):
                data.pop("bot_token")  # Keep existing unmasked token

        if "cooldown_sec" in data:
            try:
                val = float(data["cooldown_sec"])
                if val < 1.0:
                    return jsonify(ok=False, error="cooldown_sec must be >= 1.0"), 400
                data["cooldown_sec"] = val
            except (ValueError, TypeError):
                return jsonify(ok=False, error="cooldown_sec must be a number"), 400

        if "loitering_sec" in data:
            try:
                val = float(data["loitering_sec"])
                if val < 1.0:
                    return jsonify(ok=False, error="loitering_sec must be >= 1.0"), 400
                data["loitering_sec"] = val
            except (ValueError, TypeError):
                return jsonify(ok=False, error="loitering_sec must be a number"), 400

        if "camera_index" in data:
            try:
                data["camera_index"] = int(data["camera_index"])
            except (ValueError, TypeError):
                return jsonify(ok=False, error="camera_index must be an integer"), 400

        config.update(data)
        save_config(config)
        return jsonify(ok=True)

    s = config.copy()
    s["has_token"] = bool(s.get("bot_token"))
    s["bot_token"] = mask_secret(s.get("bot_token", ""))
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
    if request.method == "OPTIONS": return jsonify(ok=True)
    data = request.get_json(silent=True) or {}
    custom_tok = data.get("token")
    custom_chat = data.get("chat_id")
    ok, msg = telegram.test(token=custom_tok, chat_id=custom_chat)
    if not ok:
        return jsonify(ok=False, message=msg), 400
    return jsonify(ok=True, message=msg)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
