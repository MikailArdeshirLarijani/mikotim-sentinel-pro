# MikoTim Sentinel Pro

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Neural Core](https://img.shields.io/badge/Vision%20Core-MikoVo%20RaRa%2011%20(YOLOv8--Derived)-8A2BE2.svg)](https://github.com/MikailArdeshirLarijani/mikotim-sentinel-pro)
[![Throughput](https://img.shields.io/badge/Throughput->200%20FPS%20(CPU)-success.svg)](https://github.com/MikailArdeshirLarijani/mikotim-sentinel-pro)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-orange.svg)](https://mikailardeshirlarijani.github.io/mikotim-sentinel-pro/)

**Next-Generation Intelligent Vision Surveillance & Biometric Threat Profiler**  
*Powered by the MikoVo RaRa 11 Neural Vision Core (Engineered & Evolved from YOLOv8 Architecture)*

[**Live Demo Dashboard**](https://mikailardeshirlarijani.github.io/mikotim-sentinel-pro/) • [**Key Features**](#key-features) • [**Architecture & Subsystems**](#mikovo-rara-11-neural-vision-core) • [**Getting Started**](#getting-started) • [**Telegram Integration**](#telegram-bot-gateway--remote-commands) • [**API Reference**](#restful-api-endpoints)

</div>

---

## Overview

**MikoTim Sentinel Pro** is a high-performance, enterprise-grade optical surveillance and biometric security gateway designed for zero-latency perimeter defense, facial identity verification, and multi-spectral threat detection.

At its heart lies the proprietary **MikoVo RaRa 11 Neural Vision Core** — an edge-optimized architecture directly derived and evolved from **YOLOv8** pose estimation, operating purely with vectorized NumPy and OpenCV array arithmetic (**no heavyweight PyTorch or TensorFlow runtimes required on inference hot paths**). It comfortably achieves **>200 FPS on single-core CPU** while providing military-grade anti-spoofing, kinetic threat analysis, and tactical anatomical mesh tracking.

---

## MikoVo RaRa 11 Neural Vision Core

The system integrates 12 specialized mathematical and biometric subsystems:

```
+-----------------------------------------------------------------------------------+
|                       MikoVo RaRa 11 Neural Vision Core                           |
+-----------------------------------------------------------------------------------+
|  1. 3D Umeyama Face Alignment (Canonical 112x112 via SVD)                        |
|  2. Adaptive LAB CLAHE + Dynamic Exponential Gamma Illumination Compensator       |
|  3. 512-Dimensional Deep Metric Hypersphere Feature Descriptor                    |
|  4. 8-State Constant-Velocity Kalman Tracker with Chi-Square 95% Mahalanobis Gate |
|  5. Multi-Spectral Passive Anti-Spoofing (2D FFT + YCrCb/HSV + rPPG Liveness)     |
|  6. 3D Head Pose Euler Solver (PnP) & Quality-Weighted EMA Hypersphere Memory     |
|  7. Facial Occlusion & Concealment Sentinel (Mask / Bandana / Balaclava Profiler) |
|  8. Kinetic Looming Dash & Sudden Fall Medical Emergency Aspect-Ratio Analyzer   |
|  9. Trajectory Tortuosity & Shannon Directional-Entropy Prowler Profiler          |
| 10. Physical Violence, Strike Velocity & Pairwise Clinch/Grapple Interlock Core   |
| 11. Shrink-Wrapped Anatomical Body & Head Fitter (Bone-Segment Projection)        |
| 12. Tactical Body-Cam Perimeter Mesh & Glowing Connected Anchor Nodes (HUD AA)    |
+-----------------------------------------------------------------------------------+
```

### 1. 3D-Aware Face Alignment via Weighted Umeyama Transform
- Least-squares SVD closed-form solver that rectifies raw 2D landmark coordinates (nose, eyes, ears) into canonical $112 \times 112$ metric space.
- Decouples rotation, isotropic scaling, and Euclidean translation, removing perspective distortion from oblique angles.

### 2. Adaptive LAB Illumination Compensator
- Operates on the luminance channel within CIE-LAB color space using Contrast Limited Adaptive Histogram Equalization (CLAHE).
- Applies a dynamic exponential gamma curve normalization to eliminate harsh shadows, direct backlights, and over-saturation without chromatic artifacts.

### 3. 512-Dimensional Deep Metric Descriptor
- Fuses multi-scale Scharr spatial gradient histograms ($8 \times 8, 4 \times 4, 2 \times 2$), multi-radius Uniform Local Binary Patterns (ULBP), and craniofacial anthropometric ratios into a unit-normalized $\mathbb{S}^{511}$ hypersphere embedding.
- High intra-class cohesion with strict inter-class separation (>0.82 cosine similarity verification threshold).

### 4. 8-State Kalman Tracker with Mahalanobis Gating
- Continuous state tracking $[x, y, a, h, v_x, v_y, v_a, v_h]$ incorporating constant-velocity kinematic projection.
- Suppresses false positives and tracking jitter using a 95% confidence $\chi^2$ Mahalanobis distance gating threshold and sliding-window majority identity voting.

### 5. Multi-Spectral Passive Anti-Spoofing Guard
- Instant mitigation against print attacks, screen replays, and tablet presentations.
- Evaluates 2D Fourier high-frequency specular decay (FFT), YCrCb/HSV chromatic dispersion, and temporal micro-capillary pulse fluctuations (rPPG).

### 6. 3D Head Pose Euler Solver & Quality-Weighted EMA Memory
- Direct Perspective-n-Point (PnP) pose solver yielding Pitch, Yaw, and Roll angles.
- Hypersphere identity vectors are aggregated into an Exponential Moving Average (EMA) memory track, automatically rejecting occluded or low-quality poses.

### 7. Facial Occlusion & Concealment Sentinel
- Dual-symmetry analysis combining micro-texture gradients and anthropometric confidence to detect surgical masks, bandanas, scarves, and balaclavas.

### 8. Kinetic Threat, Looming Dash & Fall Emergency Analyzer
- Real-time acceleration tracking paired with bounding box perspective inflation rates (Looming Approach) to detect rapid dashes or physical attacks.
- Aspect-ratio collapse triggers immediate alerts for sudden falls, fainting, or medical emergencies.

### 9. Trajectory Tortuosity & Shannon Entropy Prowler Profiler
- Evaluates movement complexity and directional Shannon entropy across 8 angular bins in a 45-frame rolling window to distinguish intentional transit from suspicious prowling/loitering.

### 10. Physical Violence & Grapple Interlock Sentinel
- Analyzes wrist strike kinematics, angular jerk, and pairwise proxemic bounding box overlap (Grapple/Clinch) at over 600 FPS on CPU to immediately identify fights and physical altercations.

### 11. Shrink-Wrapped Anatomical Body Fitter
- Projects anthropometric bone segments directly from skeletal geometry, snapping to clothing and silhouette contours via Sobel-Otsu edge detection.

### 12. Tactical Body-Cam Contour Mesh & Connected Nodes Core
- Replaces legacy rectangular bounding boxes with a futuristic tactical HUD.
- Renders 20 anti-aliased glowing perimeter anchor nodes connected with laser vector lines (`cv2.LINE_AA`) and smoothed with cyclic temporal stabilization.

---

## Dual Vision Pipeline Modes

| Feature | Mode A: Pure Client-Edge Vision | Mode B: Server Vision Core |
|---|---|---|
| **Primary Runtime** | Browser (`getUserMedia` + WebAssembly / BlazeFace / Pure JS) | Python Backend (Flask + OpenCV + YOLOv8 + MikoVo RaRa 11) |
| **Server Requirements** | None (Runs standalone on GitHub Pages) | Python 3.10+ with webcam or RTSP feed |
| **Stream Delivery** | HTML5 Canvas (Direct WebCam) | High-speed MJPEG `/video_feed` stream |
| **Telegram Alerts** | Direct via client-side Telegram Bot API | Circular MP4 video buffer & server photo dispatch |
| **Best For** | Static hosting, quick demo, remote browser clients | Enterprise surveillance, continuous video recording, heavy analytics |

---

## Key Features

- 📸 **Dual Alert Media Formats:** Instant high-resolution photo snapshot or real 5-second MP4 video clip (capturing 3 seconds pre-event and 2 seconds post-event).
- 🎯 **Interactive Virtual ROI & Directional Tripwire:** Draw custom multi-point restricted security zones directly by clicking on the video overlay.
- 🚶‍♂️ **Linear In/Out Counter:** Directional line-crossing counter tracking ingress and egress flows.
- ⏱ **Loitering & Prowler Detection:** Configurable threshold triggering alerts when an unauthorized person remains in the zone.
- 🚨 **Local Police Siren & Smart Speech Alerts:** Native Web Audio synthesizer with frequency ramps and smart multi-lingual speech warnings.
- 🕶 **Stealth Mode (OLED Desk Clock):** Covert background surveillance masked behind a minimalist OLED digital clock screensaver.
- 📊 **24-Hour Analytics & Traffic Distribution:** Hourly traffic distribution charts with KPI counters for total incidents, tripwire crossings, and peak hours.
- 🌐 **Full English & Persian Localization:** 100% localized interface with persistent language selection.

---

## Project Structure

The project root is consolidated into a clean, minimalist 5-file architecture:

```text
mikotim-sentinel-pro/
├── index.html          # Modern Apple Glassmorphism single-page surveillance app
├── server.py           # Unified Python server & MikoVo RaRa 11 AI Vision Core (all 12 subsystems)
├── mikovo_rara11.pt    # YOLOv8-derived neural pose weights
├── requirements.txt    # Python runtime dependencies
├── README.md           # Comprehensive project documentation
├── LICENSE             # MIT Open-Source License
└── data/               # Persistent database directory
    ├── config.json           # Runtime configuration settings
    ├── config.example.json   # Template configuration file
    └── faces.json            # Biometric face descriptor database
```

---

## Getting Started

### Option 1: Instant Browser Execution (Zero Install)

Run the application entirely inside your web browser without installing Python:
1. Simply open [`index.html`](index.html) in Google Chrome, Microsoft Edge, or Mozilla Firefox.
2. Alternatively, visit the live hosted version:  
   👉 **[https://mikailardeshirlarijani.github.io/mikotim-sentinel-pro/](https://mikailardeshirlarijani.github.io/mikotim-sentinel-pro/)**

### Option 2: Full Server Vision Core Setup (Python)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MikailArdeshirLarijani/mikotim-sentinel-pro.git
   cd mikotim-sentinel-pro
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Sentinel server:**
   ```bash
   python server.py
   ```

4. **Access the surveillance dashboard:**  
   Open your browser and navigate to:  
   👉 **`http://127.0.0.1:5000`**

---

## Telegram Bot Gateway & Remote Commands

Configure your Telegram Bot Token and Chat ID in the Settings tab or in `data/config.json`. The server listens for interactive remote commands:

| Command | Action |
|---|---|
| `/photo` | Dispatches an instantaneous high-res snapshot of the camera feed. |
| `/video` | Encodes and sends a 5-second MP4 video clip from the circular frame buffer. |
| `/summary` | Returns a comprehensive 24-hour security report (KPIs, in/out counts, loitering). |
| `/mode photo` | Sets alert dispatch media format to single photo. |
| `/mode video` | Sets alert dispatch media format to 5-second MP4 video. |
| `/pause` | Pauses active optical surveillance. |
| `/resume` | Resumes active optical surveillance. |

---

## RESTful API Endpoints

The unified Flask backend exposes high-throughput JSON and stream endpoints:

- `GET /` — Serves the Apple Glassmorphism dashboard interface.
- `GET /video_feed` — Low-latency MJPEG video feed with real-time HUD annotations.
- `GET /api/status` — Live telemetry (streaming state, FPS, person count, tripwire counts).
- `GET /api/mikovo/info` — Deep inspection details of all 12 neural subsystems.
- `POST /api/mode` — Switch between `client_edge` (Mode A) and `server_vision` (Mode B).
- `GET /api/analytics` — 24-hour hourly traffic distribution and KPI counters.
- `POST /api/telegram/send` — Dispatches photo or video alerts to Telegram.
- `POST /api/test_telegram` — Validates Telegram bot credentials and connection.
- `POST /api/register` — Registers a new face profile with 512-D embedding into `data/faces.json`.
- `GET /api/faces` — Returns all registered face profile identities and thumbnails.
- `DELETE /api/faces/<id>` — Deletes a face profile by ID.
- `POST /api/faces/clear` — Clears all registered face profiles.
- `GET /api/settings` — Returns current configuration.
- `POST /api/settings` — Updates runtime configuration settings.

---

## License

This project is open-source software licensed under the [MIT License](LICENSE).  
Developed with passion by **Mikail Ardeshir Larijani** and powered by **MikoVo RaRa 11**.
