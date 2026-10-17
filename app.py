import os
import json
import time
import base64
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, Response, render_template, request, jsonify

from detector import MikoDetector
from telegram_bot import TelegramGateway

app = Flask(__name__, template_folder="templates")
app.config["JSON_ENSURE_ASCII"] = False

CONFIG_FILE = Path(__file__).parent / "settings.json"


def get_default_config():
    return {
        "bot_token": "",
        "chat_id": "",
        "custom_caption": "هشدار تردد",
        "auto_send": True,
        "remote_control": False,
        "cooldown_sec": 6,
        "only_new_person": True,
        "sound_alert": True,
        "shift_enabled": False,
        "shift_start": "23:00",
        "shift_end": "07:00",
        "camera_index": 0,
        "language": "fa",
        "theme": "dark",
        "conf_threshold": 0.45,
    }


def load_config():
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text("utf-8"))
        except Exception:
            pass
    return get_default_config()


def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")


config = load_config()

state = {
    "is_streaming": False,
    "is_paused": False,
    "bot_token": config.get("bot_token", ""),
    "chat_id": config.get("chat_id", ""),
    "auto_send": config.get("auto_send", True),
    "remote_control": config.get("remote_control", False),
    "stats": {"fps": 0, "person_count": 0, "total_captures": 0, "telegram_sent": 0, "motion_level": 0},
    "detector": None,
    "logs": [],
}

detector = MikoDetector()
state["detector"] = detector
telegram = TelegramGateway(state)
telegram.start_polling()

_last_alert_time = 0
_tracked_ids = set()


def is_within_shift():
    if not config.get("shift_enabled"):
        return True
    try:
        now = datetime.now()
        current_minute = now.hour * 60 + now.minute
        sh, sm = map(int, config["shift_start"].split(":"))
        eh, em = map(int, config["shift_end"].split(":"))
        start_min = sh * 60 + sm
        end_min = eh * 60 + em

        if start_min <= end_min:
            return start_min <= current_minute <= end_min
        return current_minute >= start_min or current_minute <= end_min
    except Exception:
        return True


def alert_worker():
    global _last_alert_time, _tracked_ids

    while True:
        time.sleep(0.3)
        if not state["is_streaming"] or state["is_paused"]:
            continue
        if not is_within_shift():
            continue

        cooldown = config.get("cooldown_sec", 6)
        if time.time() - _last_alert_time < cooldown:
            continue

        detections = detector.last_detections
        if not detections:
            continue

        det_ids = {d["id"] for d in detections}
        has_new_person = bool(det_ids - _tracked_ids)
        if config.get("only_new_person", True) and not has_new_person:
            continue

        _last_alert_time = time.time()
        _tracked_ids |= det_ids

        snapshot = detector.take_raw_snapshot()
        if snapshot is None:
            continue

        conf = max((d["score"] for d in detections), default=0)
        count = len(detections)
        names = [d["name"] for d in detections if d.get("name")]

        state["stats"]["total_captures"] += 1
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "id": f"evt_{int(time.time() * 1000)}",
            "time": now_str,
            "count": count,
            "conf": round(conf, 2),
            "names": names,
            "sent": False,
        }
        state["logs"].insert(0, entry)
        if len(state["logs"]) > 100:
            state["logs"].pop()

        if config.get("auto_send") and state["bot_token"] and state["chat_id"]:
            caption = config.get("custom_caption", "")
            sent = telegram.dispatch_incident(snapshot, count, conf, names, caption)
            if sent:
                state["stats"]["telegram_sent"] += 1
                entry["sent"] = True


worker_thread = threading.Thread(target=alert_worker, daemon=True)
worker_thread.start()


# HTTP Endpoints

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def mjpeg_stream():
        while True:
            frame = detector.latest_annotated_jpg
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            else:
                time.sleep(0.04)

    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/start", methods=["POST"])
def api_start():
    idx = config.get("camera_index", 0)
    ok = detector.start(device_index=int(idx))
    if ok:
        state["is_streaming"] = True
        _tracked_ids.clear()
        return jsonify(ok=True)
    return jsonify(ok=False, error="Camera could not be opened"), 500


@app.route("/api/stop", methods=["POST"])
def api_stop():
    detector.stop()
    state["is_streaming"] = False
    detector.stats.update({"fps": 0, "person_count": 0, "motion_level": 0})
    return jsonify(ok=True)


@app.route("/api/status")
def api_status():
    dets = []
    for d in detector.last_detections:
        dets.append({
            "bbox": d["bbox"],
            "score": round(d["score"], 3),
            "name": d.get("name"),
            "match_score": round(d.get("match_score", 0), 3),
        })

    return jsonify(
        streaming=state["is_streaming"],
        paused=state["is_paused"],
        fps=detector.stats.get("fps", 0),
        person_count=detector.stats.get("person_count", 0),
        motion_level=detector.stats.get("motion_level", 0),
        total_captures=state["stats"]["total_captures"],
        telegram_sent=state["stats"]["telegram_sent"],
        detections=dets,
        logs=state["logs"][:40],
    )


@app.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    raw = detector.take_raw_snapshot()
    if raw is None:
        return jsonify(ok=False, error="Camera is offline"), 400
    return jsonify(ok=True, image="data:image/jpeg;base64," + base64.b64encode(raw).decode())


@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify(ok=False, error="Profile name is required"), 400

    ok, res = detector.register_face_from_frame(name)
    return jsonify(ok=ok, message=res if not ok else "Face registered", face_id=res if ok else None)


@app.route("/api/faces")
def api_faces():
    return jsonify(faces=detector.db.all_records_json())


@app.route("/api/faces/<face_id>", methods=["DELETE"])
def api_delete_face(face_id):
    deleted = detector.db.delete(face_id)
    return jsonify(ok=deleted)


@app.route("/api/faces/clear", methods=["POST"])
def api_clear_faces():
    detector.db.clear()
    return jsonify(ok=True)


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    s = {k: v for k, v in config.items() if k != "bot_token"}
    s["has_token"] = bool(config.get("bot_token"))
    return jsonify(s)


@app.route("/api/settings", methods=["POST"])
def api_save_settings():
    global config
    data = request.get_json() or {}

    for k in get_default_config():
        if k in data:
            config[k] = data[k]

    if "bot_token" in data:
        config["bot_token"] = data["bot_token"]

    state["bot_token"] = config.get("bot_token", "")
    state["chat_id"] = config.get("chat_id", "")
    state["auto_send"] = config.get("auto_send", True)
    state["remote_control"] = config.get("remote_control", False)

    save_config(config)
    return jsonify(ok=True)


@app.route("/api/test_telegram", methods=["POST"])
def api_test_telegram():
    state["bot_token"] = config.get("bot_token", "")
    state["chat_id"] = config.get("chat_id", "")
    ok, msg = telegram.test_connection()
    return jsonify(ok=ok, message=msg)


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    if hasattr(detector, "_bg_subtractor"):
        detector._bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
