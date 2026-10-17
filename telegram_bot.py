import threading
import time
import requests
from datetime import datetime


class TelegramGateway:
    def __init__(self, state_ref):
        self.state = state_ref
        self._last_update_id = 0
        self._poll_thread = None
        self._poll_active = False

    def start_polling(self):
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._poll_active = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def stop_polling(self):
        self._poll_active = False

    def _poll_loop(self):
        while self._poll_active:
            try:
                token = self.state.get("bot_token", "").strip()
                chat_id = self.state.get("chat_id", "").strip()
                if not token or not chat_id or not self.state.get("remote_control"):
                    time.sleep(4)
                    continue

                url = f"https://api.telegram.org/bot{token}/getUpdates"
                resp = requests.get(url, params={"offset": self._last_update_id + 1, "timeout": 3}, timeout=6)
                data = resp.json()

                if data.get("ok") and data.get("result"):
                    for update in data["result"]:
                        self._last_update_id = update["update_id"]
                        msg = update.get("message", {})
                        text = (msg.get("text") or "").strip().lower()
                        incoming_chat = str(msg.get("chat", {}).get("id", ""))

                        if incoming_chat != str(chat_id):
                            continue

                        if text in ("/photo", "عکس", "photo"):
                            self._handle_photo(token, chat_id)
                        elif text in ("/status", "وضعیت", "status"):
                            self._handle_status(token, chat_id)
                        elif text in ("/pause", "توقف", "pause"):
                            self.state["is_paused"] = True
                            self._send_message(token, chat_id, "⏸ Monitoring paused.")
                        elif text in ("/resume", "ادامه", "resume"):
                            self.state["is_paused"] = False
                            self._send_message(token, chat_id, "▶ Monitoring resumed.")
                        elif text in ("/help", "help", "راهنما"):
                            self._send_message(token, chat_id,
                                "MikoTim Commands:\n"
                                "/photo - Capture live photo\n"
                                "/status - System status\n"
                                "/pause - Pause alert loop\n"
                                "/resume - Resume alert loop")
            except Exception:
                pass
            time.sleep(3.5)

    def _handle_photo(self, token, chat_id):
        det = self.state.get("detector")
        if det is None or not self.state.get("is_streaming"):
            self._send_message(token, chat_id, "Camera is currently offline.")
            return

        raw = det.take_raw_snapshot()
        if raw:
            caption = f"Manual snapshot | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            self._send_photo(token, chat_id, raw, caption)
        else:
            self._send_message(token, chat_id, "Frame not ready.")

    def _handle_status(self, token, chat_id):
        stats = self.state.get("stats", {})
        streaming = self.state.get("is_streaming", False)
        paused = self.state.get("is_paused", False)
        text = (
            f"Sentinel Status:\n"
            f"- Camera: {'Active' if streaming else 'Offline'}\n"
            f"- State: {'Paused' if paused else 'Monitoring'}\n"
            f"- FPS: {stats.get('fps', 0)}\n"
            f"- Detected persons: {stats.get('person_count', 0)}\n"
            f"- Captures: {stats.get('total_captures', 0)}\n"
            f"- Alerts sent: {stats.get('telegram_sent', 0)}\n"
            f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._send_message(token, chat_id, text)

    def dispatch_incident(self, jpeg_bytes: bytes, person_count: int,
                          confidence: float, recognized_names: list[str],
                          custom_caption: str = "") -> bool:
        token = self.state.get("bot_token", "").strip()
        chat_id = self.state.get("chat_id", "").strip()
        if not token or not chat_id or not self.state.get("auto_send"):
            return False

        names_str = ", ".join(recognized_names) if recognized_names else "Unknown"
        caption = (
            f"Alert: Movement Detected\n"
            f"{custom_caption + chr(10) if custom_caption else ''}"
            f"Persons: {person_count}\n"
            f"Confidence: {confidence:.0%}\n"
            f"Identified: {names_str}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return self._send_photo(token, chat_id, jpeg_bytes, caption)

    def test_connection(self) -> tuple[bool, str]:
        token = self.state.get("bot_token", "").strip()
        chat_id = self.state.get("chat_id", "").strip()
        if not token or not chat_id:
            return False, "Bot token or Chat ID is empty."
        ok = self._send_message(token, chat_id, "MikoTim connection test successful.")
        return ok, "Connection successful." if ok else "Failed to send message."

    @staticmethod
    def _send_message(token, chat_id, text) -> bool:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=8,
            )
            return r.json().get("ok", False)
        except Exception:
            return False

    @staticmethod
    def _send_photo(token, chat_id, jpeg_bytes: bytes, caption: str) -> bool:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendPhoto",
                data={"chat_id": chat_id, "caption": caption},
                files={"photo": ("capture.jpg", jpeg_bytes, "image/jpeg")},
                timeout=15,
            )
            return r.json().get("ok", False)
        except Exception:
            return False
