"""
Automated Test Suite for MikoTim Sentinel Pro Upgrade
Powered by MikoVo RaRa 11 AI Neural Vision Core
"""

import unittest
import json
import base64
import numpy as np
import cv2

# Import app components
import app as sentinel_app


class SentinelProUpgradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sentinel_app.app.testing = True
        cls.client = sentinel_app.app.test_client()

    def test_01_config_and_state(self):
        """Verify default configuration and runtime state structure."""
        self.assertIn("vision_mode", sentinel_app.config)
        self.assertIn("vision_mode", sentinel_app.state)
        self.assertIn("client_detections", sentinel_app.state)
        self.assertEqual(len(sentinel_app.state["hourly_traffic"]), 24)

    def test_02_face_database(self):
        """Test FaceDatabase CRUD operations."""
        db = sentinel_app.face_db
        test_emb = np.random.randn(512).astype(np.float32)
        test_emb /= np.linalg.norm(test_emb)
        rec = db.add("TestSubject", test_emb, "thumb_data")
        self.assertTrue(rec["id"].startswith("face_"))
        self.assertEqual(rec["name"], "TestSubject")

        # Find match
        name, score = db.find_match(test_emb)
        self.assertEqual(name, "TestSubject")
        self.assertGreaterEqual(score, 0.99)

        # Delete
        ok = db.delete(rec["id"])
        self.assertTrue(ok)

    def test_03_test_pattern_generation(self):
        """Test tactical simulated pattern generator."""
        engine = sentinel_app.detector
        frame = engine._generate_test_pattern_frame(w=640, h=480)
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_04_camera_fallback_and_snapshot(self):
        """Test bounded retry and graceful simulated fallback for offline camera."""
        engine = sentinel_app.detector
        # Open invalid camera index (e.g. 9999)
        started = engine.start(index=9999)
        self.assertTrue(started)
        self.assertTrue(engine.running)
        self.assertTrue(engine.is_simulated)

        # Snapshot getter
        raw_f, ann_f, dets, st, sim = engine.get_snapshot()
        self.assertIsInstance(dets, list)
        self.assertIsInstance(st, dict)
        self.assertTrue(sim)
        engine.stop()
        self.assertFalse(engine.running)

    def test_05_api_status(self):
        """Test GET /api/status endpoint."""
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("streaming", data)
        self.assertIn("vision_mode", data)
        self.assertIn("is_simulated", data)
        self.assertIn("hourly_traffic", data)
        self.assertEqual(len(data["hourly_traffic"]), 24)

    def test_06_api_settings_masking_and_validation(self):
        """Test secret masking on GET /api/settings and validation on POST."""
        # Set a test token in config
        sentinel_app.config["bot_token"] = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
        res = self.client.get("/api/settings")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("has_token"))
        self.assertIn("••••", data.get("bot_token"))
        masked = data.get("bot_token")

        # POST with masked token should NOT overwrite real token
        post_res = self.client.post("/api/settings", json={"bot_token": masked, "cooldown_sec": 5})
        self.assertEqual(post_res.status_code, 200)
        self.assertEqual(sentinel_app.config["bot_token"], "123456789:ABCdefGHIjklMNOpqrsTUVwxyz")

        # POST with invalid cooldown_sec should return 400
        bad_res = self.client.post("/api/settings", json={"cooldown_sec": 0.2})
        self.assertEqual(bad_res.status_code, 400)

        # POST with invalid non-JSON body
        bad_json = self.client.post("/api/settings", data="not json", content_type="text/plain")
        self.assertEqual(bad_json.status_code, 400)

    def test_07_api_mode_switching(self):
        """Test mode switching between Mode A and Mode B."""
        # Switch to client_edge
        res = self.client.post("/api/mode", json={"mode": "client_edge"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json().get("mode"), "client_edge")
        self.assertEqual(sentinel_app.state["vision_mode"], "client_edge")

        # Switch to invalid mode
        bad_res = self.client.post("/api/mode", json={"mode": "invalid_mode"})
        self.assertEqual(bad_res.status_code, 400)

        # Switch back to server_vision
        res2 = self.client.post("/api/mode", json={"mode": "server_vision"})
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.get_json().get("mode"), "server_vision")

    def test_08_api_analytics(self):
        """Test GET /api/analytics endpoint."""
        res = self.client.get("/api/analytics")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("hourly_traffic", data)
        self.assertIn("total_captures", data)
        self.assertIn("in_count", data)
        self.assertIn("out_count", data)
        self.assertIn("loitering_count", data)
        self.assertIn("peak_hour", data)

    def test_09_api_detections(self):
        """Test GET and POST /api/detections endpoint."""
        # GET server detections
        res = self.client.get("/api/detections")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("detections", data)
        self.assertIn("fps", data)

        # POST client-edge detections
        client_dets = [{"bbox": [50, 50, 100, 150], "name": "Visitor", "score": 0.95}]
        post_res = self.client.post("/api/detections", json={
            "detections": client_dets,
            "fps": 28.5,
            "count": 1
        })
        self.assertEqual(post_res.status_code, 200)
        self.assertEqual(sentinel_app.state["client_fps"], 28.5)

    def test_10_api_register_validation(self):
        """Test validation on /api/register."""
        # Missing name
        res = self.client.post("/api/register", json={})
        self.assertEqual(res.status_code, 400)

        empty_res = self.client.post("/api/register", json={"name": "   "})
        self.assertEqual(empty_res.status_code, 400)

    def test_11_api_telegram_send_validation(self):
        """Test validation on /api/telegram/send."""
        # Missing photo
        res = self.client.post("/api/telegram/send", json={"caption": "Test Alert"})
        self.assertEqual(res.status_code, 400)

        # Invalid base64
        res2 = self.client.post("/api/telegram/send", json={"photo": "!!!not_base64!!!"})
        self.assertEqual(res2.status_code, 400)


if __name__ == "__main__":
    unittest.main()
