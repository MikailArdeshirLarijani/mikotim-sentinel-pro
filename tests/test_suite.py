"""
Unit and integration tests for MikoTim Sentinel Pro Python edition.
"""
import os
import sys
import json
import tempfile
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector import FaceDatabase, _cosine_similarity, MikoDetector
from telegram_bot import TelegramGateway
from app import app


class TestFaceDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()
        self.db = FaceDatabase(db_path=self.temp_file.name)

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_add_and_find_exact(self):
        vec = np.random.randn(128).astype(np.float32)
        vec /= np.linalg.norm(vec)

        face_id = self.db.add("Ali", vec)
        self.assertIsNotNone(face_id)
        self.assertEqual(len(self.db.faces), 1)

        # Query with exact vector
        found_name, score = self.db.find(vec)
        self.assertEqual(found_name, "Ali")
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_find_no_match_orthogonal(self):
        vec1 = np.zeros(128, dtype=np.float32)
        vec1[0] = 1.0

        vec2 = np.zeros(128, dtype=np.float32)
        vec2[1] = 1.0

        self.db.add("Reza", vec1)
        found_name, score = self.db.find(vec2)
        self.assertIsNone(found_name)
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_delete(self):
        vec = np.random.randn(128).astype(np.float32)
        vec /= np.linalg.norm(vec)
        face_id = self.db.add("Sara", vec)
        self.assertEqual(len(self.db.faces), 1)

        ok = self.db.delete(face_id)
        self.assertTrue(ok)
        self.assertEqual(len(self.db.faces), 0)

    def test_clear(self):
        vec = np.random.randn(128).astype(np.float32)
        self.db.add("P1", vec)
        self.db.add("P2", vec)
        self.assertEqual(len(self.db.faces), 2)
        self.db.clear()
        self.assertEqual(len(self.db.faces), 0)


class TestMathAndEmbeddings(unittest.TestCase):
    def test_cosine_similarity_identical(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sim = _cosine_similarity(a, a)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_cosine_similarity_opposite(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        sim = _cosine_similarity(a, b)
        self.assertAlmostEqual(sim, -1.0, places=5)

    def test_cosine_similarity_zeros(self):
        a = np.zeros(10, dtype=np.float32)
        b = np.zeros(10, dtype=np.float32)
        sim = _cosine_similarity(a, b)
        self.assertEqual(sim, 0.0)


class TestTelegramGateway(unittest.TestCase):
    def test_empty_token_fails_gracefully(self):
        state = {
            "bot_token": "",
            "chat_id": "",
            "is_streaming": False,
            "is_paused": False
        }
        gw = TelegramGateway(state)
        ok, msg = gw.test_connection()
        self.assertFalse(ok)
        self.assertIn("Bot token", msg)

        # Dispatch should also return False
        raw_dummy = b"\xff\xd8\xff\xe0"
        dispatched = gw.dispatch_incident(raw_dummy, 1, 0.95, ["Test"], "Alert")
        self.assertFalse(dispatched)


class TestFlaskEndpoints(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_index_route(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"MikoTim Sentinel Pro", res.data)

    def test_status_endpoint(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("streaming", data)
        self.assertIn("fps", data)
        self.assertIn("detections", data)

    def test_settings_get_and_post(self):
        res = self.client.get("/api/settings")
        self.assertEqual(res.status_code, 200)

        # Update caption
        res_post = self.client.post("/api/settings", json={"custom_caption": "تست امنیتی"})
        self.assertEqual(res_post.status_code, 200)
        self.assertTrue(res_post.get_json()["ok"])

    def test_faces_list_and_clear(self):
        res = self.client.get("/api/faces")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("faces", data)

        res_clear = self.client.post("/api/faces/clear")
        self.assertEqual(res_clear.status_code, 200)


if __name__ == "__main__":
    unittest.main()
