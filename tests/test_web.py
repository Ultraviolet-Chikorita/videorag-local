from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from longvideo_rag import SQLiteIndex, ingest_manifest_data
from longvideo_rag.web import LocalWebApplication, WebConfig, make_handler


class StubGenerator:
    def generate(self, question, evidence):
        return "Basis vectors encode a linear transformation as a matrix. [S1]"


class WebApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.video_path = root / "lesson.mp4"
        self.video_path.write_bytes(b"0123456789")
        self.db_path = root / "web.sqlite3"
        manifest = {
            "collection": {"id": "course", "title": "Course"},
            "videos": [
                {
                    "id": "video",
                    "external_id": "video",
                    "title": "Transformation Lesson",
                    "source_uri": str(self.video_path),
                    "segments": [
                        {
                            "start_ms": 1200,
                            "end_ms": 5800,
                            "transcript": (
                                "Basis vectors encode a linear transformation as a matrix."
                            ),
                            "visual_caption": "Two basis arrows are drawn on a coordinate grid.",
                        }
                    ],
                }
            ],
        }
        with SQLiteIndex(self.db_path) as index:
            ingest_manifest_data(index, manifest)
        app = LocalWebApplication(WebConfig(db_path=self.db_path), StubGenerator)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def test_static_interface_and_status_are_served(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/") as response:
            html = response.read().decode("utf-8")
        with urllib.request.urlopen(f"{self.base_url}/api/status") as response:
            status = json.loads(response.read().decode("utf-8"))
        self.assertIn("LOCAL VIDEO RAG", html.upper())
        self.assertIn("Source Clips", html)
        self.assertIn("LOCAL ANSWER | 8 CLIPS", html)
        self.assertEqual(status["stats"]["videos"], 1)
        self.assertTrue(status["ready"])

    def test_chat_returns_citable_source_clip_metadata(self) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(
                {"question": "How are basis vectors used?", "mode": "answer", "top_k": 4}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.assertIn("[S1]", payload["answer"])
        self.assertEqual(payload["sources"][0]["video_id"], "video")
        self.assertEqual(payload["sources"][0]["start_ms"], 1200)
        self.assertEqual(payload["sources"][0]["media_url"], "/media/video")
        self.assertIn("basis arrows", payload["sources"][0]["visual_caption"])

    def test_media_route_supports_byte_range_requests(self) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/media/video",
            headers={"Range": "bytes=2-5"},
        )
        with urllib.request.urlopen(request) as response:
            body = response.read()
            content_range = response.headers["Content-Range"]
            status = response.status
        self.assertEqual(status, 206)
        self.assertEqual(content_range, "bytes 2-5/10")
        self.assertEqual(body, b"2345")


if __name__ == "__main__":
    unittest.main()
