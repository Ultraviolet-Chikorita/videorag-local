from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from longvideo_rag import (
    ExtractionPipeline,
    QueryFilters,
    Retriever,
    SQLiteIndex,
    VisualEnrichmentPipeline,
    ingest_manifest_data,
)
from longvideo_rag.media import FFmpegMediaExtractor
from longvideo_rag.providers import KeywordMetadataExtractor


class StubTranscriber:
    name = "stub-transcript"

    def transcribe(self, artifact) -> str:
        return f"Project Atlas appears in clip {artifact.ordinal} of {artifact.video_path.stem}."


class StubCaptioner:
    name = "stub-captioner"

    def caption(self, frame_paths, transcript, focus=None) -> str:
        return f"A {frame_paths[0].parent.parent.name} diagram is visible."


class StubVisualEmbedder:
    name = "stub-visual-embedder"

    def embed_images(self, frame_paths) -> tuple[float, ...]:
        return (1.0, 0.0)

    def embed_text(self, text) -> tuple[float, ...]:
        return (1.0, 0.0)


class ExtractionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.video_a = self._make_video("episode-a.mp4", "blue")
        self.video_b = self._make_video("episode-b.mp4", "red")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_raw_video_extracts_artifacts_and_indexes_cross_video_evidence(self) -> None:
        media = FFmpegMediaExtractor(
            artifacts_dir=self.root / "artifacts",
            segment_seconds=1,
            frames_per_segment=2,
        )
        pipeline = ExtractionPipeline(
            media=media,
            transcriber=StubTranscriber(),
            metadata_extractor=KeywordMetadataExtractor(),
            collection_title="Synthetic Episodes",
            visual_captioner=StubCaptioner(),
            visual_embedder=StubVisualEmbedder(),
        )
        manifest = pipeline.extract([self.video_a, self.video_b])
        self.assertEqual(len(manifest["videos"]), 2)
        self.assertEqual(len(manifest["videos"][0]["segments"]), 2)
        extraction = manifest["videos"][0]["segments"][0]["metadata"]["extraction"]
        self.assertTrue(Path(extraction["audio_path"]).is_file())
        self.assertEqual(extraction["visual_captioner"], "stub-captioner")
        self.assertEqual(len(manifest["videos"][0]["segments"][0]["frame_paths"]), 2)
        self.assertTrue(list((self.root / "artifacts").rglob("*.jpg")))
        self.assertIn("diagram is visible", manifest["videos"][0]["segments"][0]["visual_caption"])
        self.assertEqual(manifest["videos"][0]["segments"][0]["visual_embedding"], [1.0, 0.0])

        with SQLiteIndex(self.root / "index.sqlite3") as index:
            stats = ingest_manifest_data(index, manifest)
            result = Retriever(index, StubVisualEmbedder()).search(
                "What happens with Project Atlas?",
                top_k=3,
                filters=QueryFilters(temporal_window=0),
            )
        self.assertGreater(stats.cross_video_links, 0)
        self.assertEqual({hit.context.video.title for hit in result.hits}, {"episode-a", "episode-b"})

    def test_visual_enrichment_reuses_existing_transcript_manifest(self) -> None:
        manifest = {
            "collection": {"id": "existing", "title": "Existing"},
            "videos": [
                {
                    "id": "video",
                    "external_id": "video",
                    "title": "Video",
                    "source_uri": str(self.video_a),
                    "segments": [
                        {
                            "ordinal": 0,
                            "start_ms": 0,
                            "end_ms": 1000,
                            "transcript": "Already transcribed text.",
                        },
                        {
                            "ordinal": 1,
                            "start_ms": 1000,
                            "end_ms": 1600,
                            "transcript": "Second existing segment.",
                        },
                    ],
                }
            ],
        }
        pipeline = VisualEnrichmentPipeline(
            media=FFmpegMediaExtractor(
                artifacts_dir=self.root / "visual-artifacts",
                segment_seconds=1,
                frames_per_segment=1,
            ),
            visual_captioner=StubCaptioner(),
            visual_embedder=StubVisualEmbedder(),
        )
        progress = []
        enriched = pipeline.enrich(
            manifest,
            on_progress=lambda _value, complete, total: progress.append((complete, total)),
        )
        first = enriched["videos"][0]["segments"][0]
        self.assertEqual(first["transcript"], "Already transcribed text.")
        self.assertTrue(Path(first["frame_paths"][0]).is_file())
        self.assertEqual(first["visual_embedding"], [1.0, 0.0])
        self.assertIn("diagram is visible", first["visual_caption"])
        self.assertEqual(progress, [(1, 2), (2, 2)])
        resumed = pipeline.enrich(enriched, skip_existing_captions=True)
        self.assertEqual(
            resumed["videos"][0]["segments"][0]["visual_caption"], first["visual_caption"]
        )
        resumed["videos"][0]["segments"][0]["visual_caption"] = "[0.01, 0.3, 0.99, 0.46]"
        repaired = pipeline.enrich(resumed, skip_existing_captions=True)
        self.assertIn("diagram is visible", repaired["videos"][0]["segments"][0]["visual_caption"])
        alternate = VisualEnrichmentPipeline(
            media=pipeline.media,
            visual_captioner=type(
                "AlternateCaptioner",
                (),
                {
                    "name": "different-captioner",
                    "caption": lambda _self, _paths, _transcript, focus=None: "Replacement diagram caption.",
                },
            )(),
        )
        switched = alternate.enrich(enriched, skip_existing_captions=True)
        self.assertEqual(switched["videos"][0]["segments"][0]["visual_caption"], "Replacement diagram caption.")

    def _make_video(self, filename: str, color: str) -> Path:
        path = self.root / filename
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=160x90:r=10:d=1.6",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=16000:duration=1.6",
                "-shortest",
                "-c:v",
                "mpeg4",
                "-c:a",
                "aac",
                str(path),
            ],
            check=True,
        )
        return path


if __name__ == "__main__":
    unittest.main()
