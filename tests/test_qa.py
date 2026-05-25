from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from longvideo_rag import GroundedQA, QueryFilters, Retriever, SQLiteIndex, ingest_manifest_data
from longvideo_rag.qa import (
    FocusedVisualRefiner,
    OllamaGenerator,
    _align_citations,
    _is_usable_cited_answer,
    format_evidence,
)


class RecordingGenerator:
    def __init__(self) -> None:
        self.evidence = None

    def generate(self, question, evidence):
        self.evidence = evidence
        return f"Grounded response for {question} [S1]"


class StubCaptioner:
    name = "stub-captioner"

    def caption(self, frame_paths, transcript, focus=None) -> str:
        return f"A focused diagram answering {focus} is visible."


class GroundedQATests(unittest.TestCase):
    def test_answer_generation_receives_citable_retrieved_segments(self) -> None:
        manifest = {
            "collection": {"id": "course", "title": "Course"},
            "videos": [
                {
                    "id": "video",
                    "external_id": "video",
                    "title": "Lesson",
                    "segments": [
                        {
                            "start_ms": 0,
                            "end_ms": 30000,
                            "transcript": "A basis describes the vectors used for coordinates.",
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with SQLiteIndex(Path(temp_dir) / "qa.sqlite3") as index:
                ingest_manifest_data(index, manifest)
                generator = RecordingGenerator()
                result = GroundedQA(Retriever(index), generator).ask(
                    "What is a basis?",
                    filters=QueryFilters(temporal_window=0),
                )
        self.assertIn("[S1]", result.answer)
        self.assertIsNotNone(generator.evidence)
        self.assertIn("[S1] Lesson [00:00:00-00:00:30]", format_evidence(result.evidence))
        uncited_generator = OllamaGenerator()
        uncited_generator._chat = lambda prompt: "An answer without a source marker."
        fallback_answer = uncited_generator.generate("What is a basis?", result.evidence)
        self.assertIn("did not produce a citable synthesis", fallback_answer)
        self.assertIn("[S1] Lesson [00:00:00-00:00:30]", fallback_answer)
        responses = iter(
            [
                "A basis supplies vectors for coordinates in this explanation. [S1]",
                "The transcript says a basis describes vectors used for coordinates. [S1]",
            ]
        )
        audited_generator = OllamaGenerator()
        audited_generator._chat = lambda prompt: next(responses)
        audited_answer = audited_generator.generate("What is a basis?", result.evidence)
        self.assertIn("transcript says", audited_answer)

    def test_no_evidence_returns_without_calling_generator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with SQLiteIndex(Path(temp_dir) / "empty.sqlite3") as index:
                generator = RecordingGenerator()
                result = GroundedQA(Retriever(index), generator).ask("No indexed content")
        self.assertIn("No relevant", result.answer)
        self.assertIsNone(generator.evidence)

    def test_cited_answer_must_include_explanatory_text(self) -> None:
        self.assertFalse(_is_usable_cited_answer("[S1] [S2] [S3]"))
        self.assertTrue(
            _is_usable_cited_answer(
                "Basis vectors are the vectors that coordinate scalars scale. [S1]"
            )
        )

    def test_alignment_moves_claim_to_matching_evidence_label(self) -> None:
        manifest = {
            "collection": {"id": "math", "title": "Math"},
            "videos": [
                {
                    "id": "null",
                    "external_id": "null",
                    "title": "Null",
                    "segments": [
                        {
                            "start_ms": 0,
                            "end_ms": 10000,
                            "transcript": "Null space contains vectors that land on the origin.",
                        }
                    ],
                },
                {
                    "id": "determinant",
                    "external_id": "determinant",
                    "title": "Determinant",
                    "segments": [
                        {
                            "start_ms": 0,
                            "end_ms": 10000,
                            "transcript": "The determinant is zero when space collapses onto a line.",
                        }
                    ],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with SQLiteIndex(Path(temp_dir) / "align.sqlite3") as index:
                ingest_manifest_data(index, manifest)
                evidence = Retriever(index).search(
                    "determinant zero null space line",
                    top_k=2,
                    filters=QueryFilters(cross_video=False, temporal_window=0),
                )
        determinant_label = next(
            f"[S{number}]"
            for number, hit in enumerate(evidence.hits, start=1)
            if hit.context.video.id == "determinant"
        )
        aligned = _align_citations(
            "- The determinant is zero when space collapses onto a line. [S99]",
            evidence,
        )
        self.assertIn(determinant_label, aligned)

    def test_visual_caption_is_in_answer_context_and_can_be_refined(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            frame = Path(temp_dir) / "frame.jpg"
            frame.write_bytes(b"frame")
            manifest = {
                "collection": {"id": "visual", "title": "Visual"},
                "videos": [
                    {
                        "id": "clip",
                        "external_id": "clip",
                        "title": "Clip",
                        "segments": [
                            {
                                "start_ms": 0,
                                "end_ms": 10000,
                                "transcript": "The speaker refers to the illustration.",
                                "visual_caption": "A rough illustration is visible.",
                                "frame_paths": [str(frame)],
                            }
                        ],
                    }
                ],
            }
            with SQLiteIndex(Path(temp_dir) / "visual.sqlite3") as index:
                ingest_manifest_data(index, manifest)
                result = GroundedQA(
                    Retriever(index),
                    RecordingGenerator(),
                    refiner=FocusedVisualRefiner(StubCaptioner()),
                ).ask("What illustration is visible?", filters=QueryFilters(temporal_window=0))
        formatted = format_evidence(result.evidence)
        self.assertIn(
            "Visual caption: A focused diagram answering What illustration is visible? is visible.",
            formatted,
        )


if __name__ == "__main__":
    unittest.main()
