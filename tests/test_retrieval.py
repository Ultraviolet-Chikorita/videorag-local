from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from longvideo_rag import QueryFilters, Retriever, SQLiteIndex, ingest_manifest_data


class StubVisualEmbedder:
    name = "stub-visual-embedder"

    def embed_images(self, frame_paths) -> tuple[float, ...]:
        return (1.0, 0.0)

    def embed_text(self, text) -> tuple[float, ...]:
        return (1.0, 0.0)


MANIFEST = {
    "collection": {"id": "course", "title": "Course", "metadata": {"domain": "education"}},
    "videos": [
        {
            "id": "video_a",
            "external_id": "a",
            "title": "Definitions",
            "tags": ["lecture"],
            "metadata": {"speaker": "ada"},
            "segments": [
                {
                    "start_ms": 0,
                    "end_ms": 10000,
                    "transcript": "Transcript evidence is retrieved before answer generation.",
                    "metadata": {"entities": ["hybrid retrieval"]},
                },
                {
                    "start_ms": 10000,
                    "end_ms": 20000,
                    "transcript": "The following explanation describes citation timestamps.",
                },
            ],
        },
        {
            "id": "video_b",
            "external_id": "b",
            "title": "Experiment",
            "tags": ["documentary"],
            "metadata": {"speaker": "ben"},
            "segments": [
                {
                    "start_ms": 0,
                    "end_ms": 10000,
                    "transcript": "An experiment compares recall over separate episodes.",
                    "metadata": {"entities": ["hybrid retrieval"]},
                }
            ],
        },
    ],
}


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.index = SQLiteIndex(Path(self.tmp.name) / "test.sqlite3")
        self.stats = ingest_manifest_data(self.index, MANIFEST)
        self.retriever = Retriever(self.index)

    def tearDown(self) -> None:
        self.index.close()
        self.tmp.cleanup()

    def test_ingestion_builds_cross_video_links(self) -> None:
        self.assertEqual(self.stats.cross_video_links, 1)
        self.assertEqual(self.index.stats()["segments"], 3)

    def test_reingest_updates_timestamp_without_duplicate_segment(self) -> None:
        updated = deepcopy(MANIFEST)
        updated["videos"][0]["segments"][0]["end_ms"] = 12000
        ingest_manifest_data(self.index, updated)
        self.assertEqual(self.index.stats()["segments"], 3)

    def test_cross_video_links_expand_a_text_match(self) -> None:
        result = self.retriever.search(
            "How is transcript evidence retrieved?",
            top_k=3,
            filters=QueryFilters(temporal_window=0),
        )
        video_ids = {hit.context.video.id for hit in result.hits}
        self.assertIn("video_a", video_ids)
        self.assertIn("video_b", video_ids)
        linked = next(hit for hit in result.hits if hit.context.video.id == "video_b")
        self.assertIn("cross-video shared_entity: hybrid retrieval", linked.reasons)

    def test_repeated_lowercase_phrase_builds_a_cross_video_link(self) -> None:
        phrase_manifest = {
            "collection": {"id": "phrases", "title": "Phrases"},
            "videos": [
                {
                    "id": "first",
                    "external_id": "first",
                    "title": "First",
                    "segments": [
                        {
                            "start_ms": 0,
                            "end_ms": 10000,
                            "transcript": "the last video introduced linear algebra and useful vectors",
                        }
                    ],
                },
                {
                    "id": "second",
                    "external_id": "second",
                    "title": "Second",
                    "segments": [
                        {
                            "start_ms": 0,
                            "end_ms": 10000,
                            "transcript": "the last video made linear algebra continue with a new basis",
                        }
                    ],
                },
            ],
        }
        with SQLiteIndex(Path(self.tmp.name) / "phrases.sqlite3") as phrase_index:
            stats = ingest_manifest_data(phrase_index, phrase_manifest)
            result = Retriever(phrase_index).search(
                "linear algebra vectors",
                top_k=2,
                filters=QueryFilters(temporal_window=0),
            )
        self.assertGreater(stats.cross_video_links, 0)
        self.assertIn(
            "cross-video shared_phrase: linear algebra",
            [reason for hit in result.hits for reason in hit.reasons],
        )
        self.assertNotIn(
            "cross-video shared_phrase: last video",
            [reason for hit in result.hits for reason in hit.reasons],
        )

    def test_temporal_window_adds_adjacent_context(self) -> None:
        result = self.retriever.search(
            "transcript evidence retrieved",
            top_k=3,
            filters=QueryFilters(cross_video=False, temporal_window=1),
        )
        reasons = [
            reason for hit in result.hits for reason in hit.reasons if reason.startswith("temporal")
        ]
        self.assertTrue(reasons)

    def test_concept_pairs_seed_evidence_when_terms_are_separated_in_transcript(self) -> None:
        concept_manifest = {
            "collection": {"id": "concepts", "title": "Concepts"},
            "videos": [
                {
                    "id": "determinant",
                    "external_id": "determinant",
                    "title": "Determinants",
                    "segments": [
                        {
                            "start_ms": 0,
                            "end_ms": 10000,
                            "transcript": "The determinant is exactly zero after all area collapses.",
                        }
                    ],
                }
            ],
        }
        with SQLiteIndex(Path(self.tmp.name) / "concepts.sqlite3") as concept_index:
            ingest_manifest_data(concept_index, concept_manifest)
            result = Retriever(concept_index).search(
                "How does determinant zero explain collapse?",
                top_k=1,
                filters=QueryFilters(cross_video=False, temporal_window=0),
            )
        self.assertIn("concept match: determinant zero", result.hits[0].reasons)

    def test_tag_and_facet_filters_restrict_results(self) -> None:
        result = self.retriever.search(
            "experiment compares recall",
            filters=QueryFilters(tags=("documentary",), facets={"speaker": "ben"}),
        )
        self.assertTrue(result.hits)
        self.assertTrue(all(hit.context.video.id == "video_b" for hit in result.hits))

    def test_visual_embedding_retrieval_finds_clip_without_matching_transcript(self) -> None:
        visual_manifest = {
            "collection": {"id": "visual", "title": "Visual"},
            "videos": [
                {
                    "id": "diagram",
                    "external_id": "diagram",
                    "title": "Diagram",
                    "segments": [
                        {
                            "start_ms": 0,
                            "end_ms": 10000,
                            "transcript": "",
                            "visual_caption": "",
                            "visual_embedding": [1.0, 0.0],
                        }
                    ],
                },
                {
                    "id": "board",
                    "external_id": "board",
                    "title": "Board",
                    "segments": [
                        {
                            "start_ms": 0,
                            "end_ms": 10000,
                            "transcript": "",
                            "visual_caption": "",
                            "visual_embedding": [0.0, 1.0],
                        }
                    ],
                },
            ],
        }
        with SQLiteIndex(Path(self.tmp.name) / "visual.sqlite3") as visual_index:
            ingest_manifest_data(visual_index, visual_manifest)
            result = Retriever(visual_index, StubVisualEmbedder()).search(
                "Locate the requested display",
                top_k=1,
                filters=QueryFilters(cross_video=False, temporal_window=0),
            )
        self.assertEqual(result.hits[0].context.video.id, "diagram")
        self.assertTrue(
            any(reason.startswith("visual similarity") for reason in result.hits[0].reasons)
        )


if __name__ == "__main__":
    unittest.main()
