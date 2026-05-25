from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .models import QueryFilters, RetrievalHit, RetrievalResult, SegmentContext
from .providers import VisualEmbedder
from .store import SQLiteIndex


_STOP_WORDS = {
    "a",
    "across",
    "all",
    "an",
    "and",
    "appearance",
    "are",
    "as",
    "being",
    "both",
    "cases",
    "connect",
    "do",
    "does",
    "explain",
    "failure",
    "for",
    "from",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "make",
    "of",
    "on",
    "possible",
    "the",
    "that",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
    "video",
    "videos",
}


@dataclass(slots=True)
class _Candidate:
    context: SegmentContext
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def add(self, score: float, reason: str) -> None:
        self.score += score
        if reason not in self.reasons:
            self.reasons.append(reason)


class Retriever:
    def __init__(self, index: SQLiteIndex, visual_embedder: VisualEmbedder | None = None):
        self.index = index
        self.visual_embedder = visual_embedder

    def search(
        self, query: str, top_k: int = 5, filters: QueryFilters | None = None
    ) -> RetrievalResult:
        filters = filters or QueryFilters()
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        candidates: dict[str, _Candidate] = {}

        if self.visual_embedder is not None:
            query_embedding = self.visual_embedder.embed_text(query)
            for context, similarity in self.index.search_visual(
                query_embedding, filters, max(top_k * 4, 10)
            ):
                if similarity > 0:
                    self._add(
                        candidates,
                        context,
                        similarity * 0.9,
                        f"visual similarity {similarity:.3f}",
                    )

        fts_query = _fts_query(query)
        if fts_query:
            for rank, context in enumerate(
                self.index.search_lexical(fts_query, filters, max(top_k * 8, 20)), start=1
            ):
                self._add(candidates, context, 1.0 / rank, f"text match rank {rank}")

        for label, concept_query in _concept_queries(query):
            concept_videos: set[str] = set()
            for rank, context in enumerate(
                self.index.search_lexical(concept_query, filters, max(top_k, 6)), start=1
            ):
                if context.video.id in concept_videos:
                    continue
                self._add(
                    candidates,
                    context,
                    0.55 / (len(concept_videos) + 1),
                    f"concept match: {label}",
                )
                concept_videos.add(context.video.id)
                if len(concept_videos) >= 3:
                    break

        for rank, (context, entity) in enumerate(
            self.index.search_entity_mentions(query, filters, max(top_k * 4, 10)), start=1
        ):
            self._add(candidates, context, 0.7 / rank, f"query entity: {entity}")

        seeds = sorted(candidates.values(), key=lambda item: item.score, reverse=True)[
            : max(top_k, 5)
        ]
        base_scores = {seed.context.segment.id: seed.score for seed in seeds}

        if filters.cross_video:
            for context, link in self.index.linked_contexts(base_scores, filters):
                linked_score = base_scores[link.source_segment_id] * link.weight * 0.55
                self._add(
                    candidates,
                    context,
                    linked_score,
                    f"cross-video {link.relation_type}: {link.relation_key}",
                )

        if filters.temporal_window > 0:
            for seed in seeds:
                for context, distance in self.index.neighbor_contexts(
                    seed.context, filters.temporal_window, filters
                ):
                    score = seed.score * 0.3 / distance
                    self._add(
                        candidates,
                        context,
                        score,
                        f"temporal neighbor of {seed.context.citation}",
                    )

        selected = self._select(candidates, top_k, filters.diversify)
        hits = tuple(
            RetrievalHit(
                context=candidate.context,
                score=round(candidate.score, 6),
                reasons=tuple(candidate.reasons),
            )
            for candidate in selected
        )
        return RetrievalResult(query=query, hits=hits)

    @staticmethod
    def _add(
        candidates: dict[str, _Candidate],
        context: SegmentContext,
        score: float,
        reason: str,
    ) -> None:
        candidate = candidates.setdefault(context.segment.id, _Candidate(context=context))
        candidate.add(score, reason)

    @staticmethod
    def _select(
        candidates: dict[str, _Candidate], top_k: int, diversify: bool
    ) -> list[_Candidate]:
        remaining = list(candidates.values())
        selected: list[_Candidate] = []
        video_counts: Counter[str] = Counter()
        while remaining and len(selected) < top_k:
            def selection_score(candidate: _Candidate) -> float:
                if not diversify:
                    return candidate.score
                repeats = video_counts[candidate.context.video.id]
                return candidate.score * max(0.45, 1.0 - (repeats * 0.25))

            chosen = max(remaining, key=selection_score)
            remaining.remove(chosen)
            selected.append(chosen)
            video_counts[chosen.context.video.id] += 1
        return selected


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query.casefold())
    meaningful = [token for token in tokens if token not in _STOP_WORDS]
    chosen = meaningful or tokens
    return " OR ".join(f'"{token}"' for token in dict.fromkeys(chosen))


def _concept_queries(query: str) -> list[tuple[str, str]]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query.casefold())
    meaningful = [token for token in tokens if token not in _STOP_WORDS]
    output = []
    for left, right in zip(meaningful, meaningful[1:]):
        label = f"{left} {right}"
        output.append((label, f'"{left}" AND "{right}"'))
    return output
