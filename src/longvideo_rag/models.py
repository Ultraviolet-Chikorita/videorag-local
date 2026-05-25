from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Metadata = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Collection:
    id: str
    title: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Video:
    id: str
    collection_id: str
    external_id: str
    title: str
    source_uri: str = ""
    sequence_index: int | None = None
    duration_ms: int | None = None
    tags: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Segment:
    id: str
    video_id: str
    ordinal: int
    start_ms: int
    end_ms: int
    transcript: str = ""
    visual_caption: str = ""
    frame_paths: tuple[str, ...] = ()
    visual_embedding: tuple[float, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)

    @property
    def retrieval_text(self) -> str:
        blocks = []
        if self.visual_caption.strip():
            blocks.append(f"Visual caption: {self.visual_caption.strip()}")
        if self.transcript.strip():
            blocks.append(f"Transcript: {self.transcript.strip()}")
        return "\n".join(blocks)


@dataclass(frozen=True, slots=True)
class SegmentContext:
    collection: Collection
    video: Video
    segment: Segment

    @property
    def citation(self) -> str:
        start = format_timestamp(self.segment.start_ms)
        end = format_timestamp(self.segment.end_ms)
        return f"{self.video.title} [{start}-{end}]"


@dataclass(frozen=True, slots=True)
class SegmentLink:
    source_segment_id: str
    target_segment_id: str
    relation_type: str
    relation_key: str
    weight: float


@dataclass(frozen=True, slots=True)
class QueryFilters:
    collection_id: str | None = None
    video_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    facets: dict[str, str] = field(default_factory=dict)
    cross_video: bool = True
    temporal_window: int = 1
    diversify: bool = True


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    context: SegmentContext
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    hits: tuple[RetrievalHit, ...]


def format_timestamp(milliseconds: int) -> str:
    seconds = max(0, milliseconds // 1000)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
