from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ids import stable_id
from .models import Collection, Segment, Video
from .store import SQLiteIndex


@dataclass(frozen=True, slots=True)
class IngestStats:
    collection_id: str
    videos: int
    segments: int
    cross_video_links: int


def ingest_manifest(index: SQLiteIndex, path: str | Path) -> IngestStats:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return ingest_manifest_data(index, data, fallback_name=manifest_path.stem)


def ingest_manifest_data(
    index: SQLiteIndex, data: dict[str, Any], fallback_name: str = "collection"
) -> IngestStats:
    collection_data = data.get("collection") or {}
    collection_title = str(collection_data.get("title") or fallback_name)
    collection_id = str(
        collection_data.get("id") or stable_id("col", collection_title, fallback_name)
    )
    collection = Collection(
        id=collection_id,
        title=collection_title,
        metadata=dict(collection_data.get("metadata") or {}),
    )
    index.upsert_collection(collection)

    video_count = 0
    segment_count = 0
    for position, video_data in enumerate(data.get("videos") or (), start=1):
        external_id = str(
            video_data.get("external_id")
            or video_data.get("source_uri")
            or video_data.get("title")
            or f"video-{position}"
        )
        video_id = str(video_data.get("id") or stable_id("vid", collection_id, external_id))
        video = Video(
            id=video_id,
            collection_id=collection_id,
            external_id=external_id,
            title=str(video_data.get("title") or external_id),
            source_uri=str(video_data.get("source_uri") or ""),
            sequence_index=video_data.get("sequence_index", position),
            duration_ms=video_data.get("duration_ms"),
            tags=tuple(str(tag) for tag in video_data.get("tags") or ()),
            metadata=dict(video_data.get("metadata") or {}),
        )
        index.upsert_video(video)
        video_count += 1

        for ordinal, segment_data in enumerate(video_data.get("segments") or ()):
            start_ms = int(segment_data["start_ms"])
            end_ms = int(segment_data["end_ms"])
            segment_ordinal = int(segment_data.get("ordinal", ordinal))
            segment = Segment(
                id=str(
                    segment_data.get("id")
                    or stable_id("seg", video_id, segment_ordinal)
                ),
                video_id=video_id,
                ordinal=segment_ordinal,
                start_ms=start_ms,
                end_ms=end_ms,
                transcript=str(segment_data.get("transcript") or ""),
                visual_caption=str(segment_data.get("visual_caption") or ""),
                frame_paths=tuple(str(path) for path in segment_data.get("frame_paths") or ()),
                visual_embedding=tuple(
                    float(value) for value in segment_data.get("visual_embedding") or ()
                ),
                tags=tuple(str(tag) for tag in segment_data.get("tags") or ()),
                metadata=dict(segment_data.get("metadata") or {}),
            )
            if not segment.retrieval_text and not segment.visual_embedding:
                raise ValueError(
                    f"Segment {segment.id} has no transcript, visual caption, or visual embedding."
                )
            index.upsert_segment(segment)
            segment_count += 1

    links = index.rebuild_entity_links()
    return IngestStats(
        collection_id=collection_id,
        videos=video_count,
        segments=segment_count,
        cross_video_links=links,
    )
