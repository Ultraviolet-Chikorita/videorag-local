from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .ids import stable_id
from .media import FFmpegMediaExtractor
from .providers import (
    MetadataExtractor,
    Transcriber,
    VisualCaptioner,
    VisualEmbedder,
    _usable_caption,
)


@dataclass(slots=True)
class ExtractionPipeline:
    media: FFmpegMediaExtractor
    transcriber: Transcriber
    metadata_extractor: MetadataExtractor
    collection_title: str
    collection_id: str | None = None
    collection_metadata: dict[str, Any] = field(default_factory=dict)
    video_tags: tuple[str, ...] = ()
    video_metadata: dict[str, Any] = field(default_factory=dict)
    visual_captioner: VisualCaptioner | None = None
    visual_embedder: VisualEmbedder | None = None

    def extract(self, video_paths: Sequence[str | Path]) -> dict[str, Any]:
        if not video_paths:
            raise ValueError("At least one video path is required.")
        collection_id = self.collection_id or stable_id("col", self.collection_title)
        videos = []
        for sequence, video_path in enumerate(video_paths, start=1):
            source = Path(video_path).expanduser().resolve()
            external_id = str(source)
            video_id = stable_id("vid", collection_id, external_id)
            media_info, artifacts = self.media.extract(source, video_id)
            segments = []
            for artifact in artifacts:
                transcript = self.transcriber.transcribe(artifact).strip()
                visual_caption = (
                    self.visual_captioner.caption(artifact.frame_paths, transcript).strip()
                    if self.visual_captioner and artifact.frame_paths
                    else ""
                )
                visual_embedding = (
                    self.visual_embedder.embed_images(artifact.frame_paths)
                    if self.visual_embedder and artifact.frame_paths
                    else ()
                )
                if not transcript and not visual_caption and not visual_embedding:
                    continue
                metadata = dict(self.metadata_extractor.extract(transcript).metadata)
                metadata["extraction"] = {
                    "transcriber": self.transcriber.name,
                    "metadata_extractor": self.metadata_extractor.name,
                    "audio_path": str(artifact.audio_path) if artifact.audio_path else None,
                    "visual_captioner": (
                        self.visual_captioner.name if self.visual_captioner else None
                    ),
                    "visual_embedder": (
                        self.visual_embedder.name if self.visual_embedder else None
                    ),
                }
                segments.append(
                    {
                        "ordinal": artifact.ordinal,
                        "start_ms": artifact.start_ms,
                        "end_ms": artifact.end_ms,
                        "transcript": transcript,
                        "visual_caption": visual_caption,
                        "frame_paths": [str(path) for path in artifact.frame_paths],
                        "visual_embedding": list(visual_embedding),
                        "metadata": metadata,
                    }
                )
            if not segments:
                raise RuntimeError(f"No usable transcript or visual caption was extracted from {source}.")
            metadata = dict(self.video_metadata)
            metadata["extraction"] = {
                "transcriber": self.transcriber.name,
                "metadata_extractor": self.metadata_extractor.name,
                "has_audio": media_info.has_audio,
                "visual_captioner": (
                    self.visual_captioner.name if self.visual_captioner else None
                ),
                "visual_embedder": (
                    self.visual_embedder.name if self.visual_embedder else None
                ),
            }
            videos.append(
                {
                    "id": video_id,
                    "external_id": external_id,
                    "title": source.stem,
                    "source_uri": str(source),
                    "sequence_index": sequence,
                    "duration_ms": media_info.duration_ms,
                    "tags": list(self.video_tags),
                    "metadata": metadata,
                    "segments": segments,
                }
            )
        return {
            "collection": {
                "id": collection_id,
                "title": self.collection_title,
                "metadata": dict(self.collection_metadata),
            },
            "videos": videos,
        }

    @staticmethod
    def save_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return output


@dataclass(slots=True)
class VisualEnrichmentPipeline:
    media: FFmpegMediaExtractor
    visual_captioner: VisualCaptioner | None = None
    visual_embedder: VisualEmbedder | None = None

    def enrich(
        self,
        manifest: dict[str, Any],
        on_progress: Callable[[dict[str, Any], int, int], None] | None = None,
        skip_existing_captions: bool = False,
    ) -> dict[str, Any]:
        enriched = deepcopy(manifest)
        total_segments = sum(len(video.get("segments") or ()) for video in enriched.get("videos") or ())
        complete = 0
        for video in enriched.get("videos") or ():
            video_id = str(video["id"])
            source = Path(str(video["source_uri"])).expanduser().resolve()
            _media_info, artifacts = self.media.extract(source, video_id)
            artifacts_by_ordinal = {artifact.ordinal: artifact for artifact in artifacts}
            for segment in video.get("segments") or ():
                ordinal = int(segment.get("ordinal", 0))
                artifact = artifacts_by_ordinal.get(ordinal)
                if artifact is None:
                    raise RuntimeError(f"No extracted media artifact exists for segment {ordinal}.")
                transcript = str(segment.get("transcript") or "")
                existing_caption = str(segment.get("visual_caption") or "").strip()
                metadata = dict(segment.get("metadata") or {})
                extraction = dict(metadata.get("extraction") or {})
                existing_captioner = str(extraction.get("visual_captioner") or "")
                should_caption = not (
                    skip_existing_captions
                    and _usable_caption(existing_caption)
                    and self.visual_captioner is not None
                    and existing_captioner == self.visual_captioner.name
                )
                caption = existing_caption
                if self.visual_captioner and artifact.frame_paths and should_caption:
                    caption = self.visual_captioner.caption(artifact.frame_paths, transcript).strip()
                embedding = (
                    self.visual_embedder.embed_images(artifact.frame_paths)
                    if self.visual_embedder and artifact.frame_paths
                    else tuple(segment.get("visual_embedding") or ())
                )
                segment["visual_caption"] = caption
                segment["frame_paths"] = [str(path) for path in artifact.frame_paths]
                segment["visual_embedding"] = list(embedding)
                extraction["visual_captioner"] = (
                    self.visual_captioner.name if self.visual_captioner else None
                )
                extraction["visual_embedder"] = (
                    self.visual_embedder.name if self.visual_embedder else None
                )
                metadata["extraction"] = extraction
                segment["metadata"] = metadata
                complete += 1
                if on_progress is not None:
                    on_progress(enriched, complete, total_segments)
        return enriched
