from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .extraction import ExtractionPipeline, VisualEnrichmentPipeline
from .ingest import ingest_manifest
from .media import FFmpegMediaExtractor
from .models import QueryFilters
from .providers import (
    CLIPVisualEmbedder,
    FasterWhisperTranscriber,
    KeywordMetadataExtractor,
    OllamaVisionCaptioner,
)
from .qa import FocusedVisualRefiner, GroundedQA, OllamaGenerator
from .retrieval import Retriever
from .store import SQLiteIndex
from .web import WebConfig, run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index and retrieve multimodal evidence across videos.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(".longvideo/index.sqlite3"),
        help="SQLite index path.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create an empty evidence index.")

    ingest = commands.add_parser("ingest", help="Ingest an extraction manifest JSON file.")
    ingest.add_argument("manifest", type=Path)

    extract = commands.add_parser("extract", help="Extract an evidence manifest from local videos.")
    _add_extraction_arguments(extract)

    index_videos = commands.add_parser(
        "index-videos", help="Extract evidence from videos and ingest it into the index."
    )
    _add_extraction_arguments(index_videos)

    enrich_visual = commands.add_parser(
        "enrich-visual", help="Add visual frames/features to an existing transcript manifest."
    )
    enrich_visual.add_argument("manifest", type=Path)
    enrich_visual.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(".longvideo/artifacts"),
        help="Directory for extracted frame artifacts.",
    )
    enrich_visual.add_argument(
        "--output",
        type=Path,
        default=Path(".longvideo/multimodal_manifest.json"),
        help="Output enriched manifest JSON path.",
    )
    enrich_visual.add_argument("--segment-seconds", type=int, default=30)
    enrich_visual.add_argument("--frames-per-segment", type=int, default=5)
    enrich_visual.add_argument("--overwrite-artifacts", action="store_true")
    enrich_visual.add_argument("--vision-model")
    enrich_visual.add_argument("--visual-embedding-model")
    enrich_visual.add_argument("--vision-device", default="cpu")
    enrich_visual.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    enrich_visual.add_argument("--vision-timeout-seconds", type=int, default=900)
    enrich_visual.add_argument(
        "--checkpoint",
        action="store_true",
        help="Persist the output manifest after each processed segment.",
    )
    enrich_visual.add_argument(
        "--skip-existing-captions",
        action="store_true",
        help="Keep existing non-empty visual captions when resuming a long caption pass.",
    )
    enrich_visual.add_argument("--ingest", action="store_true")

    search = commands.add_parser("search", help="Retrieve citable evidence.")
    _add_retrieval_arguments(search, default_top_k=5)
    _add_visual_query_arguments(search)
    search.add_argument("--json", action="store_true", dest="as_json")

    ask = commands.add_parser("ask", help="Answer a question using retrieved timestamped evidence.")
    _add_retrieval_arguments(ask, default_top_k=8)
    _add_visual_query_arguments(ask)
    ask.add_argument("--model", default="qwen2.5:3b", help="Installed Ollama model name.")
    ask.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ask.add_argument("--temperature", type=float, default=0.1)
    ask.add_argument("--max-context-chars", type=int, default=18000)
    ask.add_argument("--timeout-seconds", type=int, default=900)
    ask.add_argument("--max-answer-tokens", type=int, default=256)
    ask.add_argument(
        "--show-evidence",
        action="store_true",
        help="Print retrieved timestamped evidence after the generated answer.",
    )

    serve = commands.add_parser("serve", help="Run the local chat and source-clips web UI.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--model", default="qwen2.5:3b", help="Installed Ollama model name.")
    serve.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    serve.add_argument("--temperature", type=float, default=0.1)
    serve.add_argument("--max-context-chars", type=int, default=14000)
    serve.add_argument("--timeout-seconds", type=int, default=900)
    serve.add_argument("--max-answer-tokens", type=int, default=192)
    serve.add_argument("--collection")
    serve.add_argument("--facet", action="append", default=[], metavar="KEY=VALUE")
    _add_visual_query_arguments(serve)

    commands.add_parser("stats", help="Print index record counts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _enable_unicode_console_output()
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        run_server(
            WebConfig(
                db_path=args.db,
                model=args.model,
                ollama_host=args.ollama_host,
                default_top_k=8,
                temperature=args.temperature,
                max_context_chars=args.max_context_chars,
                timeout_seconds=args.timeout_seconds,
                max_answer_tokens=args.max_answer_tokens,
                collection_id=args.collection,
                facets=_parse_facets(args.facet),
                visual_embedding_model=args.visual_embedding_model,
                vision_model=args.vision_model,
                vision_device=args.vision_device,
            ),
            host=args.host,
            port=args.port,
        )
        return 0
    with SQLiteIndex(args.db) as index:
        if args.command == "init":
            print(f"Initialized index at {index.path}")
            return 0
        if args.command == "ingest":
            stats = ingest_manifest(index, args.manifest)
            print(
                f"Ingested {stats.videos} videos and {stats.segments} segments "
                f"into {stats.collection_id}; built {stats.cross_video_links} cross-video links."
            )
            return 0
        if args.command in {"extract", "index-videos"}:
            pipeline = _extraction_pipeline(args)
            manifest = pipeline.extract(args.videos)
            output_path = pipeline.save_manifest(manifest, args.output)
            print(f"Wrote extracted evidence manifest to {output_path}")
            if args.command == "index-videos":
                stats = ingest_manifest(index, output_path)
                print(
                    f"Ingested {stats.videos} videos and {stats.segments} segments "
                    f"into {stats.collection_id}; built {stats.cross_video_links} cross-video links."
                )
            return 0
        if args.command == "enrich-visual":
            with args.manifest.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            pipeline = _visual_enrichment_pipeline(args)
            def on_progress(value: dict, complete: int, total: int) -> None:
                if args.checkpoint:
                    ExtractionPipeline.save_manifest(value, args.output)
                print(f"Processed visual segment {complete}/{total}", flush=True)

            enriched = pipeline.enrich(
                manifest,
                on_progress=on_progress,
                skip_existing_captions=args.skip_existing_captions,
            )
            output_path = ExtractionPipeline.save_manifest(enriched, args.output)
            print(f"Wrote visual evidence manifest to {output_path}")
            if args.ingest:
                stats = ingest_manifest(index, output_path)
                print(
                    f"Ingested {stats.videos} videos and {stats.segments} segments "
                    f"into {stats.collection_id}; built {stats.cross_video_links} cross-video links."
                )
            return 0
        if args.command == "stats":
            print(json.dumps(index.stats(), indent=2))
            return 0

        filters = QueryFilters(
            collection_id=args.collection,
            video_ids=tuple(args.video),
            tags=tuple(args.tag),
            facets=_parse_facets(args.facet),
            cross_video=not args.no_cross_video,
            temporal_window=max(0, args.temporal_window),
            diversify=not args.no_diversify,
        )
        retriever = Retriever(index, _visual_embedder(args))
        if args.command == "ask":
            qa = GroundedQA(
                retriever,
                OllamaGenerator(
                    model=args.model,
                    host=args.ollama_host,
                    temperature=args.temperature,
                    max_context_chars=args.max_context_chars,
                    timeout_seconds=args.timeout_seconds,
                    max_answer_tokens=args.max_answer_tokens,
                ),
                refiner=_visual_refiner(args, args.ollama_host, args.timeout_seconds),
            )
            try:
                answer = qa.ask(args.query, top_k=args.top_k, filters=filters)
            except RuntimeError as error:
                print(f"Error: {error}", file=sys.stderr)
                return 2
            print(answer.answer)
            if args.show_evidence:
                print("\nRetrieved evidence:")
                _print_result(answer.evidence)
            return 0
        result = retriever.search(args.query, top_k=args.top_k, filters=filters)
        if args.as_json:
            print(json.dumps(_result_json(result), indent=2))
        else:
            _print_result(result)
        return 0


def _enable_unicode_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def _add_retrieval_arguments(parser: argparse.ArgumentParser, default_top_k: int) -> None:
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=default_top_k)
    parser.add_argument("--collection")
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument(
        "--facet",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Require a segment or video metadata facet.",
    )
    parser.add_argument("--no-cross-video", action="store_true")
    parser.add_argument("--temporal-window", type=int, default=1)
    parser.add_argument("--no-diversify", action="store_true")


def _add_visual_query_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--visual-embedding-model",
        help="CLIP-compatible local/Hugging Face model used for visual clip retrieval.",
    )
    parser.add_argument(
        "--vision-model",
        help="Installed Ollama vision model used to refine captions for retrieved clips.",
    )
    parser.add_argument("--vision-device", default="cpu", help="Torch device for CLIP embeddings.")


def _add_extraction_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("videos", nargs="+", type=Path, help="Paths to local video files.")
    parser.add_argument("--collection-title", default="Extracted Video Collection")
    parser.add_argument("--collection-id")
    parser.add_argument(
        "--collection-facet",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Attach collection metadata used for organization.",
    )
    parser.add_argument("--video-tag", action="append", default=[])
    parser.add_argument(
        "--video-facet",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Attach shared queryable metadata to each input video.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(".longvideo/artifacts"),
        help="Directory for extracted WAV clips.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".longvideo/extracted_manifest.json"),
        help="Output evidence manifest JSON path.",
    )
    parser.add_argument("--segment-seconds", type=int, default=30)
    parser.add_argument("--overwrite-artifacts", action="store_true")
    parser.add_argument("--language", help="Optional ISO-639-1 audio language hint, e.g. en.")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--whisper-device", default="auto")
    parser.add_argument("--whisper-compute-type", default="default")
    parser.add_argument(
        "--frames-per-segment",
        type=int,
        default=5,
        help="Number of sampled frames when a visual indexing provider is enabled.",
    )
    parser.add_argument(
        "--vision-model",
        help="Installed Ollama vision model used to create rough visual captions.",
    )
    parser.add_argument(
        "--visual-embedding-model",
        help="CLIP-compatible local/Hugging Face model used to index frame features.",
    )
    parser.add_argument("--vision-device", default="cpu", help="Torch device for CLIP embeddings.")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--vision-timeout-seconds", type=int, default=900)


def _extraction_pipeline(args: argparse.Namespace) -> ExtractionPipeline:
    media = FFmpegMediaExtractor(
        artifacts_dir=args.artifacts_dir,
        segment_seconds=args.segment_seconds,
        frames_per_segment=(
            args.frames_per_segment if args.vision_model or args.visual_embedding_model else 0
        ),
        overwrite=args.overwrite_artifacts,
    )
    return ExtractionPipeline(
        media=media,
        transcriber=FasterWhisperTranscriber(
            model=args.whisper_model,
            device=args.whisper_device,
            compute_type=args.whisper_compute_type,
            language=args.language,
        ),
        metadata_extractor=KeywordMetadataExtractor(),
        collection_title=args.collection_title,
        collection_id=args.collection_id,
        collection_metadata=_parse_facets(args.collection_facet),
        video_tags=tuple(args.video_tag),
        video_metadata=_parse_facets(args.video_facet),
        visual_captioner=(
            OllamaVisionCaptioner(
                args.vision_model, host=args.ollama_host, timeout_seconds=args.vision_timeout_seconds
            )
            if args.vision_model
            else None
        ),
        visual_embedder=_visual_embedder(args),
    )


def _visual_embedder(args: argparse.Namespace) -> CLIPVisualEmbedder | None:
    model = getattr(args, "visual_embedding_model", None)
    return CLIPVisualEmbedder(model=model, device=args.vision_device) if model else None


def _visual_enrichment_pipeline(args: argparse.Namespace) -> VisualEnrichmentPipeline:
    return VisualEnrichmentPipeline(
        media=FFmpegMediaExtractor(
            artifacts_dir=args.artifacts_dir,
            segment_seconds=args.segment_seconds,
            frames_per_segment=args.frames_per_segment,
            overwrite=args.overwrite_artifacts,
        ),
        visual_captioner=(
            OllamaVisionCaptioner(
                args.vision_model, host=args.ollama_host, timeout_seconds=args.vision_timeout_seconds
            )
            if args.vision_model
            else None
        ),
        visual_embedder=_visual_embedder(args),
    )


def _visual_refiner(
    args: argparse.Namespace, host: str, timeout_seconds: int
) -> FocusedVisualRefiner | None:
    if not args.vision_model:
        return None
    return FocusedVisualRefiner(
        OllamaVisionCaptioner(args.vision_model, host=host, timeout_seconds=timeout_seconds)
    )


def _parse_facets(raw_values: list[str]) -> dict[str, str]:
    output = {}
    for raw in raw_values:
        key, separator, value = raw.partition("=")
        if not separator or not key.strip() or not value.strip():
            raise SystemExit(f"Invalid facet {raw!r}; expected KEY=VALUE.")
        output[key.strip()] = value.strip()
    return output


def _result_json(result):
    return {
        "query": result.query,
        "hits": [
            {
                "segment_id": hit.context.segment.id,
                "video_id": hit.context.video.id,
                "video_title": hit.context.video.title,
                "citation": hit.context.citation,
                "start_ms": hit.context.segment.start_ms,
                "end_ms": hit.context.segment.end_ms,
                "score": hit.score,
                "reasons": list(hit.reasons),
                "text": hit.context.segment.retrieval_text,
                "metadata": hit.context.segment.metadata,
            }
            for hit in result.hits
        ],
    }


def _print_result(result) -> None:
    if not result.hits:
        print("No matching evidence found.")
        return
    for number, hit in enumerate(result.hits, start=1):
        text = hit.context.segment.retrieval_text.replace("\n", " ")
        if len(text) > 240:
            text = text[:237] + "..."
        print(f"{number}. {hit.context.citation}  score={hit.score:.3f}")
        print(f"   why: {'; '.join(hit.reasons)}")
        print(f"   {text}")
