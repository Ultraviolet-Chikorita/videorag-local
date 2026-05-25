from __future__ import annotations

import json
import mimetypes
import re
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote, urlparse

from .models import QueryFilters, RetrievalResult
from .providers import CLIPVisualEmbedder, OllamaVisionCaptioner
from .qa import AnswerGenerator, FocusedVisualRefiner, GroundedQA, OllamaGenerator
from .retrieval import Retriever
from .store import SQLiteIndex


_STATIC_DIRECTORY = Path(__file__).with_name("web_static")
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/assets/app.css": ("app.css", "text/css; charset=utf-8"),
    "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
_RANGE_PATTERN = re.compile(r"bytes=(\d*)-(\d*)$")


@dataclass(frozen=True, slots=True)
class WebConfig:
    db_path: Path
    model: str = "qwen2.5:3b"
    ollama_host: str = "http://127.0.0.1:11434"
    default_top_k: int = 8
    temperature: float = 0.1
    max_context_chars: int = 14000
    timeout_seconds: int = 900
    max_answer_tokens: int = 192
    collection_id: str | None = None
    facets: dict[str, str] = field(default_factory=dict)
    visual_embedding_model: str | None = None
    vision_model: str | None = None
    vision_device: str = "cpu"


class LocalWebApplication:
    def __init__(
        self,
        config: WebConfig,
        generator_factory: Callable[[], AnswerGenerator] | None = None,
    ) -> None:
        self.config = config
        self._generator_factory = generator_factory or self._default_generator
        self._visual_embedder = (
            CLIPVisualEmbedder(config.visual_embedding_model, config.vision_device)
            if config.visual_embedding_model
            else None
        )
        self._visual_refiner = (
            FocusedVisualRefiner(
                OllamaVisionCaptioner(
                    config.vision_model,
                    host=config.ollama_host,
                    timeout_seconds=config.timeout_seconds,
                )
            )
            if config.vision_model
            else None
        )

    def status(self) -> dict[str, Any]:
        with SQLiteIndex(self.config.db_path) as index:
            stats = index.stats()
        return {
            "ready": stats["segments"] > 0,
            "stats": stats,
            "model": self.config.model,
            "database": self.config.db_path.name,
        }

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question") or "").strip()
        if not question:
            raise ValueError("Enter a question before sending.")
        if len(question) > 4000:
            raise ValueError("Question is too long.")
        mode = "answer"
        top_k = self.config.default_top_k
        filters = QueryFilters(
            collection_id=self.config.collection_id,
            facets=dict(self.config.facets),
            temporal_window=1,
            cross_video=True,
            diversify=True,
        )
        with SQLiteIndex(self.config.db_path) as index:
            retriever = Retriever(index, self._visual_embedder)
            result = GroundedQA(
                retriever, self._generator_factory(), refiner=self._visual_refiner
            ).ask(
                question, top_k=top_k, filters=filters
            )
            evidence = result.evidence
            answer = result.answer
        return {
            "question": question,
            "mode": mode,
            "answer": answer,
            "sources": _source_payload(evidence),
        }

    def media_path(self, video_id: str) -> Path | None:
        with SQLiteIndex(self.config.db_path) as index:
            video = index.get_video(video_id)
        if video is None or not video.source_uri:
            return None
        media_path = Path(video.source_uri).expanduser()
        if not media_path.is_absolute():
            media_path = (Path.cwd() / media_path).resolve()
        return media_path if media_path.is_file() else None

    def static_asset(self, route: str) -> tuple[Path, str] | None:
        configured = _STATIC_FILES.get(route)
        if configured is None:
            return None
        filename, content_type = configured
        asset = _STATIC_DIRECTORY / filename
        return (asset, content_type) if asset.is_file() else None

    def _default_generator(self) -> AnswerGenerator:
        return OllamaGenerator(
            model=self.config.model,
            host=self.config.ollama_host,
            temperature=self.config.temperature,
            max_context_chars=self.config.max_context_chars,
            timeout_seconds=self.config.timeout_seconds,
            max_answer_tokens=self.config.max_answer_tokens,
        )


def run_server(config: WebConfig, host: str = "127.0.0.1", port: int = 8787) -> None:
    application = LocalWebApplication(config)
    server = ThreadingHTTPServer((host, port), make_handler(application))
    print(f"Local Video RAG running at http://{host}:{port}")
    print(f"Using index {config.db_path} and local model {config.model}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def make_handler(application: LocalWebApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "LocalVideoRAG/0.1"

        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route == "/api/status":
                self._send_json(application.status())
                return
            if route.startswith("/media/"):
                video_id = unquote(route.removeprefix("/media/"))
                media_path = application.media_path(video_id)
                if media_path is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "Source video is unavailable.")
                    return
                self._send_media(media_path, include_body=True)
                return
            static_asset = application.static_asset(route)
            if static_asset is not None:
                self._send_file(*static_asset)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_HEAD(self) -> None:
            route = urlparse(self.path).path
            if route.startswith("/media/"):
                media_path = application.media_path(unquote(route.removeprefix("/media/")))
                if media_path is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "Source video is unavailable.")
                    return
                self._send_media(media_path, include_body=False)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/chat":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length > 1_000_000:
                    raise ValueError("Request body is too large.")
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Invalid request payload.")
                self._send_json(application.chat(payload))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            except RuntimeError as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.SERVICE_UNAVAILABLE)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_json(self, value: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str) -> None:
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _send_media(self, path: Path, include_body: bool) -> None:
            size = path.stat().st_size
            start, end, partial = _requested_bytes(self.headers.get("Range"), size)
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not include_body:
                return
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    block = handle.read(min(1024 * 256, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

    return Handler


def _source_payload(evidence: RetrievalResult) -> list[dict[str, Any]]:
    sources = []
    for number, hit in enumerate(evidence.hits, start=1):
        sources.append(
            {
                "label": f"S{number}",
                "video_id": hit.context.video.id,
                "title": hit.context.video.title,
                "citation": hit.context.citation,
                "start_ms": hit.context.segment.start_ms,
                "end_ms": hit.context.segment.end_ms,
                "transcript": hit.context.segment.transcript,
                "visual_caption": hit.context.segment.visual_caption,
                "reasons": list(hit.reasons),
                "media_url": f"/media/{quote(hit.context.video.id, safe='')}",
            }
        )
    return sources


def _requested_bytes(header: str | None, size: int) -> tuple[int, int, bool]:
    if not header:
        return 0, size - 1, False
    match = _RANGE_PATTERN.fullmatch(header.strip())
    if match is None:
        return 0, size - 1, False
    first, last = match.groups()
    if not first:
        suffix = min(size, max(1, int(last or "1")))
        return size - suffix, size - 1, True
    start = min(max(0, int(first)), size - 1)
    end = min(int(last), size - 1) if last else size - 1
    if end < start:
        end = start
    return start, end, True
