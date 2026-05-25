from __future__ import annotations

import base64
import io
import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .media import SegmentArtifact


@dataclass(frozen=True, slots=True)
class TranscriptMetadata:
    metadata: dict[str, Any] = field(default_factory=dict)


class Transcriber(Protocol):
    name: str

    def transcribe(self, artifact: SegmentArtifact) -> str:
        ...


class MetadataExtractor(Protocol):
    name: str

    def extract(self, transcript: str) -> TranscriptMetadata:
        ...


class VisualCaptioner(Protocol):
    name: str

    def caption(
        self, frame_paths: tuple[Path, ...], transcript: str, focus: str | None = None
    ) -> str:
        ...


class VisualEmbedder(Protocol):
    name: str

    def embed_images(self, frame_paths: tuple[Path, ...]) -> tuple[float, ...]:
        ...

    def embed_text(self, text: str) -> tuple[float, ...]:
        ...


class KeywordMetadataExtractor:
    name = "transcript-keywords"
    _proper_name = re.compile(r"\b[A-Z][\w-]+(?:\s+[A-Z][\w-]+)+\b")
    _explicit_marker = re.compile(r"\[\s*entity\s*:\s*([^\]]+)\]", re.IGNORECASE)

    def extract(self, transcript: str) -> TranscriptMetadata:
        entities = {
            match.strip() for match in self._explicit_marker.findall(transcript) if match.strip()
        }
        entities.update(match.strip() for match in self._proper_name.findall(transcript))
        return TranscriptMetadata(metadata={"entities": sorted(entities)} if entities else {})


class FasterWhisperTranscriber:
    name = "faster-whisper"

    def __init__(
        self,
        model: str = "small",
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = None,
    ):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def transcribe(self, artifact: SegmentArtifact) -> str:
        if artifact.audio_path is None:
            return ""
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        segments, _info = self._model.transcribe(str(artifact.audio_path), language=self.language)
        return " ".join(segment.text.strip() for segment in segments).strip()


class OllamaVisionCaptioner:
    def __init__(
        self,
        model: str,
        host: str = "http://127.0.0.1:11434",
        timeout_seconds: int = 900,
    ):
        self.model = model
        self.name = f"ollama-vision:{model}"
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def caption(
        self, frame_paths: tuple[Path, ...], transcript: str, focus: str | None = None
    ) -> str:
        if not frame_paths:
            return ""
        purpose = (
            f"Pay particular attention to evidence relevant to this question: {focus}\n"
            if focus
            else "Write a rough visual caption for indexing and retrieval.\n"
        )
        prompt = (
            "These are ordered frames sampled from one timestamped video segment.\n"
            f"{purpose}"
            "Output one factual sentence of at most 40 words describing visible diagrams, "
            "objects, actions, and readable on-screen text. Do not infer hidden meaning.\n"
            f"Speech transcript for context only:\n{(transcript or '[no speech]')[:300]}"
        )
        caption = self._caption_once(prompt, frame_paths)
        if _usable_caption(caption):
            return caption
        individual = (
            [
                self._caption_once(
                    f"{prompt}\nThis is frame {number} of {len(frame_paths)}.", (path,)
                )
                for number, path in enumerate(frame_paths, start=1)
            ]
            if len(frame_paths) > 1
            else []
        )
        caption = " ".join(text for text in individual if _usable_caption(text)).strip()
        if caption:
            return caption
        fallback = "Describe only what is visible in this image in one sentence."
        if focus:
            fallback += f" Focus on anything relevant to: {focus}"
        individual = [self._caption_once(fallback, (path,)) for path in frame_paths]
        return " ".join(text for text in individual if _usable_caption(text)).strip()

    def _caption_once(self, prompt: str, frame_paths: tuple[Path, ...]) -> str:
        images = [_encoded_contact_sheet(frame_paths)]
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt, "images": images}],
            "options": {"temperature": 0, "num_predict": 70},
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", "replace")
            try:
                details = str(json.loads(details).get("error") or details)
            except json.JSONDecodeError:
                pass
            raise RuntimeError(
                f"Local vision model {self.model!r} rejected the image request: {details}"
            ) from error
        except (TimeoutError, urllib.error.URLError) as error:
            raise RuntimeError(
                f"Could not call local vision model {self.model!r} at {self.host}."
            ) from error
        return str(result["message"]["content"]).strip()


class CLIPVisualEmbedder:
    name = "clip"

    def __init__(self, model: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.model_name = model
        self.device = device
        self._model = None
        self._processor = None

    def embed_images(self, frame_paths: tuple[Path, ...]) -> tuple[float, ...]:
        if not frame_paths:
            return ()
        self._load()
        from PIL import Image
        import torch

        images = [Image.open(path).convert("RGB") for path in frame_paths]
        inputs = self._processor(images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            vectors = _feature_tensor(self._model.get_image_features(**inputs))
            vector = vectors.mean(dim=0)
        return _normalized_tuple(vector)

    def embed_text(self, text: str) -> tuple[float, ...]:
        self._load()
        import torch

        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            vector = _feature_tensor(self._model.get_text_features(**inputs))[0]
        return _normalized_tuple(vector)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise RuntimeError(
                "Visual embeddings require the optional vision dependencies. "
                "Install with: python -m pip install -e .[vision]"
            ) from error
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()


def _normalized_tuple(vector: Any) -> tuple[float, ...]:
    norm = vector.norm(p=2)
    if float(norm) == 0:
        return ()
    return tuple(float(value) for value in (vector / norm).detach().cpu().tolist())


def _feature_tensor(output: Any) -> Any:
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    return output


def _usable_caption(text: str) -> bool:
    return len(re.findall(r"[^\W\d_]", text, flags=re.UNICODE)) >= 12


def _encoded_contact_sheet(frame_paths: tuple[Path, ...]) -> str:
    if len(frame_paths) == 1:
        return base64.b64encode(frame_paths[0].read_bytes()).decode("ascii")
    from PIL import Image, ImageDraw

    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    for frame in frames:
        frame.thumbnail((640, 360))
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    columns = min(len(frames), 3)
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new("RGB", (columns * width, rows * height), "black")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        x = (index % columns) * width
        y = (index // columns) * height
        sheet.paste(frame.resize((width, height)), (x, y))
        draw.rectangle((x + 6, y + 6, x + 66, y + 30), fill="black")
        draw.text((x + 12, y + 10), f"F{index + 1}", fill="white")
    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
