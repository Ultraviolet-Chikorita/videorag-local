from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MediaInfo:
    duration_ms: int
    has_audio: bool


@dataclass(frozen=True, slots=True)
class SegmentArtifact:
    video_path: Path
    ordinal: int
    start_ms: int
    end_ms: int
    audio_path: Path | None
    frame_paths: tuple[Path, ...] = ()


class FFmpegMediaExtractor:
    def __init__(
        self,
        artifacts_dir: str | Path,
        segment_seconds: int = 30,
        frames_per_segment: int = 0,
        overwrite: bool = False,
    ):
        if segment_seconds < 1:
            raise ValueError("segment_seconds must be positive.")
        if frames_per_segment < 0:
            raise ValueError("frames_per_segment must not be negative.")
        self.artifacts_dir = Path(artifacts_dir)
        self.segment_ms = segment_seconds * 1000
        self.frames_per_segment = frames_per_segment
        self.overwrite = overwrite
        self._check_binary("ffprobe")
        self._check_binary("ffmpeg")

    def extract(self, video_path: str | Path, artifact_key: str) -> tuple[MediaInfo, list[SegmentArtifact]]:
        source = Path(video_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Video file not found: {source}")
        media = self.probe(source)
        output_dir = self.artifacts_dir / artifact_key
        output_dir.mkdir(parents=True, exist_ok=True)

        segment_count = max(1, math.ceil(media.duration_ms / self.segment_ms))
        artifacts = []
        for ordinal in range(segment_count):
            start_ms = ordinal * self.segment_ms
            end_ms = min(media.duration_ms, start_ms + self.segment_ms)
            artifacts.append(self._extract_segment(source, output_dir, media, ordinal, start_ms, end_ms))
        return media, artifacts

    def probe(self, video_path: str | Path) -> MediaInfo:
        result = self._run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type",
                "-of",
                "json",
                str(video_path),
            ]
        )
        payload = json.loads(result.stdout)
        duration = float(payload.get("format", {}).get("duration") or 0)
        if duration <= 0:
            raise RuntimeError(f"Could not determine positive media duration for {video_path}.")
        streams = payload.get("streams") or []
        has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
        return MediaInfo(duration_ms=max(1, round(duration * 1000)), has_audio=has_audio)

    def _extract_segment(
        self,
        source: Path,
        output_dir: Path,
        media: MediaInfo,
        ordinal: int,
        start_ms: int,
        end_ms: int,
    ) -> SegmentArtifact:
        segment_dir = output_dir / f"{ordinal:06d}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        duration_seconds = (end_ms - start_ms) / 1000
        start_seconds = start_ms / 1000

        audio_path = segment_dir / "audio.wav" if media.has_audio else None
        if audio_path is not None and (self.overwrite or not audio_path.exists()):
            self._run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start_seconds:.3f}",
                    "-t",
                    f"{duration_seconds:.3f}",
                    "-i",
                    str(source),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(audio_path),
                ]
            )

        frame_paths = []
        for number in range(self.frames_per_segment):
            frame_path = segment_dir / f"frame-{number + 1:03d}.jpg"
            if self.overwrite or not frame_path.exists():
                position = start_seconds + (
                    duration_seconds * (number + 0.5) / self.frames_per_segment
                )
                self._run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-y",
                        "-loglevel",
                        "error",
                        "-ss",
                        f"{position:.3f}",
                        "-i",
                        str(source),
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        str(frame_path),
                    ]
                )
            frame_paths.append(frame_path)

        return SegmentArtifact(
            video_path=source,
            ordinal=ordinal,
            start_ms=start_ms,
            end_ms=end_ms,
            audio_path=audio_path,
            frame_paths=tuple(frame_paths),
        )

    @staticmethod
    def _check_binary(binary: str) -> None:
        if shutil.which(binary) is None:
            raise RuntimeError(f"Required media command not available on PATH: {binary}")

    @staticmethod
    def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            details = error.stderr.strip() or error.stdout.strip()
            raise RuntimeError(f"Media command failed: {' '.join(command)}\n{details}") from error
