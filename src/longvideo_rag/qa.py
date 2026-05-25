from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from .models import QueryFilters, RetrievalHit, RetrievalResult
from .providers import VisualCaptioner
from .retrieval import Retriever


@dataclass(frozen=True, slots=True)
class AnswerResult:
    question: str
    answer: str
    evidence: RetrievalResult


class AnswerGenerator(Protocol):
    def generate(self, question: str, evidence: RetrievalResult) -> str:
        ...


class EvidenceRefiner(Protocol):
    def refine(self, question: str, evidence: RetrievalResult) -> RetrievalResult:
        ...


class GroundedQA:
    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        refiner: EvidenceRefiner | None = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.refiner = refiner

    def ask(
        self, question: str, top_k: int = 8, filters: QueryFilters | None = None
    ) -> AnswerResult:
        evidence = self.retriever.search(question, top_k=top_k, filters=filters)
        if not evidence.hits:
            return AnswerResult(
                question=question,
                answer="No relevant transcript or visual evidence was retrieved for this question.",
                evidence=evidence,
            )
        if self.refiner is not None:
            evidence = self.refiner.refine(question, evidence)
        return AnswerResult(
            question=question,
            answer=self.generator.generate(question, evidence),
            evidence=evidence,
        )


class FocusedVisualRefiner:
    def __init__(self, captioner: VisualCaptioner):
        self.captioner = captioner

    def refine(self, question: str, evidence: RetrievalResult) -> RetrievalResult:
        refined_hits = []
        for hit in evidence.hits:
            frame_paths = tuple(
                path for path in (Path(value) for value in hit.context.segment.frame_paths)
                if path.is_file()
            )
            if not frame_paths:
                refined_hits.append(hit)
                continue
            caption = self.captioner.caption(
                frame_paths, hit.context.segment.transcript, focus=question
            ).strip()
            if not caption:
                refined_hits.append(hit)
                continue
            segment = replace(hit.context.segment, visual_caption=caption)
            context = replace(hit.context, segment=segment)
            refined_hits.append(replace(hit, context=context))
        return RetrievalResult(query=evidence.query, hits=tuple(refined_hits))


class OllamaGenerator:
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.1,
        max_context_chars: int = 18000,
        timeout_seconds: int = 900,
        max_answer_tokens: int = 256,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_context_chars = max_context_chars
        self.timeout_seconds = timeout_seconds
        self.max_answer_tokens = max_answer_tokens

    def generate(self, question: str, evidence: RetrievalResult) -> str:
        evidence_text = format_evidence(evidence, self.max_context_chars)
        user_prompt = (
            f"Question: {question}\n\n"
            "EVIDENCE:\n"
            f"{evidence_text}\n\n"
            "Write 2 to 4 short bullet points that answer the question. Each bullet must state "
            "a supported claim and end with source labels such as [S1] or [S1, S3]. Use only "
            "facts stated in EVIDENCE. Do not add outside examples or definitions. Do not "
            "return source labels without an explanation."
        )
        answer = self._chat(user_prompt)
        if not _is_usable_cited_answer(answer):
            answer = self._chat(
                f"{user_prompt}\n\n"
                "Produce the answer now as short explanatory bullets. Every bullet must contain "
                "both words explaining the answer and one or more EVIDENCE labels."
            )
        if not _is_usable_cited_answer(answer):
            return _evidence_fallback(evidence_text)
        checked_answer = self._chat(
            f"Question: {question}\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            f"DRAFT ANSWER:\n{answer}\n\n"
            "Audit the draft against EVIDENCE. Return 2 to 4 short bullet points containing "
            "only claims explicitly supported by EVIDENCE, each with exact source labels. "
            "Remove or correct any unsupported inference; do not add new facts. Preserve "
            "dimension words such as line, plane, and space exactly as the evidence states them."
        )
        if not _is_usable_cited_answer(checked_answer):
            return _evidence_fallback(evidence_text)
        return _align_citations(checked_answer, evidence)

    def _chat(self, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied timestamped transcript and visual evidence. Use concise bullet "
                        "claims with exact source labels. Never invent material."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_answer_tokens,
            },
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
        except (TimeoutError, urllib.error.URLError) as error:
            raise RuntimeError(
                f"Could not call local Ollama at {self.host}. Ensure Ollama is running and "
                f"the model {self.model!r} is installed and responsive."
            ) from error
        return str(result["message"]["content"]).strip()


def _has_source_citation(answer: str) -> bool:
    return re.search(r"\[S\d+(?:\s*,\s*S\d+)*\]", answer) is not None


def _is_usable_cited_answer(answer: str) -> bool:
    text_without_labels = re.sub(r"\[S\d+(?:\s*,\s*S\d+)*\]", "", answer)
    explanatory_letters = re.sub(r"[^A-Za-z]", "", text_without_labels)
    return _has_source_citation(answer) and len(explanatory_letters) >= 30


def _evidence_fallback(evidence_text: str) -> str:
    return (
        "The local model did not produce a citable synthesis. The most relevant retrieved "
        "timestamped evidence is provided below:\n\n"
        f"{evidence_text}"
    )


_CITATION_RE = re.compile(r"\[S\d+(?:\s*,\s*S\d+)*\]")
_ALIGNMENT_STOP_WORDS = {
    "and",
    "are",
    "because",
    "does",
    "for",
    "from",
    "have",
    "how",
    "into",
    "since",
    "such",
    "that",
    "the",
    "their",
    "this",
    "with",
}


def _align_citations(answer: str, evidence: RetrievalResult) -> str:
    document_terms = [_answer_terms(hit.context.segment.retrieval_text) for hit in evidence.hits]
    frequencies = Counter(term for terms in document_terms for term in terms)
    aligned_lines = []
    for line in answer.splitlines():
        claim_terms = _answer_terms(_CITATION_RE.sub("", line))
        if not claim_terms or not _has_source_citation(line):
            aligned_lines.append(line)
            continue
        scores = []
        for number, terms in enumerate(document_terms, start=1):
            score = sum(1.0 / frequencies[term] for term in claim_terms & terms)
            scores.append((score, number))
        scores.sort(reverse=True)
        highest = scores[0][0] if scores else 0.0
        if highest <= 0:
            aligned_lines.append(line)
            continue
        source_numbers = [
            number for score, number in scores[:2] if score >= highest * 0.65 and score > 0
        ]
        replacement = "[" + ", ".join(f"S{number}" for number in source_numbers) + "]"
        aligned_lines.append(_CITATION_RE.sub(replacement, line))
    return "\n".join(aligned_lines)


def _answer_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z][a-z-]+", text.casefold())
        if len(term) >= 3 and term not in _ALIGNMENT_STOP_WORDS
    }


def format_evidence(evidence: RetrievalResult, max_chars: int = 18000) -> str:
    blocks = []
    current_length = 0
    for number, hit in enumerate(evidence.hits, start=1):
        block = (
            f"[S{number}] {hit.context.citation}\n"
            f"{hit.context.segment.retrieval_text}\n"
        )
        if blocks and current_length + len(block) > max_chars:
            break
        blocks.append(block)
        current_length += len(block)
    return "\n".join(blocks)
