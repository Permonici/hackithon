from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class TopicResult:
    topic: str | None
    label: str
    confidence: float


@dataclass(frozen=True)
class AssistantAnswer:
    question: str
    answer: str
    topic: str | None
    topic_label: str
    topic_confidence: float
    sources: list[dict[str, Any]]
    used_llm: bool
