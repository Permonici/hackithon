from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    strict_mode: bool = False
    top_k: int = Field(default=5, ge=1, le=12)


class Source(BaseModel):
    source: str
    topic: str | None = None
    score: float
    excerpt: str
    summary: str | None = None
    intent: str | None = None
    resolution: str | None = None


class AgentStep(BaseModel):
    id: str
    label: str
    status: Literal["queued", "running", "done", "warning", "error"]
    detail: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    topic: str
    topic_label: str
    confidence: float
    sources: list[Source]
    steps: list[AgentStep]
    escalation_packet: str | None = None
    used_llm: bool


class IngestResponse(BaseModel):
    collection: str
    chunks_indexed: int
    records_seen: int
    records_with_text: int
    files_seen: int


class StatsResponse(BaseModel):
    collection: str
    points_count: int
    topics: list[dict[str, Any]]
    api_ready: bool
    qdrant_ready: bool


class EvalScenario(BaseModel):
    id: str | None = None
    question: str
    expected_topic: str | None = None


class EvalRequest(BaseModel):
    scenarios: list[EvalScenario]
    strict_mode: bool = False


class EvalResult(BaseModel):
    id: str | None
    question: str
    expected_topic: str | None
    topic: str
    topic_label: str
    match: bool | None
    confidence: float
    answer: str
    top_source: str | None


class EvalResponse(BaseModel):
    total: int
    topic_matches: int
    results: list[EvalResult]
