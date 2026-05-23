from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


AgentMode = Literal["support", "patient", "triage", "scheduler", "handoff"]


class UserInfo(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    clinic: str | None = Field(default=None, max_length=160)
    role: str | None = Field(default=None, max_length=120)
    software_version: str | None = Field(default=None, max_length=80)
    contact: str | None = Field(default=None, max_length=160)
    patient_name: str | None = Field(default=None, max_length=120)
    patient_identifier: str | None = Field(default=None, max_length=120)
    patient_age: str | None = Field(default=None, max_length=40)
    patient_city: str | None = Field(default=None, max_length=120)
    patient_address: str | None = Field(default=None, max_length=220)
    patient_phone: str | None = Field(default=None, max_length=80)
    patient_email: str | None = Field(default=None, max_length=160)
    preferred_contact_method: Literal["phone", "email", "sms", "any"] | None = "any"
    urgency: Literal["low", "normal", "high", "critical"] | None = None
    problem_summary: str | None = Field(default=None, max_length=1200)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    agent_mode: AgentMode = "support"
    strict_mode: bool = False
    top_k: int = Field(default=5, ge=1, le=12)
    retrieval_tolerance: Literal["strict", "balanced", "broad"] = "balanced"
    session_id: str | None = None
    user: UserInfo | None = None


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


class UsageEstimate(BaseModel):
    chat_model: str
    embedding_model: str
    estimated_chat_input_tokens: int
    estimated_chat_output_tokens: int
    estimated_embedding_tokens: int
    estimated_chat_cost_usd: float
    estimated_embedding_cost_usd: float
    total_estimated_cost_usd: float
    note: str


class TriageResult(BaseModel):
    urgency: Literal["low", "normal", "high", "critical"]
    label: str
    confidence: float
    reasons: list[str]
    recommendation: str
    needs_immediate_care: bool = False


class ClinicOption(BaseModel):
    name: str
    city: str
    address: str
    distance_km: float | None = None
    accepting_new_patients: bool
    services: list[str] = Field(default_factory=list)
    map_x: float = Field(default=50, ge=0, le=100)
    map_y: float = Field(default=50, ge=0, le=100)
    phone: str
    email: str
    earliest_slot: str | None = None
    note: str


class AppointmentProposal(BaseModel):
    status: Literal["pre_reserved", "needs_contact", "unavailable"]
    clinic_name: str | None = None
    slot_start: str | None = None
    reservation_id: str | None = None
    message: str
    confirmation_required: bool = True


class ChatResponse(BaseModel):
    answer: str
    agent_mode: AgentMode = "support"
    agent_label: str = "Technicka podpora"
    topic: str
    topic_label: str
    confidence: float
    answer_confidence: float | None = None
    sources: list[Source]
    steps: list[AgentStep]
    escalation_packet: str | None = None
    used_llm: bool
    session_id: str | None = None
    user: UserInfo | None = None
    retrieval_tolerance: Literal["strict", "balanced", "broad"] = "balanced"
    usage: UsageEstimate | None = None
    triage: TriageResult | None = None
    clinics: list[ClinicOption] = Field(default_factory=list)
    appointment: AppointmentProposal | None = None
    memory_updates: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class PriceInfoResponse(BaseModel):
    currency: str
    chat_model: str
    embedding_model: str
    chat_input_price_per_1m: float
    chat_output_price_per_1m: float
    embedding_price_per_1m: float
    note: str
    reference_url: str


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


class FrequentQuery(BaseModel):
    query: str
    count: int


class CacheStatsResponse(BaseModel):
    active_entries: int
    total_tracked_queries: int
    top_frequent: list[FrequentQuery]
