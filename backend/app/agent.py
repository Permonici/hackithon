from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from math import ceil

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .cache import QueryCache
from .config import Settings
from .pricing import resolve_chat_prices, resolve_embedding_price
from .schemas import AgentStep, ChatResponse, Source, UsageEstimate, UserInfo
from .topics import classify_topic
from .utils import append_jsonl, utc_now
from .vectorstore import search_sources


@lru_cache(maxsize=4)
def _get_llm(model: str, api_key: str) -> ChatOpenAI:
    return ChatOpenAI(model=model, api_key=api_key, temperature=1)


# Module-level cache singleton – initialised lazily on first use.
_agent_cache: QueryCache | None = None
_cache_init_lock = threading.Lock()


def get_agent_cache(settings: Settings) -> QueryCache:
    global _agent_cache
    if _agent_cache is None:
        with _cache_init_lock:
            if _agent_cache is None:
                _agent_cache = QueryCache(
                    persist_path=settings.logs_dir / "frequent_queries.json"
                )
    return _agent_cache


SYSTEM_PROMPT = """Jsi AI operátor 1. úrovně podpory pro stomatologický software XDENT.
Odpovídej česky, velmi stručně a přímo k věci.
Použij pouze dodané zdroje z transkripcí.
Zdroje jsou seřazené podle relevance; odpověď stav hlavně na prvních zdrojích.
Nesmíš míchat nesouvisející úryvky. Pokud zdroj popisuje jiný problém, ignoruj ho.
Dej konkrétní postup jen tehdy, když je ve zdrojích opravdu uvedený.
Urgenci pacienta použij jen pro doporučení priority dalšího postupu, ne pro vymýšlení neověřených řešení.
Pokud zdroje nestačí, řekni to a doporuč eskalaci na podporu.
Na konec uveď krátký řádek `Zdroj: ...` a použij přesný název zdroje z kontextu, ne pouze číslo v hranaté závorce.
Nevymýšlej postupy mimo kontext."""


@tool
def classify_issue_tool(message: str) -> str:
    """Rozpozná tematickou oblast dotazu zákazníka."""
    result = classify_topic(message)
    return f"{result.topic or 'unknown'}|{result.label}|{result.confidence}"


@tool
def create_escalation_packet_tool(message: str, topic: str, reason: str) -> str:
    """Vytvoří krátký balíček pro předání dotazu 2. úrovni podpory."""
    return (
        "Eskalace pro 2. úroveň podpory\n"
        f"- Téma: {topic}\n"
        f"- Dotaz: {message}\n"
        f"- Důvod: {reason}\n"
        "- Doplnit: screenshot/chybovou hlášku, účet uživatele, čas výskytu a kroky k reprodukci."
    )


@dataclass
class SupportAgent:
    settings: Settings

    def answer(
        self,
        message: str,
        *,
        strict_mode: bool,
        top_k: int,
        retrieval_tolerance: str = "balanced",
        session_id: str | None = None,
        user: UserInfo | None = None,
    ) -> ChatResponse:
        # ── cache lookup ──────────────────────────────────────────────────────
        cache = get_agent_cache(self.settings)
        cache_key, normalized_query = cache.build_key(message, strict_mode, retrieval_tolerance, top_k)
        can_use_cache = not self._has_case_context(user)
        cached = cache.get(cache_key) if can_use_cache else None
        if cached is not None:
            return cached.model_copy(update={"session_id": session_id, "user": user})

        # ── normal pipeline ───────────────────────────────────────────────────
        steps: list[AgentStep] = []

        classified = classify_topic(message)
        steps.append(
            AgentStep(
                id="classify",
                label="Rozpoznání tématu",
                status="done",
                detail=f"Dotaz spadá do oblasti: {classified.label}.",
                payload={
                    "topic": classified.topic,
                    "label": classified.label,
                    "confidence": classified.confidence,
                },
            )
        )

        topic_hint = classified.topic if classified.confidence >= 0.35 else None
        retrieved_sources = search_sources(
            self.settings,
            message,
            top_k=top_k,
            topic_hint=topic_hint,
            tolerance=retrieval_tolerance,
        )
        sources = self._select_answer_sources(retrieved_sources, classified.topic)
        best_score = max((source.score for source in sources), default=0.0)
        steps.append(
            AgentStep(
                id="retrieve",
                label="Vyhledání v transkripcích",
                status="done" if sources else "warning",
                detail=(
                    f"Nalezeno {len(retrieved_sources)} relevantních částí hovorů, "
                    f"pro odpověď použito {len(sources)} zdrojů "
                    f"({self._tolerance_label(retrieval_tolerance)} hledání)."
                ),
                payload={
                    "top_score": best_score,
                    "tolerance": retrieval_tolerance,
                    "sources": [source.model_dump() for source in sources[:3]],
                },
            )
        )

        min_score = self._min_score(strict_mode, retrieval_tolerance)
        grounded = bool(sources) and best_score >= min_score
        steps.append(
            AgentStep(
                id="validate",
                label="Kontrola jistoty",
                status="done" if grounded else "warning",
                detail=(
                    "Kontext je dostatečný pro odpověď."
                    if grounded
                    else "Kontext není dostatečně jistý, aktivuji bezpečný fallback."
                ),
                payload={
                    "min_score": min_score,
                    "best_score": best_score,
                    "strict_mode": strict_mode,
                    "tolerance": retrieval_tolerance,
                },
            )
        )

        if not grounded:
            escalation_packet = create_escalation_packet_tool.invoke(
                {
                    "message": message,
                    "topic": classified.label,
                    "reason": "Nízká shoda v transkripcích nebo chybějící ověřený postup.",
                }
            )
            answer = (
                f"Téma: {classified.label}.\n"
                "V dostupných transkripcích k tomu nemám dost jistý podklad. "
                "Předejte dotaz 2. úrovni podpory a přiložte přesnou chybovou hlášku nebo screenshot.\n"
                "Zdroj: nedostatečný kontext"
            )
            used_llm = False
            answer_status = "done"
            answer_detail = "Odpověď je bezpečně eskalovaná."
        else:
            escalation_packet = None
            try:
                answer, used_llm = self._draft_answer(message, classified.label, sources, user)
                answer_status = "done"
                answer_detail = "Odpověď je krátká, opřená o zdroje a připravená pro chat."
            except Exception as exc:
                answer, used_llm = self._fallback_answer(classified.label, sources)
                answer_status = "warning"
                answer_detail = f"LLM odpověď se nepodařila, použit bezpečný fallback ze zdrojů: {exc}"

        steps.append(
            AgentStep(
                id="answer",
                label="Sestavení odpovědi",
                status=answer_status,
                detail=answer_detail,
                payload={"used_llm": used_llm},
            )
        )

        response = ChatResponse(
            answer=answer,
            topic=classified.topic or "unknown",
            topic_label=classified.label,
            confidence=classified.confidence,
            sources=sources,
            steps=steps,
            escalation_packet=escalation_packet,
            used_llm=used_llm,
            session_id=session_id,
            user=user,
            retrieval_tolerance=retrieval_tolerance,
            usage=self._estimate_usage(message, answer, sources, used_llm),
        )
        self._log(message, response)

        # ── cache the result (keyed without session/user) ─────────────────────
        if can_use_cache:
            cache.put(cache_key, normalized_query, response)

        return response

    def _draft_answer(
        self,
        message: str,
        topic_label: str,
        sources: list[Source],
        user: UserInfo | None,
    ) -> tuple[str, bool]:
        if not self.settings.openai_api_key:
            return self._fallback_answer(topic_label, sources)

        llm = _get_llm(self.settings.openai_chat_model, self.settings.openai_api_key)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Kontext zákazníka: {user_context}\n"
                    "Téma: {topic}\n"
                    "Dotaz zákazníka: {message}\n\n"
                    "Zdroje:\n{sources}\n\n"
                    "Odpověz maximálně 4 krátkými větami. Neopakuj celý úryvek, vytáhni jen smysluplný postup.",
                ),
            ]
        )
        result = (prompt | llm).invoke(
            {
                "topic": topic_label,
                "message": message,
                "user_context": self._user_context(user),
                "sources": self._source_context(sources),
            }
        )
        return str(result.content).strip(), True

    def _fallback_answer(self, topic_label: str, sources: list[Source]) -> tuple[str, bool]:
        if not sources:
            return (
                f"Téma: {topic_label}.\n"
                "V dostupných transkripcích k tomu nemám dost jistý podklad. Předejte dotaz podpoře.\n"
                "Zdroj: nedostatečný kontext"
            ), False

        best = sources[0]
        resolution = best.resolution or best.summary or best.excerpt
        return (
            f"Téma: {topic_label}.\n"
            f"Podle dostupných hovorů: {resolution}\n"
            f"Zdroj: {best.source}"
        ), False

    def _select_answer_sources(self, sources: list[Source], topic: str | None) -> list[Source]:
        if not sources:
            return []

        if topic:
            topic_sources = [source for source in sources if source.topic == topic]
            if topic_sources:
                return topic_sources[:4]

        return sources[:4]

    def _source_context(self, sources: list[Source]) -> str:
        return "\n\n".join(
            f"[{index}] {source.source}\n"
            f"Relevance: {source.score}\n"
            f"Shrnutí: {source.summary}\n"
            f"Možné řešení: {source.resolution}\n"
            f"Úryvek: {source.excerpt}"
            for index, source in enumerate(sources[:4], start=1)
        )

    def _user_context(self, user: UserInfo | None) -> str:
        if not user:
            return "neuvedeno"
        parts = [
            ("operátor", user.name),
            ("ordinace", user.clinic),
            ("role", user.role),
            ("verze XDENT", user.software_version),
            ("kontakt", user.contact),
            ("pacient", user.patient_name),
            ("ID pacienta/karta", user.patient_identifier),
            ("věk pacienta", user.patient_age),
            ("urgence", self._urgency_label(user.urgency)),
            ("konkrétní problém", user.problem_summary),
        ]
        filled = [f"{label}: {value}" for label, value in parts if value]
        return "; ".join(filled) if filled else "neuvedeno"

    def _has_case_context(self, user: UserInfo | None) -> bool:
        if not user:
            return False
        return any(
            bool(value)
            for value in (
                user.patient_name,
                user.patient_identifier,
                user.patient_age,
                user.urgency,
                user.problem_summary,
            )
        )

    def _urgency_label(self, urgency: str | None) -> str | None:
        labels = {
            "low": "nízká",
            "normal": "běžná",
            "high": "vysoká",
            "critical": "kritická",
        }
        return labels.get(urgency or "")

    def _min_score(self, strict_mode: bool, retrieval_tolerance: str) -> float:
        if strict_mode or retrieval_tolerance == "strict":
            return self.settings.strict_min_score
        if retrieval_tolerance == "broad":
            return self.settings.broad_min_score
        return self.settings.lenient_min_score

    def _tolerance_label(self, retrieval_tolerance: str) -> str:
        labels = {
            "strict": "přesné",
            "balanced": "vyvážené",
            "broad": "širší",
        }
        return labels.get(retrieval_tolerance, "vyvážené")

    def _estimate_usage(
        self,
        message: str,
        answer: str,
        sources: list[Source],
        used_llm: bool,
    ) -> UsageEstimate:
        source_context = self._source_context(sources)
        chat_input_tokens = self._estimate_tokens(SYSTEM_PROMPT + message + source_context) if used_llm else 0
        chat_output_tokens = self._estimate_tokens(answer) if used_llm else 0
        embedding_tokens = self._estimate_tokens(message)
        chat_input_price, chat_output_price, _ = resolve_chat_prices(self.settings)
        embedding_price, _ = resolve_embedding_price(self.settings)
        chat_cost = (
            (chat_input_tokens * chat_input_price)
            + (chat_output_tokens * chat_output_price)
        ) / 1_000_000
        embedding_cost = (embedding_tokens * embedding_price) / 1_000_000
        total = chat_cost + embedding_cost
        return UsageEstimate(
            chat_model=self.settings.openai_chat_model,
            embedding_model=self.settings.openai_embedding_model,
            estimated_chat_input_tokens=chat_input_tokens,
            estimated_chat_output_tokens=chat_output_tokens,
            estimated_embedding_tokens=embedding_tokens,
            estimated_chat_cost_usd=round(chat_cost, 8),
            estimated_embedding_cost_usd=round(embedding_cost, 8),
            total_estimated_cost_usd=round(total, 8),
            note="Orientační odhad: tokeny jsou počítány z délky textu a ceny se berou z .env.",
        )

    def _estimate_tokens(self, text: str) -> int:
        return max(1, ceil(len(text) / 4))

    def _log(self, message: str, response: ChatResponse) -> None:
        append_jsonl(
            self.settings.interactions_log_path,
            {
                "timestamp": utc_now(),
                "question": message,
                "answer": response.answer,
                "topic": response.topic,
                "topic_label": response.topic_label,
                "confidence": response.confidence,
                "sources": [source.model_dump() for source in response.sources],
                "used_llm": response.used_llm,
                "session_id": response.session_id,
                "retrieval_tolerance": response.retrieval_tolerance,
                "user": response.user.model_dump() if response.user else None,
                "usage": response.usage.model_dump() if response.usage else None,
            },
        )
