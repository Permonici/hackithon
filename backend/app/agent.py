from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .config import Settings
from .schemas import AgentStep, ChatResponse, Source
from .topics import classify_topic, label_for
from .utils import append_jsonl, utc_now
from .vectorstore import search_sources


SYSTEM_PROMPT = """Jsi AI operátor 1. úrovně podpory pro stomatologický software XDENT.
Odpovídej česky, velmi stručně a přímo k věci.
Použij pouze dodané zdroje z transkripcí.
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

    def answer(self, message: str, *, strict_mode: bool, top_k: int) -> ChatResponse:
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
        sources = search_sources(self.settings, message, top_k=top_k, topic_hint=topic_hint)
        best_score = max((source.score for source in sources), default=0.0)
        steps.append(
            AgentStep(
                id="retrieve",
                label="Vyhledání v transkripcích",
                status="done" if sources else "warning",
                detail=f"Nalezeno {len(sources)} relevantních částí hovorů.",
                payload={
                    "top_score": best_score,
                    "sources": [source.model_dump() for source in sources[:3]],
                },
            )
        )

        min_score = self.settings.strict_min_score if strict_mode else self.settings.lenient_min_score
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
                payload={"min_score": min_score, "best_score": best_score, "strict_mode": strict_mode},
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
                f"Zdroj: nedostatečný kontext"
            )
            used_llm = False
        else:
            escalation_packet = None
            try:
                answer, used_llm = self._draft_answer(message, classified.label, sources)
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
                status=answer_status if grounded else "done",
                detail=answer_detail if grounded else "Odpověď je bezpečně eskalovaná.",
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
        )
        self._log(message, response)
        return response

    def _draft_answer(self, message: str, topic_label: str, sources: list[Source]) -> tuple[str, bool]:
        if not self.settings.openai_api_key:
            return self._fallback_answer(topic_label, sources)

        llm = ChatOpenAI(
            model=self.settings.openai_chat_model,
            api_key=self.settings.openai_api_key,
            temperature=1,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Téma: {topic}\nDotaz zákazníka: {message}\n\nZdroje:\n{sources}\n\n"
                    "Odpověz maximálně 4 krátkými větami.",
                ),
            ]
        )
        source_text = "\n\n".join(
            f"[{index}] {source.source}\nShrnutí: {source.summary}\nMožné řešení: {source.resolution}\nÚryvek: {source.excerpt}"
            for index, source in enumerate(sources[:4], start=1)
        )
        result = (prompt | llm).invoke(
            {
                "topic": topic_label,
                "message": message,
                "sources": source_text,
            }
        )
        return str(result.content).strip(), True

    def _fallback_answer(self, topic_label: str, sources: list[Source]) -> tuple[str, bool]:
        best = sources[0]
        resolution = best.resolution or best.summary or best.excerpt
        return (
            f"Téma: {topic_label}.\n"
            f"Podle dostupných hovorů: {resolution}\n"
            f"Zdroj: {best.source}"
        ), False

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
            },
        )
