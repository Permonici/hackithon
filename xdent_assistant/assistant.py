from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .llm import OpenAICompatibleLLM
from .models import AssistantAnswer, SearchResult, TopicResult
from .retrieval import TfidfIndex, result_to_source, source_label
from .text import best_sentences
from .topics import classify_topic, infer_topic_from_results


SYSTEM_PROMPT = """Jsi AI operátor 1. úrovně podpory pro stomatologický software XDENT.
Odpovídej česky, velmi stručně a přímo k věci.
Použij pouze informace z dodaného kontextu transkripcí.
Pokud kontext nestačí, řekni to transparentně a doporuč eskalaci na podporu.
Neuváděj vymyšlené kroky ani obecné znalosti mimo kontext."""


class XDentAssistant:
    def __init__(
        self,
        index: TfidfIndex,
        *,
        log_path: str | Path = "logs/interactions.jsonl",
        llm: OpenAICompatibleLLM | None = None,
    ) -> None:
        self.index = index
        self.log_path = Path(log_path)
        self.llm = llm

    @classmethod
    def from_index_file(
        cls,
        index_path: str | Path,
        *,
        log_path: str | Path = "logs/interactions.jsonl",
        use_llm: bool = True,
    ) -> "XDentAssistant":
        llm = OpenAICompatibleLLM.from_env() if use_llm else None
        return cls(TfidfIndex.load(index_path), log_path=log_path, llm=llm)

    def answer(self, question: str, *, top_k: int = 5) -> AssistantAnswer:
        classified = classify_topic(question)
        topic_hint = classified.topic if classified.confidence >= 0.35 else None
        results = self.index.search(question, top_k=top_k, topic_hint=topic_hint)
        topic = classified if classified.topic else infer_topic_from_results(results)
        sources = [result_to_source(result) for result in results[:3]]

        used_llm = False
        if _has_sufficient_context(results) and self.llm:
            try:
                answer_text = self._llm_answer(question, topic, results)
                used_llm = True
            except RuntimeError as exc:
                answer_text = self._local_answer(question, topic, results, api_error=str(exc))
        else:
            answer_text = self._local_answer(question, topic, results)

        response = AssistantAnswer(
            question=question,
            answer=answer_text,
            topic=topic.topic,
            topic_label=topic.label,
            topic_confidence=topic.confidence,
            sources=sources,
            used_llm=used_llm,
        )
        self._log_interaction(response)
        return response

    def _llm_answer(
        self,
        question: str,
        topic: TopicResult,
        results: list[SearchResult],
    ) -> str:
        context = "\n\n".join(
            f"[{idx}] Zdroj: {source_label(result.chunk)}\n{result.chunk.text[:1200]}"
            for idx, result in enumerate(results[:4], start=1)
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Téma: {topic.label}\n"
                    f"Dotaz zákazníka: {question}\n\n"
                    f"Kontext z transkripcí:\n{context}\n\n"
                    "Vrať odpověď v chatu maximálně ve 4 krátkých větách. "
                    "Na konec uveď jeden nejrelevantnější zdroj ve tvaru `Zdroj: ...`."
                ),
            },
        ]
        assert self.llm is not None
        return self.llm.chat(messages).text

    def _local_answer(
        self,
        question: str,
        topic: TopicResult,
        results: list[SearchResult],
        *,
        api_error: str | None = None,
    ) -> str:
        if not _has_sufficient_context(results):
            return (
                f"Téma: {topic.label}.\n"
                "V dostupných transkripcích k tomu nemám dost jistý zdroj. "
                "Předejte dotaz 2. úrovni podpory a přiložte přesnou chybovou hlášku."
            )

        snippets: list[str] = []
        for result in results[:4]:
            snippets.extend(best_sentences(question, [result.chunk.text], limit=1))
            if len(snippets) >= 2:
                break

        if not snippets:
            snippets = [results[0].chunk.text[:320].strip()]

        answer = (
            f"Téma: {topic.label}.\n"
            f"Podle nalezených hovorů: {' '.join(snippets)}\n"
            f"Zdroj: {source_label(results[0].chunk)}"
        )
        if api_error:
            answer += "\nPozn.: LLM odpověď se nepodařila, použit lokální fallback."
        return answer

    def _log_interaction(self, response: AssistantAnswer) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": response.question,
            "answer": response.answer,
            "topic": response.topic,
            "topic_label": response.topic_label,
            "topic_confidence": response.topic_confidence,
            "sources": response.sources,
            "used_llm": response.used_llm,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _has_sufficient_context(results: list[SearchResult]) -> bool:
    return bool(results) and results[0].score >= 0.035
