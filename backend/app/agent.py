from __future__ import annotations

import threading
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from math import ceil

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from xdent_assistant.models import TopicResult

from .cache import QueryCache
from .clinics import assess_urgency, find_nearby_clinics, reserve_earliest_slot
from .config import Settings
from .memory import get_patient_memory_store
from .pricing import resolve_chat_prices, resolve_embedding_price
from .schemas import (
    AgentStep,
    AppointmentProposal,
    ChatResponse,
    ClinicOption,
    Source,
    TriageResult,
    UsageEstimate,
    UserInfo,
)
from .topics import classify_topic, label_for
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
Do textu odpovědi nikdy nepiš řádek `Zdroj:`; zdroje se uživateli ukazují samostatně v tlačítku.
Nevymýšlej postupy mimo kontext."""


AGENT_PROFILES = {
    "auto": {
        "label": "Chytry vyber pomoci",
        "instruction": "Vyber nejvhodnejsiho pomocnika podle dotazu a drz odpoved velmi kratkou.",
    },
    "support": {
        "label": "Problem s programem XDENT",
        "instruction": "Res technicke dotazy k XDENTu podle transkripci.",
    },
    "patient": {
        "label": "Pruvodce pacienta",
        "instruction": "Sbirej jen nutne udaje pacienta, vyhodnot urgenci a navrhni dalsi krok lidsky a strucne.",
    },
    "triage": {
        "label": "Bolest nebo akutni stav",
        "instruction": "Vyhodnot nalehavost pacienta a rekni nejbezpecnejsi dalsi krok.",
    },
    "scheduler": {
        "label": "Najit nejdrivejsi termin",
        "instruction": "Hledej nejdrivejsi vhodny termin a hlidej, zda jsou ulozene kontaktni udaje.",
    },
    "handoff": {
        "label": "Predat cloveku",
        "instruction": "Priprav kratke predani pro podporu nebo recepci, kdyz chybi jistota nebo udaje.",
    },
}


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
class CareWorkflowResult:
    triage: TriageResult | None
    clinics: list[ClinicOption]
    appointment: AppointmentProposal | None
    steps: list[AgentStep]


@dataclass
class SupportAgent:
    settings: Settings

    def answer(
        self,
        message: str,
        *,
        agent_mode: str = "support",
        strict_mode: bool,
        top_k: int,
        retrieval_tolerance: str = "balanced",
        session_id: str | None = None,
        user: UserInfo | None = None,
    ) -> ChatResponse:
        requested_agent_mode = self._normalize_agent_mode(agent_mode)
        agent_mode, route_reason = self._route_agent(message, requested_agent_mode, user)
        profile = AGENT_PROFILES[agent_mode]
        memory_store = get_patient_memory_store(str(self.settings.logs_dir / "patient_memory.json"))
        effective_user, memory_updates = memory_store.merge(
            session_id=session_id,
            incoming=user,
            message=message,
            agent_mode=agent_mode,
        )

        # ── cache lookup ──────────────────────────────────────────────────────
        cache = get_agent_cache(self.settings)
        cache_key, normalized_query = cache.build_key(message, strict_mode, retrieval_tolerance, top_k)
        cache.record_query(normalized_query)
        can_use_cache = agent_mode == "support" and not self._has_case_context(effective_user) and not memory_updates
        cached = cache.get(cache_key) if can_use_cache else None
        if cached is not None:
            return cached.model_copy(update={
                "session_id": session_id,
                "user": effective_user,
                "agent_mode": agent_mode,
                "agent_label": profile["label"],
                "requested_agent_mode": requested_agent_mode,
                "agent_route_reason": route_reason,
                "memory_updates": [],
            })

        # ── normal pipeline ───────────────────────────────────────────────────
        steps: list[AgentStep] = []
        steps.append(
            AgentStep(
                id="agent",
                label="Predani AI agentovi",
                status="done",
                detail=f"{route_reason} Aktivni agent: {profile['label']}.",
                payload={
                    "requested_agent_mode": requested_agent_mode,
                    "agent_mode": agent_mode,
                    "agent_label": profile["label"],
                    "route_reason": route_reason,
                },
            )
        )
        if memory_updates:
            steps.append(
                AgentStep(
                    id="memory",
                    label="Pamet pacienta",
                    status="done",
                    detail="Ulozeno: " + ", ".join(memory_updates) + ".",
                    payload={"updates": memory_updates},
                )
            )

        classified = classify_topic(message)
        classify_step_index = len(steps)
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
        classified = self._refine_topic(classified, retrieved_sources)
        steps[classify_step_index] = AgentStep(
            id="classify",
            label="Rozpoznání tématu",
            status="done" if classified.topic else "warning",
            detail=f"Dotaz spadá do oblasti: {classified.label}.",
            payload={
                "topic": classified.topic,
                "label": classified.label,
                "confidence": classified.confidence,
                "method": "keywords+retrieval",
            },
        )
        sources = self._select_answer_sources(retrieved_sources, classified.topic)
        best_score = max((source.score for source in sources), default=0.0)
        min_score = self._min_score(strict_mode, retrieval_tolerance)
        retrieval_fallback_used = False
        if (not sources or best_score < min_score) and retrieval_tolerance != "broad":
            broad_sources = search_sources(
                self.settings,
                message,
                top_k=max(top_k, 6),
                topic_hint=classified.topic or topic_hint,
                tolerance="broad",
            )
            broad_selected = self._select_answer_sources(broad_sources, classified.topic)
            broad_best_score = max((source.score for source in broad_selected), default=0.0)
            if broad_best_score > best_score:
                retrieved_sources = broad_sources
                sources = broad_selected
                best_score = broad_best_score
                retrieval_fallback_used = True
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
                    "fallback_used": retrieval_fallback_used,
                    "sources": [source.model_dump() for source in sources[:3]],
                },
            )
        )

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

        wants_care = agent_mode in {"patient", "triage", "scheduler", "handoff"} or self._wants_care_workflow(message, effective_user)
        care_result = self._run_care_workflow(message, effective_user, force_booking=agent_mode == "scheduler") if wants_care else None
        if care_result:
            steps.extend(care_result.steps)

        if agent_mode == "handoff":
            escalation_packet = self._build_escalation_packet(
                message=message,
                topic=classified.label,
                reason="Uzivatel zvolil predani cloveku nebo je potreba rychle predani.",
                user=effective_user,
                sources=sources,
                care=care_result,
            )
            answer = self._compose_handoff_answer(effective_user, care_result)
            used_llm = False
            answer_status = "done"
            answer_detail = "Predani cloveku je pripravene."
        elif care_result and agent_mode in {"patient", "triage", "scheduler"}:
            escalation_packet = None
            answer = self._compose_care_answer(care_result, effective_user, agent_mode)
            used_llm = False
            answer_status = "done"
            answer_detail = f"{profile['label']} sestavil kratky dalsi krok bez zbytecneho LLM volani."
        elif not grounded:
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
                "Předejte dotaz 2. úrovni podpory a přiložte přesnou chybovou hlášku nebo screenshot."
            )
            used_llm = False
            answer_status = "done"
            answer_detail = "Odpověď je bezpečně eskalovaná."
        else:
            escalation_packet = None
            try:
                answer, used_llm = self._draft_answer(message, classified.label, sources, effective_user, profile["instruction"])
                answer_status = "done"
                answer_detail = "Odpověď je krátká, opřená o zdroje a připravená pro chat."
            except Exception as exc:
                answer, used_llm = self._fallback_answer(classified.label, sources)
                answer_status = "warning"
                answer_detail = f"LLM odpověď se nepodařila, použit bezpečný fallback ze zdrojů: {exc}"

        if agent_mode != "handoff" and (not grounded or (care_result and self._needs_escalation(care_result))):
            reason = (
                "Nizka shoda v transkripcich nebo chybejici overeny postup."
                if not grounded
                else "Pacientsky pripad vyzaduje rychle predani recepci nebo 2. urovni."
            )
            escalation_packet = self._build_escalation_packet(
                message=message,
                topic=classified.label,
                reason=reason,
                user=effective_user,
                sources=sources,
                care=care_result,
            )

        if care_result and agent_mode == "support":
            answer = self._append_care_context(answer, care_result)

        answer = self._tighten_answer(self._strip_source_lines(answer), max_lines=6 if care_result else 4)

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
            answer_confidence=self._answer_confidence(classified.confidence, sources, grounded, care_result, agent_mode),
            agent_mode=agent_mode,
            agent_label=profile["label"],
            requested_agent_mode=requested_agent_mode,
            agent_route_reason=route_reason,
            sources=sources,
            steps=steps,
            escalation_packet=escalation_packet,
            used_llm=used_llm,
            session_id=session_id,
            user=effective_user,
            retrieval_tolerance=retrieval_tolerance,
            usage=self._estimate_usage(message, answer, sources, used_llm),
            triage=care_result.triage if care_result else None,
            clinics=care_result.clinics if care_result else [],
            appointment=care_result.appointment if care_result else None,
            memory_updates=memory_updates,
            next_actions=self._next_actions(agent_mode, effective_user, care_result, bool(escalation_packet), grounded),
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
        agent_instruction: str,
    ) -> tuple[str, bool]:
        if not self.settings.openai_api_key:
            return self._fallback_answer(topic_label, sources)

        llm = _get_llm(self.settings.openai_chat_model, self.settings.openai_api_key)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Aktivni agent: {agent_instruction}\n"
                    "Kontext zákazníka: {user_context}\n"
                    "Téma: {topic}\n"
                    "Dotaz zákazníka: {message}\n\n"
                    "Zdroje:\n{sources}\n\n"
                    "Odpověz maximálně 3 krátkými větami. Neopakuj celý úryvek, vytáhni jen smysluplný postup.",
                ),
            ]
        )
        result = (prompt | llm).invoke(
            {
                "topic": topic_label,
                "message": message,
                "agent_instruction": agent_instruction,
                "user_context": self._user_context(user),
                "sources": self._source_context(sources),
            }
        )
        return str(result.content).strip(), True

    def _fallback_answer(self, topic_label: str, sources: list[Source]) -> tuple[str, bool]:
        if not sources:
            return (
                f"Téma: {topic_label}.\n"
                "V dostupných transkripcích k tomu nemám dost jistý podklad. Předejte dotaz podpoře."
            ), False

        best = sources[0]
        resolution = best.resolution or best.summary or best.excerpt
        return (
            f"Téma: {topic_label}.\n"
            f"Podle dostupných hovorů: {resolution}"
        ), False

    def _run_care_workflow(self, message: str, user: UserInfo | None, *, force_booking: bool = False) -> CareWorkflowResult:
        triage = assess_urgency(message, user)
        city = self._care_location(user)
        service_query = " ".join(
            filter(
                None,
                [
                    message,
                    user.problem_summary if user else None,
                ],
            )
        )
        clinics = find_nearby_clinics(city, triage.urgency, service_query=service_query)
        appointment = (
            reserve_earliest_slot(message=message, user=user, triage=triage, clinics=clinics)
            if force_booking or self._wants_booking(message, user, triage)
            else None
        )

        steps = [
            AgentStep(
                id="triage",
                label="Kontrola nalehavosti",
                status="warning" if triage.needs_immediate_care else "done",
                detail=f"Vyhodnocena urgence: {triage.label}.",
                payload=triage.model_dump(),
            ),
            AgentStep(
                id="clinics",
                label="Vyber ordinace",
                status="done" if clinics else "warning",
                detail=(
                    f"Nalezeny {len(clinics)} demo ordinace pobliz: {city or 'bez zadaneho mesta'}."
                    if clinics
                    else "Bez zadaneho mesta nelze presne seradit ordinace."
                ),
                payload={"clinics": [clinic.model_dump() for clinic in clinics]},
            ),
        ]
        if appointment:
            steps.append(
                AgentStep(
                    id="booking",
                    label="Predbezna rezervace",
                    status="done" if appointment.status == "pre_reserved" else "warning",
                    detail=appointment.message,
                    payload=appointment.model_dump(),
                )
            )

        return CareWorkflowResult(
            triage=triage,
            clinics=clinics,
            appointment=appointment,
            steps=steps,
        )

    def _append_care_context(self, answer: str, care: CareWorkflowResult) -> str:
        lines: list[str] = []
        if care.triage:
            lines.append(f"Nalehavost: {care.triage.label} - {care.triage.recommendation}")

        if care.clinics:
            clinic_parts = []
            for clinic in care.clinics[:3]:
                accepting = "prijima nove pacienty" if clinic.accepting_new_patients else "nove pacienty jen po domluve"
                distance = f", {clinic.distance_km} km" if clinic.distance_km is not None else ""
                clinic_parts.append(f"{clinic.name} ({clinic.city}{distance}, {accepting}, nejdrive {clinic.earliest_slot})")
            lines.append("Nejdrivejsi terminy: " + "; ".join(clinic_parts))

        if care.appointment:
            if care.appointment.status == "pre_reserved":
                lines.append(
                    "Predrezervace: "
                    f"{care.appointment.clinic_name}, {care.appointment.slot_start}, kod {care.appointment.reservation_id}. "
                    "Recepce musi termin potvrdit."
                )
            else:
                lines.append(f"Termin: {care.appointment.message}")

        if lines:
            lines.append("Pozn.: ordinace a terminy jsou demo workflow; v ostrem provozu se napoji kalendar XDENT.")
        return answer.strip() + "\n\n" + "\n".join(lines)

    def _compose_care_answer(self, care: CareWorkflowResult, user: UserInfo | None, agent_mode: str) -> str:
        lines: list[str] = []
        if care.triage:
            if care.triage.urgency == "critical":
                lines.append("Tohle vypada akutne: volejte ordinaci nebo zubni pohotovost hned.")
            elif care.triage.urgency == "high":
                lines.append("Priorita je vysoka. Doporucuji nejblizsi akutni termin a potvrdit kontakt.")
            else:
                lines.append(f"Priorita: {care.triage.label}. {care.triage.recommendation}")

        if care.appointment:
            if care.appointment.status == "pre_reserved":
                lines.append(
                    f"Nasel jsem nejdrivejsi termin: {care.appointment.clinic_name}, "
                    f"{care.appointment.slot_start}. Kod predrezervace: {care.appointment.reservation_id}."
                )
            elif care.appointment.status == "needs_contact":
                lines.append("Termin umim predbezne pripravit, ale chybi telefon nebo e-mail pro potvrzeni.")
            else:
                lines.append(care.appointment.message)

        if care.clinics:
            lines.append("Moznosti v okoli:")
            for clinic in care.clinics[:2]:
                accepting = "bere nove pacienty" if clinic.accepting_new_patients else "nove pacienty jen po domluve"
                lines.append(f"- {clinic.name}: {clinic.earliest_slot}, {accepting}, tel. {clinic.phone}")

        missing = self._missing_patient_fields(user, require_location=agent_mode in {"patient", "scheduler"})
        if missing:
            lines.append("Aby to slo dokoncit, doplnte: " + ", ".join(missing) + ".")

        if care.appointment and care.appointment.status == "pre_reserved":
            lines.append("Recepce musi termin jeste potvrdit; v ostrem provozu se napoji primo na kalendar XDENT.")
        return "\n".join(lines)

    def _needs_escalation(self, care: CareWorkflowResult) -> bool:
        return bool(
            (care.triage and care.triage.urgency == "critical")
            or (care.appointment and care.appointment.status in {"needs_contact", "unavailable"})
        )

    def _compose_handoff_answer(self, user: UserInfo | None, care: CareWorkflowResult | None) -> str:
        missing = self._missing_patient_fields(user, require_location=True)
        if care and care.triage and care.triage.urgency == "critical":
            return "Pripravena eskalace. Kvuli kriticke urgenci volejte ordinaci nebo pohotovost ihned."
        if missing:
            return "Pripravena eskalace. Pro rychle predani jeste doplnte: " + ", ".join(missing) + "."
        return "Pripravena eskalace pro podporu. Predani obsahuje dotaz, kontakt, urgenci a nalezene podklady."

    def _next_actions(
        self,
        agent_mode: str,
        user: UserInfo | None,
        care: CareWorkflowResult | None,
        has_escalation: bool,
        grounded: bool,
    ) -> list[str]:
        actions: list[str] = []
        if agent_mode in {"patient", "triage", "scheduler"}:
            missing_fields = self._missing_patient_fields(user, require_location=True)
            for missing in missing_fields:
                actions.append(self._action_for_missing(missing))
            if (
                care
                and care.appointment
                and care.appointment.status == "needs_contact"
                and "telefon/e-mail" not in missing_fields
            ):
                actions.append("Doplnit kontakt pro potvrzeni")
            if care and care.triage and care.triage.urgency in {"high", "critical"}:
                actions.append("Zavolat ordinaci")
            if agent_mode != "triage":
                actions.append("Najit nejdrivejsi termin")
            if agent_mode == "triage":
                actions.append("Predat recepci")
        if has_escalation:
            actions.append("Kopirovat eskalaci")
        if not grounded and agent_mode == "support":
            actions.extend(["Prilozit screenshot", "Predat 2. urovni"])
        return list(dict.fromkeys(actions))[:4]

    def _action_for_missing(self, missing: str) -> str:
        labels = {
            "telefon/e-mail": "Doplnit telefon nebo e-mail",
            "mesto": "Doplnit mesto pacienta",
            "popis problemu": "Doplnit, co pacienta trapi",
        }
        return labels.get(missing, f"Doplnit {missing}")

    def _missing_patient_fields(self, user: UserInfo | None, *, require_location: bool) -> list[str]:
        missing: list[str] = []
        if not user or not self._best_patient_contact(user):
            missing.append("telefon/e-mail")
        if require_location and not self._care_location(user):
            missing.append("mesto")
        if not user or not user.problem_summary:
            missing.append("popis problemu")
        return missing

    def _normalize_agent_mode(self, agent_mode: str) -> str:
        return agent_mode if agent_mode in AGENT_PROFILES else "support"

    def _route_agent(
        self,
        message: str,
        requested_agent_mode: str,
        user: UserInfo | None,
    ) -> tuple[str, str]:
        if requested_agent_mode != "auto":
            return requested_agent_mode, "Rucne vybrany AI agent."

        text = self._plain(message)
        has_patient_context = bool(
            user
            and (
                user.patient_name
                or user.patient_city
                or user.patient_phone
                or user.patient_email
                or user.problem_summary
                or user.urgency
            )
        )

        critical_terms = (
            "otok",
            "horeck",
            "krvac",
            "uraz",
            "nesnesitelna bolest",
            "hnis",
            "absces",
            "nemuzu spat",
        )
        symptom_terms = (
            "bolest",
            "boli",
            "bolavy zub",
            "zub",
            "pulzuje",
            "zanet",
            "neustavajici bolest",
            "akut",
        )
        booking_terms = (
            "termin",
            "objednat",
            "rezerv",
            "predobjednat",
            "nejdrive",
            "nejblizsi",
            "hygien",
            "ordinac",
            "pobliz",
            "zubar",
            "zubni",
            "kontrola",
            "cisteni",
        )
        program_terms = ("xdent", "erecept", "epoukaz", "certifikat", "prihlas", "tisk", "sablon", "instal", "program")
        program_problem_terms = ("nejde", "nefunguje", "chyba", "pise", "spadne", "neodesila", "nelze", "problem")
        patient_terms = ("pacient", "kontakt", "telefon", "email", "e-mail", "bydlim", "jsem z")
        escalation_terms = ("eskal", "predat", "2. urov", "druha urov", "operator", "recepce", "clovek")

        if any(term in text for term in escalation_terms):
            return "handoff", "AI poznala, ze je lepsi pripravit predani cloveku."
        if any(term in text for term in critical_terms):
            return "triage", "AI poznala priznaky, ktere mohou byt akutni."
        if any(term in text for term in program_terms) and any(term in text for term in program_problem_terms):
            return "support", "AI poznala technicky problem v programu XDENT."
        if any(term in text for term in booking_terms):
            return "scheduler", "AI poznala, ze pacient pravdepodobne potrebuje termin nebo ordinaci."
        if any(term in text for term in symptom_terms):
            return "triage", "AI poznala zdravotni popis a nejdrive overi nalehavost."
        if has_patient_context or any(term in text for term in patient_terms):
            return "patient", "AI pracuje s pacientskymi udaji a doplni, co chybi."
        return "support", "AI poznala dotaz k programu XDENT."

    def _build_escalation_packet(
        self,
        *,
        message: str,
        topic: str,
        reason: str,
        user: UserInfo | None,
        sources: list[Source],
        care: CareWorkflowResult | None,
    ) -> str:
        patient = user.patient_name if user and user.patient_name else "neuvedeno"
        contact = self._best_patient_contact(user)
        location = self._care_location(user) or "neuvedeno"
        urgency = care.triage.label if care and care.triage else self._urgency_label(user.urgency if user else None) or "neuvedeno"
        next_slot = "neuvedeno"
        reservation = "bez rezervace"
        if care and care.appointment:
            next_slot = care.appointment.slot_start or "neuvedeno"
            reservation = care.appointment.reservation_id or care.appointment.status
        clinic_lines = []
        if care:
            for clinic in care.clinics[:3]:
                accepting = "ano" if clinic.accepting_new_patients else "po domluve"
                clinic_lines.append(
                    f"  - {clinic.name}, {clinic.city}, prijima nove: {accepting}, nejdrive: {clinic.earliest_slot}, tel.: {clinic.phone}"
                )
        source_lines = [f"  - {source.source} (score {source.score})" for source in sources[:3]]
        return "\n".join(
            [
                "ESKALACE - XDENT AI",
                f"Duvod: {reason}",
                f"Tema: {topic}",
                f"Dotaz: {message}",
                "",
                "Pacient / kontakt",
                f"- Pacient: {patient}",
                f"- Kontakt: {contact or 'chybi - doplnit telefon nebo e-mail'}",
                f"- Lokalita: {location}",
                f"- Urgence: {urgency}",
                "",
                "Navrzeny dalsi krok",
                f"- Overit termin: {next_slot}",
                f"- Rezervace: {reservation}",
                "- Pokud jde o otok, horecku, uraz nebo silne krvaceni, volat ordinaci/pohotovost ihned.",
                "",
                "Doporucene ordinace",
                *(clinic_lines or ["  - nejsou k dispozici"]),
                "",
                "Pouzite zdroje",
                *(source_lines or ["  - nedostatecny kontext"]),
                "",
                "Co doplnit pri predani",
                "- screenshot/chybovou hlasku, cas vyskytu, kroky k reprodukci",
                "- potvrzeny telefon/e-mail pacienta a souhlas s kontaktovanim",
            ]
        )

    def _wants_care_workflow(self, message: str, user: UserInfo | None) -> bool:
        text = self._plain(message)
        keywords = (
            "pacient",
            "bolest",
            "akut",
            "ordinac",
            "pobliz",
            "poblíž",
            "termin",
            "termín",
            "objednat",
            "rezerv",
            "predobjednat",
            "hygien",
            "předobjednat",
        )
        has_patient_context = bool(
            user
            and (
                user.patient_name
                or user.problem_summary
                or user.patient_city
                or user.patient_phone
                or user.patient_email
                or (user.urgency and user.urgency != "normal")
            )
        )
        explicit_booking = (
            "najdi nejdrivejsi" in text
            or "nejdrivejsi termin" in text
            or "nejblizsi termin" in text
            or "akutni termin" in text
            or "ordinac pobliz" in text
            or "ordinace pobliz" in text
            or "prijima nove pacienty" in text
            or "dentalni hygien" in text
            or "predobjednat" in text
            or "rezervovat pacienta" in text
        )
        keyword_hit = any(keyword in text for keyword in keywords)
        return has_patient_context or explicit_booking or keyword_hit

    def _wants_booking(self, message: str, user: UserInfo | None, triage: TriageResult) -> bool:
        text = self._plain(message)
        booking_keywords = (
            "termin",
            "termín",
            "objednat",
            "rezerv",
            "predobjednat",
            "předobjednat",
            "hygien",
            "nejdrive",
            "nejdříve",
        )
        return (
            any(keyword in text for keyword in booking_keywords)
            or triage.urgency in {"high", "critical"}
            or bool(user and user.problem_summary and (user.patient_phone or user.patient_email or user.contact))
        )

    def _care_location(self, user: UserInfo | None) -> str | None:
        if not user:
            return None
        return user.patient_city or user.clinic or user.patient_address

    def _best_patient_contact(self, user: UserInfo | None) -> str | None:
        if not user:
            return None
        for value in (user.patient_phone, user.patient_email, user.contact):
            if value and value.strip():
                return value.strip()
        return None

    def _refine_topic(self, classified: TopicResult, sources: list[Source]) -> TopicResult:
        topic_weights: dict[str, float] = {}
        for source in sources[:8]:
            if not source.topic:
                continue
            weight = max(0.01, min(float(source.score), 1.0))
            topic_weights[source.topic] = topic_weights.get(source.topic, 0.0) + weight

        if not topic_weights:
            return classified

        best_topic, best_weight = max(topic_weights.items(), key=lambda item: item[1])
        total_weight = sum(topic_weights.values())
        source_confidence = round(best_weight / total_weight, 3) if total_weight else 0.0

        should_use_retrieval_topic = (
            classified.topic is None
            or classified.confidence < 0.55
            or (best_topic != classified.topic and source_confidence >= 0.62)
        )
        if not should_use_retrieval_topic:
            return classified

        return TopicResult(
            topic=best_topic,
            label=label_for(best_topic),
            confidence=max(classified.confidence, source_confidence),
        )

    def _strip_source_lines(self, answer: str) -> str:
        visible_lines = [
            line
            for line in answer.splitlines()
            if not line.strip().lower().startswith("zdroj:")
        ]
        return "\n".join(visible_lines).strip()

    def _tighten_answer(self, answer: str, *, max_lines: int) -> str:
        lines: list[str] = []
        seen: set[str] = set()
        for line in answer.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            fingerprint = self._plain(cleaned)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            lines.append(cleaned)
        if len(lines) <= max_lines:
            return "\n".join(lines)
        return "\n".join(lines[:max_lines])

    def _plain(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value.lower())
        return "".join(char for char in normalized if not unicodedata.combining(char))

    def _answer_confidence(
        self,
        topic_confidence: float,
        sources: list[Source],
        grounded: bool,
        care: CareWorkflowResult | None = None,
        agent_mode: str = "support",
    ) -> float:
        if care and agent_mode in {"patient", "triage", "scheduler", "handoff"}:
            triage_confidence = care.triage.confidence if care.triage else 0.55
            best_source = max((max(0.0, min(float(source.score), 1.0)) for source in sources), default=0.0)
            base = max(0.35, triage_confidence * 0.75)
            if grounded:
                base = max(base, (triage_confidence * 0.45) + (best_source * 0.35))
            if care.clinics:
                base = max(base, 0.72)
            if care.appointment and care.appointment.status == "pre_reserved":
                base = max(base, 0.82)
            if care.triage and care.triage.urgency == "critical":
                base = max(base, 0.88)
            return round(max(0.0, min(base, 0.96)), 3)

        if not grounded:
            return 0.0
        normalized_topic = max(0.0, min(float(topic_confidence), 1.0))
        best_source = max((max(0.0, min(float(source.score), 1.0)) for source in sources), default=0.0)
        if not sources:
            return round(normalized_topic * 0.6, 3)
        return round((normalized_topic * 0.45) + (best_source * 0.55), 3)

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
            ("mesto pacienta", user.patient_city),
            ("adresa pacienta", user.patient_address),
            ("telefon pacienta", user.patient_phone),
            ("e-mail pacienta", user.patient_email),
            ("preferovany kontakt", user.preferred_contact_method),
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
                user.patient_city,
                user.patient_address,
                user.patient_phone,
                user.patient_email,
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
                "requested_agent_mode": response.requested_agent_mode,
                "agent_mode": response.agent_mode,
                "agent_label": response.agent_label,
                "agent_route_reason": response.agent_route_reason,
                "topic": response.topic,
                "topic_label": response.topic_label,
                "confidence": response.confidence,
                "answer_confidence": response.answer_confidence,
                "sources": [source.model_dump() for source in response.sources],
                "used_llm": response.used_llm,
                "session_id": response.session_id,
                "review_ready": True,
                "memory_updates": response.memory_updates,
                "next_actions": response.next_actions,
                "retrieval_tolerance": response.retrieval_tolerance,
                "user": response.user.model_dump() if response.user else None,
                "usage": response.usage.model_dump() if response.usage else None,
                "triage": response.triage.model_dump() if response.triage else None,
                "clinics": [clinic.model_dump() for clinic in response.clinics],
                "appointment": response.appointment.model_dump() if response.appointment else None,
            },
        )
