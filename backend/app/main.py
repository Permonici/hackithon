from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .agent import SupportAgent, get_agent_cache
from .clinics import list_clinics
from .config import get_settings
from .memory import get_patient_memory_store
from .schemas import (
    CacheStatsResponse,
    ChatRequest,
    ChatResponse,
    ClinicOption,
    EvalRequest,
    EvalResponse,
    EvalResult,
    IngestResponse,
    StatsResponse,
)
from .topics import topic_catalog
from .vectorstore import IndexingAlreadyRunningError, get_stats, ingest_transcripts, qdrant_collection_exists


settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.get(f"{settings.api_prefix}/topics")
def topics() -> list[dict[str, object]]:
    return topic_catalog()


@app.get(f"{settings.api_prefix}/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    return get_stats(settings)


@app.get(f"{settings.api_prefix}/clinics", response_model=list[ClinicOption])
def clinics() -> list[ClinicOption]:
    return list_clinics()


@app.post(f"{settings.api_prefix}/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    if not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="Chybí OPENAI_API_KEY pro vytvoření embeddingů.")
    try:
        return ingest_transcripts(settings, recreate=False)
    except IndexingAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Indexace se nepodařila: {exc}",
        ) from exc


@app.post(f"{settings.api_prefix}/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    _ensure_ready()
    return SupportAgent(settings).answer(
        request.message,
        agent_mode=request.agent_mode,
        strict_mode=request.strict_mode,
        top_k=request.top_k,
        retrieval_tolerance=request.retrieval_tolerance,
        session_id=request.session_id,
        user=request.user,
    )


@app.post(f"{settings.api_prefix}/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    _ensure_ready()

    async def event_generator():
        yield _sse("step", {"id": "received", "label": "Přijat dotaz", "status": "done", "detail": request.message})
        await asyncio.sleep(0.15)
        for running_step in [
            {
                "id": "agent",
                "label": "XDent AI Asistent",
                "status": "running",
                "detail": "Asistent pripravuje odpoved podle dostupnych zdroju.",
            },
            {"id": "classify", "label": "Rozpoznání tématu", "status": "running", "detail": "Agent třídí dotaz podle podpůrných oblastí."},
            {"id": "retrieve", "label": "Vyhledání v transkripcích", "status": "running", "detail": "Qdrant hledá nejbližší případy a zdroje."},
            {"id": "validate", "label": "Kontrola jistoty", "status": "running", "detail": "Ověřuji, jestli je odpověď dostatečně podložená."},
        ]:
            yield _sse("step", running_step)
            await asyncio.sleep(0.25)

        try:
            response = await asyncio.to_thread(
                SupportAgent(settings).answer,
                request.message,
                agent_mode=request.agent_mode,
                strict_mode=request.strict_mode,
                top_k=request.top_k,
                retrieval_tolerance=request.retrieval_tolerance,
                session_id=request.session_id,
                user=request.user,
            )
        except Exception as exc:
            yield _sse(
                "step",
                {
                    "id": "error",
                    "label": "Chyba odpovědi",
                    "status": "error",
                    "detail": f"Odpověď se nepodařila sestavit: {exc}",
                },
            )
            yield _sse(
                "error",
                {
                    "message": "Odpověď se nepodařila sestavit. Zkuste dotaz znovu nebo použijte podporu.",
                    "detail": str(exc),
                },
            )
            return

        for step in response.steps:
            yield _sse("step", step.model_dump())
            await asyncio.sleep(0.2)
        yield _sse("final", response.model_dump())

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post(f"{settings.api_prefix}/evaluate", response_model=EvalResponse)
def evaluate(request: EvalRequest) -> EvalResponse:
    _ensure_ready()
    agent = SupportAgent(settings)
    results: list[EvalResult] = []
    matches = 0
    for scenario in request.scenarios:
        response = agent.answer(
            scenario.question,
            strict_mode=request.strict_mode,
            top_k=settings.retrieval_top_k,
            retrieval_tolerance="balanced",
        )
        match = None
        if scenario.expected_topic:
            match = response.topic == scenario.expected_topic
            matches += 1 if match else 0
        results.append(
            EvalResult(
                id=scenario.id,
                question=scenario.question,
                expected_topic=scenario.expected_topic,
                topic=response.topic,
                topic_label=response.topic_label,
                match=match,
                confidence=response.confidence,
                answer=response.answer,
                top_source=response.sources[0].source if response.sources else None,
            )
        )
    return EvalResponse(total=len(results), topic_matches=matches, results=results)


@app.get(f"{settings.api_prefix}/cache/stats", response_model=CacheStatsResponse)
def cache_stats() -> CacheStatsResponse:
    return CacheStatsResponse(**get_agent_cache(settings).stats())


@app.delete(f"{settings.api_prefix}/memory/{{session_id}}")
def forget_memory(session_id: str) -> dict[str, bool]:
    store = get_patient_memory_store(str(settings.logs_dir / "patient_memory.json"))
    return {"forgotten": store.forget(session_id)}


def _ensure_ready() -> None:
    if not qdrant_collection_exists(settings):
        raise HTTPException(
            status_code=409,
            detail="Qdrant kolekce ještě neexistuje. Nejdřív spusťte /api/ingest nebo `python -m app.scripts.ingest`.",
        )


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
