from __future__ import annotations

import uuid
import threading
import time
from collections import Counter
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from xdent_assistant.ingest import build_chunks
from xdent_assistant.retrieval import source_label
from xdent_assistant.text import best_sentences
from xdent_assistant.topics import TOPICS, classify_topic, label_for

from .config import Settings
from .schemas import IngestResponse, Source, StatsResponse
from .utils import compact_text


STATS_CACHE_TTL_SECONDS = 60.0
_stats_cache_lock = threading.Lock()
_stats_cache: dict[tuple[str, str, bool], tuple[float, StatsResponse]] = {}


@lru_cache(maxsize=4)
def make_embeddings(model: str, api_key: str, dimensions: int | None) -> OpenAIEmbeddings:
    kwargs: dict[str, Any] = {"model": model, "api_key": api_key}
    if dimensions:
        kwargs["dimensions"] = dimensions
    return OpenAIEmbeddings(**kwargs)


@lru_cache(maxsize=4)
def get_qdrant_client(url: str) -> QdrantClient:
    return QdrantClient(url=url)


@lru_cache(maxsize=2)
def make_sparse_embedding(model_name: str) -> FastEmbedSparse:
    return FastEmbedSparse(model_name=model_name)


def _embeddings(settings: Settings) -> OpenAIEmbeddings:
    return make_embeddings(
        settings.openai_embedding_model,
        settings.openai_api_key,
        settings.openai_embedding_dimensions or None,
    )


@lru_cache(maxsize=4)
def make_vector_store(
    qdrant_url: str,
    collection_name: str,
    embedding_model: str,
    api_key: str,
    dimensions: int | None,
) -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        embedding=make_embeddings(embedding_model, api_key, dimensions),
        sparse_embedding=make_sparse_embedding("Qdrant/bm25"),
        retrieval_mode=RetrievalMode.HYBRID,
        collection_name=collection_name,
        url=qdrant_url,
    )


def build_vector_store(settings: Settings) -> QdrantVectorStore:
    return make_vector_store(
        settings.qdrant_url,
        settings.qdrant_collection,
        settings.openai_embedding_model,
        settings.openai_api_key,
        settings.openai_embedding_dimensions or None,
    )


def clear_stats_cache() -> None:
    with _stats_cache_lock:
        _stats_cache.clear()


def ingest_transcripts(settings: Settings, *, recreate: bool = True) -> IngestResponse:
    ingest = build_chunks(
        settings.data_dir,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        include_control_group=False,
    )

    documents: list[Document] = []
    ids: list[str] = []
    for chunk in ingest.chunks:
        topic = chunk.metadata["topic"]
        classified = classify_topic(chunk.text)
        intent = _extract_intent(chunk.text)
        resolution = _extract_resolution(chunk.text)
        summary = _short_summary(chunk.text)
        document_id = _stable_id(chunk.id)
        ids.append(document_id)
        documents.append(
            Document(
                page_content=chunk.text,
                metadata={
                    **chunk.metadata,
                    "chunk_id": chunk.id,
                    "source": source_label(chunk),
                    "topic_label": label_for(topic),
                    "auto_topic": classified.topic,
                    "auto_topic_label": classified.label,
                    "auto_topic_confidence": classified.confidence,
                    "intent": intent,
                    "resolution": resolution,
                    "summary": summary,
                    "quality": _quality_score(chunk.text, resolution),
                },
            )
        )

    QdrantVectorStore.from_documents(
        documents,
        embedding=_embeddings(settings),
        sparse_embedding=make_sparse_embedding("Qdrant/bm25"),
        retrieval_mode=RetrievalMode.HYBRID,
        ids=ids,
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
        force_recreate=recreate,
    )
    get_qdrant_client.cache_clear()
    make_vector_store.cache_clear()
    clear_stats_cache()

    return IngestResponse(
        collection=settings.qdrant_collection,
        chunks_indexed=len(documents),
        records_seen=ingest.records_seen,
        records_with_text=ingest.records_with_text,
        files_seen=ingest.files_seen,
    )


def search_sources(
    settings: Settings,
    query: str,
    *,
    top_k: int,
    topic_hint: str | None,
    tolerance: str = "balanced",
) -> list[Source]:
    try:
        store = build_vector_store(settings)
    except Exception:
        return []

    effective_k = _effective_k(top_k, tolerance)
    topic_filter = _topic_filter(topic_hint) if topic_hint else None
    docs_with_scores: list[tuple[Document, float]] = []

    try:
        for search_query in _query_variants(query, topic_hint, tolerance):
            docs_with_scores.extend(
                store.similarity_search_with_score(
                    search_query,
                    k=effective_k,
                    filter=topic_filter,
                )
            )

        if topic_hint and tolerance in {"balanced", "broad"}:
            docs_with_scores.extend(store.similarity_search_with_score(query, k=effective_k))

        if not docs_with_scores and topic_hint:
            docs_with_scores = store.similarity_search_with_score(query, k=effective_k)
    except Exception:
        return []

    docs_with_scores = _dedupe_and_rank(docs_with_scores, limit=top_k, topic_hint=topic_hint)

    sources: list[Source] = []
    for doc, score in docs_with_scores:
        metadata = doc.metadata
        sources.append(
            Source(
                source=str(metadata.get("source") or metadata.get("chunk_id") or "neznámý zdroj"),
                topic=metadata.get("topic"),
                score=round(float(score), 4),
                excerpt=compact_text(doc.page_content, limit=520),
                summary=metadata.get("summary"),
                intent=metadata.get("intent"),
                resolution=metadata.get("resolution"),
            )
        )
    return sources


def _effective_k(top_k: int, tolerance: str) -> int:
    if tolerance == "broad":
        return min(18, max(top_k * 3, 10))
    if tolerance == "balanced":
        return min(14, max(top_k * 2, 8))
    return top_k


def _topic_filter(topic_hint: str | None) -> qdrant_models.Filter | None:
    if not topic_hint:
        return None
    return qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="metadata.topic",
                match=qdrant_models.MatchValue(value=topic_hint),
            )
        ]
    )


def _query_variants(query: str, topic_hint: str | None, tolerance: str) -> list[str]:
    if not topic_hint or tolerance == "strict":
        return [query]

    keywords = sorted(TOPICS.get(topic_hint, {}).get("keywords", []))
    label = label_for(topic_hint)
    if tolerance == "broad":
        return [
            query,
            f"{label}. {query}. {' '.join(keywords[:10])}",
        ]

    return [f"{query}. {label}. {' '.join(keywords[:5])}"]


def _dedupe_and_rank(
    docs_with_scores: list[tuple[Document, float]],
    *,
    limit: int,
    topic_hint: str | None,
) -> list[tuple[Document, float]]:
    best_by_chunk: dict[str, tuple[Document, float, float]] = {}
    for doc, score in docs_with_scores:
        metadata = doc.metadata
        key = str(metadata.get("chunk_id") or metadata.get("source") or doc.page_content[:80])
        adjusted_score = score + (0.2 if topic_hint and metadata.get("topic") == topic_hint else 0.0)
        previous = best_by_chunk.get(key)
        if previous is None or adjusted_score > previous[2]:
            best_by_chunk[key] = (doc, score, adjusted_score)
    ranked = sorted(best_by_chunk.values(), key=lambda item: item[2], reverse=True)[:limit]
    return [(doc, score) for doc, score, _ in ranked]


def get_stats(settings: Settings) -> StatsResponse:
    cache_key = (settings.qdrant_url, settings.qdrant_collection, bool(settings.openai_api_key))
    now = time.monotonic()
    with _stats_cache_lock:
        cached = _stats_cache.get(cache_key)
        if cached and now - cached[0] < STATS_CACHE_TTL_SECONDS:
            return cached[1]

    client = get_qdrant_client(settings.qdrant_url)
    qdrant_ready = True
    points_count = 0
    topic_counts: Counter[str] = Counter()
    try:
        collection_info = client.get_collection(settings.qdrant_collection)
        points_count = int(collection_info.points_count or 0)
        for topic in TOPICS:
            count_result = client.count(
                settings.qdrant_collection,
                count_filter=_topic_filter(topic),
                exact=True,
            )
            if count_result.count:
                topic_counts[topic] = int(count_result.count)
        unknown_count = max(0, points_count - sum(topic_counts.values()))
        if unknown_count:
            topic_counts["unknown"] = unknown_count
    except Exception:
        try:
            client.get_collections()
            qdrant_ready = True
        except Exception:
            qdrant_ready = False

    topics = [
        {"topic": topic, "label": label_for(topic), "chunks": count}
        for topic, count in topic_counts.most_common()
    ]
    response = StatsResponse(
        collection=settings.qdrant_collection,
        points_count=points_count,
        topics=topics,
        api_ready=bool(settings.openai_api_key),
        qdrant_ready=qdrant_ready,
    )
    with _stats_cache_lock:
        _stats_cache[cache_key] = (time.monotonic(), response)
    return response


def qdrant_collection_exists(settings: Settings) -> bool:
    client = get_qdrant_client(settings.qdrant_url)
    try:
        client.get_collection(settings.qdrant_collection)
        return True
    except UnexpectedResponse:
        return False
    except Exception:
        return False


def _stable_id(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def _short_summary(text: str) -> str:
    sentences = best_sentences("problém řešení chyba nastavení odeslat tisk přihlášení", [text], limit=1)
    return compact_text(sentences[0] if sentences else text, limit=180)


def _extract_intent(text: str) -> str:
    sentences = best_sentences("potřebuji nejde chyba problém jak kde nastavit", [text], limit=1)
    return compact_text(sentences[0] if sentences else text, limit=160)


def _extract_resolution(text: str) -> str | None:
    sentences = best_sentences("zkontrolujte změňte vyplnit nastavení uložte odešlete tiskárnu certifikát", [text], limit=1)
    if not sentences:
        return None
    return compact_text(sentences[0], limit=220)


def _quality_score(text: str, resolution: str | None) -> float:
    score = 0.35
    lowered = text.lower()
    if resolution:
        score += 0.3
    if any(term in lowered for term in ("vyřeš", "hotovo", "funguje", "podařilo", "odeslalo")):
        score += 0.2
    if len(text) > 500:
        score += 0.15
    return round(min(score, 1.0), 2)
