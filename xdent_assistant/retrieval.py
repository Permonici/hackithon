from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .models import Chunk, SearchResult
from .text import tokenize


class TfidfIndex:
    version = 1

    def __init__(
        self,
        chunks: list[Chunk],
        idf: dict[str, float],
        vectors: list[dict[str, float]],
        norms: list[float],
    ) -> None:
        self.chunks = chunks
        self.idf = idf
        self.vectors = vectors
        self.norms = norms

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "TfidfIndex":
        token_counts = [Counter(tokenize(chunk.text)) for chunk in chunks]
        document_frequency: Counter[str] = Counter()
        for counts in token_counts:
            document_frequency.update(counts.keys())

        total_docs = max(len(chunks), 1)
        idf = {
            token: math.log((1 + total_docs) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }

        vectors: list[dict[str, float]] = []
        norms: list[float] = []
        for counts in token_counts:
            vector = _weighted_vector(counts, idf)
            vectors.append(vector)
            norms.append(_norm(vector))

        return cls(chunks=chunks, idf=idf, vectors=vectors, norms=norms)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        topic_hint: str | None = None,
    ) -> list[SearchResult]:
        query_counts = Counter(tokenize(query))
        if not query_counts:
            return []

        query_vector = _weighted_vector(query_counts, self.idf)
        query_norm = _norm(query_vector)
        if query_norm == 0:
            return []

        scored: list[SearchResult] = []
        for index, vector in enumerate(self.vectors):
            norm = self.norms[index]
            if norm == 0:
                continue

            score = _cosine(query_vector, query_norm, vector, norm)
            chunk = self.chunks[index]
            if topic_hint and chunk.metadata.get("topic") == topic_hint:
                score *= 1.35
            if score > 0:
                scored.append(SearchResult(chunk=chunk, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "idf": self.idf,
            "chunks": [
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "vector": vector,
                    "norm": norm,
                }
                for chunk, vector, norm in zip(self.chunks, self.vectors, self.norms)
            ],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TfidfIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != cls.version:
            raise ValueError("Nepodporovaná verze indexu. Sestavte index znovu.")

        chunks = [
            Chunk(id=item["id"], text=item["text"], metadata=item["metadata"])
            for item in payload["chunks"]
        ]
        vectors = [item["vector"] for item in payload["chunks"]]
        norms = [item["norm"] for item in payload["chunks"]]
        return cls(chunks=chunks, idf=payload["idf"], vectors=vectors, norms=norms)


def source_label(chunk: Chunk) -> str:
    source_file = chunk.metadata.get("source_file", "neznámý zdroj")
    source_id = chunk.metadata.get("source_id")
    chunk_index = chunk.metadata.get("chunk_index")
    if source_id is not None:
        return f"{source_file}#Id={source_id}/chunk={chunk_index}"
    return f"{source_file}/chunk={chunk_index}"


def result_to_source(result: SearchResult) -> dict[str, Any]:
    return {
        "source": source_label(result.chunk),
        "topic": result.chunk.metadata.get("topic"),
        "score": round(result.score, 4),
    }


def _weighted_vector(counts: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    vector: dict[str, float] = {}
    for token, count in counts.items():
        token_idf = idf.get(token)
        if token_idf is None:
            continue
        vector[token] = (1 + math.log(count)) * token_idf
    return vector


def _norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(weight * weight for weight in vector.values()))


def _cosine(
    query_vector: dict[str, float],
    query_norm: float,
    document_vector: dict[str, float],
    document_norm: float,
) -> float:
    if len(query_vector) > len(document_vector):
        query_vector, document_vector = document_vector, query_vector

    dot = 0.0
    for token, query_weight in query_vector.items():
        dot += query_weight * document_vector.get(token, 0.0)
    return dot / (query_norm * document_norm)
