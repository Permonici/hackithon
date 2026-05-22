from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

MAX_CACHE_SIZE = 200
DEFAULT_TTL = 7_200       # 2 hours
FREQUENT_THRESHOLD = 3    # queries seen this many times are persisted to disk


class QueryCache:
    """
    Thread-safe in-memory response cache with TTL and per-query frequency tracking.

    Identical queries (same message + strict_mode + tolerance + top_k) bypass the
    full RAG pipeline and return the cached ChatResponse instantly.  Frequently
    asked queries (≥ FREQUENT_THRESHOLD hits) are persisted to disk so that hot
    queries survive a server restart.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, tuple[Any, float]] = {}   # key → (value, expire_monotonic)
        self._freq: dict[str, int] = {}                   # normalized_query → hit count
        self._persist_path = persist_path
        self._load_freq()

    # ── public API ────────────────────────────────────────────────────────────

    def build_key(
        self,
        message: str,
        strict_mode: bool,
        tolerance: str,
        top_k: int,
    ) -> tuple[str, str]:
        """Return (cache_key_hex, normalized_query)."""
        normalized = " ".join(message.lower().split())
        raw = f"{normalized}|{strict_mode}|{tolerance}|{top_k}"
        return hashlib.sha256(raw.encode()).hexdigest(), normalized

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expire_at = entry
            if time.monotonic() > expire_at:
                del self._store[key]
                return None
            return value

    def put(
        self,
        key: str,
        normalized_query: str,
        value: Any,
        ttl: int = DEFAULT_TTL,
    ) -> None:
        with self._lock:
            if len(self._store) >= MAX_CACHE_SIZE:
                self._evict()
            self._store[key] = (value, time.monotonic() + ttl)
            self._freq[normalized_query] = self._freq.get(normalized_query, 0) + 1
            if self._freq[normalized_query] >= FREQUENT_THRESHOLD:
                self._persist_freq()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            active = sum(1 for _, exp in self._store.values() if exp > now)
            top = sorted(self._freq.items(), key=lambda x: x[1], reverse=True)[:15]
            return {
                "active_entries": active,
                "total_tracked_queries": len(self._freq),
                "top_frequent": [{"query": q, "count": c} for q, c in top],
            }

    # ── internal helpers ──────────────────────────────────────────────────────

    def _evict(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._store.items() if exp <= now]
        for k in expired:
            del self._store[k]
        if len(self._store) >= MAX_CACHE_SIZE:
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]

    def _persist_freq(self) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self._persist_path.open("w", encoding="utf-8") as fh:
                json.dump(self._freq, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_freq(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with self._persist_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    self._freq = {
                        k: v for k, v in data.items()
                        if isinstance(k, str) and isinstance(v, int)
                    }
        except Exception:
            pass
