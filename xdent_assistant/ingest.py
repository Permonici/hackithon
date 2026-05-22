from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Chunk
from .text import chunk_text, normalize_transcript


TEXT_FIELDS = ("TranscriptText", "transcript", "text", "content", "body")


@dataclass(frozen=True)
class IngestResult:
    chunks: list[Chunk]
    files_seen: int
    records_seen: int
    records_with_text: int
    skipped_files: list[str]


def build_chunks(
    data_dir: str | Path,
    *,
    chunk_size: int = 950,
    overlap: int = 150,
    include_control_group: bool = False,
) -> IngestResult:
    root = Path(data_dir)
    chunks: list[Chunk] = []
    skipped_files: list[str] = []
    files_seen = 0
    records_seen = 0
    records_with_text = 0

    for path in sorted(root.rglob("*.json")):
        if not include_control_group and "_control-group" in path.parts:
            continue

        items = _load_json_items(path)
        if items is None:
            skipped_files.append(str(path))
            continue

        files_seen += 1
        topic = path.parent.name
        relative_source = str(path.relative_to(root))

        for record_index, item in enumerate(items):
            records_seen += 1
            text = normalize_transcript(_extract_text(item))
            if not text:
                continue

            records_with_text += 1
            source_id = str(item.get("Id") or item.get("id") or record_index)
            for chunk_index, piece in enumerate(chunk_text(text, chunk_size, overlap)):
                chunk_id = f"{topic}:{path.stem}:{source_id}:{chunk_index}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=piece,
                        metadata={
                            "topic": topic,
                            "source_file": relative_source,
                            "source_id": source_id,
                            "call_entity_id": item.get("CallEntityId"),
                            "chunk_index": chunk_index,
                        },
                    )
                )

    return IngestResult(
        chunks=chunks,
        files_seen=files_seen,
        records_seen=records_seen,
        records_with_text=records_with_text,
        skipped_files=skipped_files,
    )


def _load_json_items(path: Path) -> list[dict[str, Any]] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None

    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        return [raw]
    return []


def _extract_text(item: dict[str, Any]) -> str:
    for field in TEXT_FIELDS:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""
