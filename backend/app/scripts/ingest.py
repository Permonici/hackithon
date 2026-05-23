from __future__ import annotations

from app.config import get_settings
from app.vectorstore import ingest_transcripts


def main() -> None:
    settings = get_settings()
    result = ingest_transcripts(settings, recreate=False)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
