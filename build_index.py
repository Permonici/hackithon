from __future__ import annotations

import argparse
from pathlib import Path

from xdent_assistant.ingest import build_chunks
from xdent_assistant.retrieval import TfidfIndex


DEFAULT_DATA_DIR = r"C:\Users\medun\Desktop\hacktaton\hackathon-filtered-expanded"
DEFAULT_INDEX_PATH = "storage/index.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sestaví lokální RAG index nad XDENT transkripcemi.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Složka s JSON transkripcemi.")
    parser.add_argument("--index", default=DEFAULT_INDEX_PATH, help="Cesta pro uložený index.")
    parser.add_argument("--chunk-size", type=int, default=950, help="Velikost chunku ve znacích.")
    parser.add_argument("--overlap", type=int, default=150, help="Překryv chunků ve znacích.")
    parser.add_argument("--include-control-group", action="store_true", help="Zahrnout i _control-group.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data složka neexistuje: {data_dir}")

    ingest = build_chunks(
        data_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        include_control_group=args.include_control_group,
    )
    if not ingest.chunks:
        raise SystemExit("Nebyly nalezeny žádné textové chunky.")

    index = TfidfIndex.build(ingest.chunks)
    index.save(args.index)

    print("Index hotov.")
    print(f"Soubory: {ingest.files_seen}")
    print(f"Záznamy: {ingest.records_seen}")
    print(f"Záznamy s textem: {ingest.records_with_text}")
    print(f"Chunky: {len(ingest.chunks)}")
    print(f"Uloženo: {args.index}")
    if ingest.skipped_files:
        print(f"Přeskočené soubory: {len(ingest.skipped_files)}")


if __name__ == "__main__":
    main()
