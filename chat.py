from __future__ import annotations

import argparse

from xdent_assistant.assistant import XDentAssistant


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI chat pro XDENT support asistenta.")
    parser.add_argument("--index", default="storage/index.json", help="Cesta k indexu.")
    parser.add_argument("--question", "-q", help="Jednorázový dotaz. Bez něj se spustí interaktivní režim.")
    parser.add_argument("--top-k", type=int, default=5, help="Počet chunků předaných do odpovědi.")
    parser.add_argument("--log", default="logs/interactions.jsonl", help="JSONL log interakcí.")
    parser.add_argument("--no-llm", action="store_true", help="Vynutit lokální fallback bez LLM API.")
    args = parser.parse_args()

    assistant = XDentAssistant.from_index_file(args.index, log_path=args.log, use_llm=not args.no_llm)

    if args.question:
        _print_answer(assistant.answer(args.question, top_k=args.top_k))
        return

    print("XDENT asistent. Ukončení: prázdný řádek nebo Ctrl+C.")
    while True:
        try:
            question = input("\nDotaz: ").strip()
        except KeyboardInterrupt:
            print()
            break

        if not question:
            break
        _print_answer(assistant.answer(question, top_k=args.top_k))


def _print_answer(response) -> None:
    print("\nOdpověď:")
    print(response.answer)
    print("\nMetadata:")
    print(f"- téma: {response.topic_label} ({response.topic_confidence})")
    print(f"- LLM: {'ano' if response.used_llm else 'ne'}")
    for source in response.sources:
        print(f"- zdroj: {source['source']} | score={source['score']}")


if __name__ == "__main__":
    main()
