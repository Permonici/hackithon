from __future__ import annotations

import argparse
import json
from pathlib import Path

from xdent_assistant.assistant import XDentAssistant


def main() -> None:
    parser = argparse.ArgumentParser(description="Spustí asistenta nad evaluační sadou dotazů.")
    parser.add_argument("--index", default="storage/index.json", help="Cesta k indexu.")
    parser.add_argument("--questions", default="eval_questions.json", help="JSON se scénáři.")
    parser.add_argument("--out", default="storage/eval_results.json", help="Kam uložit výsledky.")
    parser.add_argument("--no-llm", action="store_true", help="Vynutit lokální fallback bez LLM API.")
    args = parser.parse_args()

    scenarios = _load_scenarios(args.questions)
    assistant = XDentAssistant.from_index_file(args.index, use_llm=not args.no_llm)

    results = []
    for scenario in scenarios:
        question = scenario["question"]
        response = assistant.answer(question)
        results.append(
            {
                "id": scenario.get("id"),
                "question": question,
                "expected_topic": scenario.get("expected_topic"),
                "topic": response.topic,
                "topic_label": response.topic_label,
                "topic_confidence": response.topic_confidence,
                "answer": response.answer,
                "sources": response.sources,
                "used_llm": response.used_llm,
            }
        )
        print(f"{scenario.get('id', '-')}: {response.topic_label} -> {response.sources[0]['source'] if response.sources else 'bez zdroje'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Výsledky uloženy: {out_path}")


def _load_scenarios(path: str | Path) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("Soubor scénářů musí být JSON pole.")
    return [item if isinstance(item, dict) else {"question": str(item)} for item in raw]


if __name__ == "__main__":
    main()
