from __future__ import annotations

from xdent_assistant.topics import TOPICS, classify_topic, label_for


def topic_catalog() -> list[dict[str, object]]:
    return [
        {
            "id": topic,
            "label": config["label"],
            "keywords": sorted(config["keywords"]),
        }
        for topic, config in TOPICS.items()
    ]


__all__ = ["TOPICS", "classify_topic", "label_for", "topic_catalog"]
