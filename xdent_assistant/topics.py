from __future__ import annotations

from collections import Counter

from .models import SearchResult, TopicResult
from .text import normalize_for_search, tokenize


TOPICS = {
    "erecept": {
        "label": "eRecept / ePoukaz",
        "keywords": {
            "erecept",
            "recept",
            "epoukaz",
            "poukaz",
            "sukl",
            "lek",
            "sms",
            "qr",
            "uhrada",
            "diagnoza",
            "zdravotnicky",
        },
    },
    "certificate-authentication-setup": {
        "label": "Certifikát a přihlášení",
        "keywords": {
            "certifikat",
            "certifikatu",
            "autentizace",
            "prihlaseni",
            "heslo",
            "login",
            "token",
            "ucet",
            "zamestnanec",
        },
    },
    "printing-templates-documents": {
        "label": "Tisk a dokumenty",
        "keywords": {
            "tisk",
            "tiskarna",
            "sablona",
            "dokument",
            "pdf",
            "stitky",
            "formular",
            "vytisknout",
        },
    },
    "installation-setup": {
        "label": "Instalace a nastavení",
        "keywords": {
            "instalace",
            "nainstalovat",
            "nastaveni",
            "konfigurace",
            "aktualizace",
            "server",
            "pocitac",
        },
    },
    "calendar-scheduling-booking": {
        "label": "Kalendář a objednávání",
        "keywords": {
            "kalendar",
            "objednat",
            "objednavka",
            "termin",
            "rezervace",
            "ordinacni",
            "pacient",
        },
    },
    "vzp": {
        "label": "VZP / pojišťovny",
        "keywords": {
            "vzp",
            "pojistovna",
            "davka",
            "vykaz",
            "vykazovani",
            "kod",
            "uhrada",
            "kontrola",
        },
    },
    "integrations": {
        "label": "Integrace",
        "keywords": {
            "integrace",
            "import",
            "export",
            "api",
            "laborator",
            "laborka",
            "cenik",
            "externi",
        },
    },
    "how-to-product-navigation": {
        "label": "Orientace v produktu",
        "keywords": {
            "kde",
            "najdu",
            "otevrit",
            "zalozka",
            "menu",
            "navigace",
            "postup",
        },
    },
    "feature-requests-usability": {
        "label": "Požadavky a použitelnost",
        "keywords": {
            "navrh",
            "pozadavek",
            "funkce",
            "chybi",
            "upravit",
            "zlepseni",
            "pouzitelnost",
        },
    },
}


def classify_topic(question: str) -> TopicResult:
    tokens = set(tokenize(question))
    normalized_question = normalize_for_search(question)
    scores: dict[str, float] = {}

    for topic, config in TOPICS.items():
        score = 0.0
        for keyword in config["keywords"]:
            normalized_keyword = normalize_for_search(keyword)
            if normalized_keyword in tokens:
                score += 1.0
            elif " " in normalized_keyword and normalized_keyword in normalized_question:
                score += 1.5
        if score:
            scores[topic] = score

    if not scores:
        return TopicResult(None, "neurčeno", 0.0)

    best_topic, best_score = max(scores.items(), key=lambda item: item[1])
    total_score = sum(scores.values())
    confidence = best_score / total_score if total_score else 0.0
    return TopicResult(best_topic, label_for(best_topic), round(confidence, 3))


def infer_topic_from_results(results: list[SearchResult]) -> TopicResult:
    if not results:
        return TopicResult(None, "neurčeno", 0.0)

    weighted = Counter()
    for result in results:
        topic = result.chunk.metadata.get("topic")
        if topic:
            weighted[topic] += result.score

    if not weighted:
        return TopicResult(None, "neurčeno", 0.0)

    best_topic, best_score = weighted.most_common(1)[0]
    total = sum(weighted.values())
    confidence = float(best_score / total) if total else 0.0
    return TopicResult(best_topic, label_for(best_topic), round(confidence, 3))


def label_for(topic: str | None) -> str:
    if not topic:
        return "neurčeno"
    return TOPICS.get(topic, {}).get("label", topic)
