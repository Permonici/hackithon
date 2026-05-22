from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


ACTION_TERMS = {
    "chybi",
    "chyb",
    "doporuc",
    "kontrol",
    "nastaven",
    "odeslat",
    "over",
    "pridat",
    "schval",
    "telefon",
    "uhrad",
    "uloz",
    "vypln",
    "vyzad",
    "zmen",
    "zkus",
}

STEM_SUFFIXES = (
    "oveho",
    "ovymi",
    "ovych",
    "emi",
    "ami",
    "eho",
    "ymi",
    "ych",
    "ich",
    "ach",
    "ech",
    "ova",
    "ove",
    "ovy",
    "ou",
    "em",
    "im",
    "ym",
    "ho",
    "mi",
    "ch",
    "a",
    "e",
    "i",
    "y",
)


STOPWORDS = {
    "a",
    "aby",
    "ale",
    "ano",
    "asi",
    "bez",
    "bude",
    "budeme",
    "by",
    "bych",
    "byla",
    "bylo",
    "byly",
    "byt",
    "co",
    "coz",
    "chci",
    "chtel",
    "chtela",
    "do",
    "dobry",
    "ho",
    "ja",
    "jak",
    "jako",
    "je",
    "jeden",
    "jsem",
    "jsme",
    "jste",
    "k",
    "kdyz",
    "ke",
    "ktera",
    "ktere",
    "ktery",
    "ma",
    "mate",
    "mi",
    "mne",
    "moc",
    "mu",
    "my",
    "na",
    "nam",
    "ne",
    "nebo",
    "nejake",
    "nejaky",
    "neni",
    "no",
    "o",
    "od",
    "on",
    "ona",
    "ono",
    "pak",
    "po",
    "podle",
    "pokud",
    "potrebuji",
    "potrebujeme",
    "pro",
    "proc",
    "prosim",
    "pri",
    "se",
    "sem",
    "si",
    "tak",
    "taky",
    "tam",
    "ten",
    "tento",
    "tady",
    "te",
    "ted",
    "to",
    "tohle",
    "tom",
    "u",
    "uz",
    "vam",
    "vas",
    "ve",
    "vlastne",
    "vy",
    "za",
    "ze",
}


def strip_diacritics(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_transcript(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("*REDACTED*", " ")
    text = re.sub(r"(?<=\w)\n(?=\w)", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_search(value: str) -> str:
    return strip_diacritics(value.lower())


def tokenize(value: str) -> list[str]:
    folded = normalize_for_search(value)
    words = re.findall(r"[a-z0-9]{2,}", folded)
    tokens: list[str] = []
    for word in words:
        if word in STOPWORDS:
            continue
        tokens.append(word)
        stem = stem_token(word)
        if stem != word and stem not in STOPWORDS:
            tokens.append(stem)
    return tokens


def stem_token(word: str) -> str:
    if len(word) < 5:
        return word

    for suffix in STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def chunk_text(text: str, chunk_size: int = 950, overlap: int = 150) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        projected_len = current_len + len(word) + (1 if current else 0)
        if current and projected_len > chunk_size:
            chunks.append(" ".join(current))
            current = _tail_by_chars(current, overlap)
            current_len = len(" ".join(current))

        current.append(word)
        current_len += len(word) + (1 if current_len else 0)

    if current:
        chunks.append(" ".join(current))

    return chunks


def _tail_by_chars(words: list[str], target_chars: int) -> list[str]:
    if target_chars <= 0:
        return []

    selected: list[str] = []
    total = 0
    for word in reversed(words):
        selected.append(word)
        total += len(word) + 1
        if total >= target_chars:
            break
    return list(reversed(selected))


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def best_sentences(query: str, texts: Iterable[str], limit: int = 2) -> list[str]:
    query_tokens = set(tokenize(query))
    scored: list[tuple[float, str]] = []

    for text in texts:
        for sentence in split_sentences(text):
            sentence_tokens = set(tokenize(sentence))
            if not sentence_tokens:
                continue
            overlap = len(query_tokens & sentence_tokens)
            normalized_sentence = normalize_for_search(sentence)
            action_score = sum(1 for term in ACTION_TERMS if term in normalized_sentence)
            if overlap == 0 and action_score == 0:
                continue
            score = (overlap / max(len(query_tokens), 1)) + (action_score * 0.12)
            if "?" in sentence:
                score *= 0.8
            if re.match(r"^(jo|no|eh|aha|jasne|dobre)[,.\s]", normalize_for_search(sentence)):
                score *= 0.75
            if 35 <= len(sentence) <= 360:
                scored.append((score, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    unique: list[str] = []
    seen = set()
    for _, sentence in scored:
        compact = normalize_for_search(sentence)
        if compact in seen:
            continue
        seen.add(compact)
        unique.append(sentence)
        if len(unique) >= limit:
            break

    return unique
