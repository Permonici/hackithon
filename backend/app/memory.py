from __future__ import annotations

import json
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schemas import UserInfo
from .utils import compact_text


EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?:\+?420\s*)?(?:\d[\s.-]?){9,}")
NAME_RE = re.compile(
    r"(?:jmenuji se|jmenuju se|pacient(?:ka)?(?: je)?|pro pacienta)\s+"
    r"([A-ZÁ-Ž][A-Za-zÁ-ž.-]+(?:\s+[A-ZÁ-Ž][A-Za-zÁ-ž.-]+){0,2})",
    re.IGNORECASE,
)

CITY_NAMES = (
    "Praha",
    "Brno",
    "Ostrava",
    "Plzen",
    "Plzeň",
    "Usti nad Labem",
    "Ústí nad Labem",
    "Liberec",
    "Olomouc",
    "Hradec Kralove",
    "Hradec Králové",
    "Pardubice",
    "Ceske Budejovice",
    "České Budějovice",
)

CITY_VARIANTS = {
    "Praha": ("praha", "prahy", "praze"),
    "Brno": ("brno", "brna", "brne", "brně"),
    "Ostrava": ("ostrava", "ostravy", "ostrave", "ostravě"),
    "Plzen": ("plzen", "plzne", "plzni"),
    "Plzeň": ("plzeň", "plzně", "plzni"),
    "Usti nad Labem": ("usti nad labem", "usti"),
    "Ústí nad Labem": ("ústí nad labem", "ústí"),
}

PATIENT_SIGNALS = (
    "pacient",
    "pacientka",
    "bolest",
    "zub",
    "otok",
    "krvac",
    "objednat",
    "termin",
    "termín",
    "kontakt",
    "telefon",
    "email",
    "e-mail",
    "mesto",
    "město",
)


class PatientMemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def merge(
        self,
        *,
        session_id: str | None,
        incoming: UserInfo | None,
        message: str,
        agent_mode: str,
    ) -> tuple[UserInfo | None, list[str]]:
        extracted = self._extract(message, agent_mode)
        if incoming:
            extracted = {**incoming.model_dump(exclude_none=True), **extracted}

        if not session_id:
            return (UserInfo(**extracted) if extracted else incoming), self._labels_for(extracted)

        with self._lock:
            data = self._read_all()
            current = data.get(session_id, {})
            merged = dict(current)
            updates: list[str] = []

            for key, value in extracted.items():
                if not value:
                    continue
                if merged.get(key) != value:
                    merged[key] = value
                    updates.append(self._label_for(key))

            if updates:
                data[session_id] = merged
                self._write_all(data)

            return (UserInfo(**merged) if merged else incoming), updates

    def forget(self, session_id: str) -> bool:
        with self._lock:
            data = self._read_all()
            existed = session_id in data
            if existed:
                del data[session_id]
                self._write_all(data)
            return existed

    def _read_all(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
            if isinstance(raw, dict):
                return {
                    str(key): value
                    for key, value in raw.items()
                    if isinstance(value, dict)
                }
        except Exception:
            return {}
        return {}

    def _write_all(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    def _extract(self, message: str, agent_mode: str) -> dict[str, Any]:
        lowered = message.lower()
        data: dict[str, Any] = {}
        email = EMAIL_RE.search(message)
        if email:
            data["patient_email"] = email.group(0)

        phone = PHONE_RE.search(message)
        if phone:
            data["patient_phone"] = re.sub(r"\s+", " ", phone.group(0)).strip(" .-")

        name = NAME_RE.search(message)
        if name:
            data["patient_name"] = name.group(1).strip(" .,")

        city = self._find_city(message)
        if city:
            data["patient_city"] = city

        urgency = self._urgency_from_text(lowered)
        if urgency:
            data["urgency"] = urgency

        strong_patient_signal = (
            agent_mode in {"patient", "handoff"}
            or bool(data)
            or any(term in lowered for term in ("boli", "bolest", "otok", "zub", "pohotovost", "nejdrivejsi termin"))
        )
        if not strong_patient_signal:
            return {}

        if len(message.strip()) >= 18 and self._looks_like_problem(lowered, agent_mode):
            data["problem_summary"] = compact_text(message, limit=360)

        return data

    def _find_city(self, message: str) -> str | None:
        normalized = self._normalize(message)
        for city, variants in CITY_VARIANTS.items():
            if any(self._normalize(variant) in normalized for variant in variants):
                return city
        for city in CITY_NAMES:
            city_normalized = self._normalize(city)
            if city_normalized in normalized:
                return city
        return None

    def _urgency_from_text(self, lowered: str) -> str | None:
        if any(term in lowered for term in ("otok", "horec", "horeč", "krvac", "krvác", "uraz", "úraz", "nesnesiteln")):
            return "critical"
        if any(term in lowered for term in ("akut", "boli", "bolí", "silna bolest", "silná bolest", "bolest", "pulzuje")):
            return "high"
        if any(term in lowered for term in ("kontrola", "prevence", "preventiv", "hygiena")):
            return "low"
        return None

    def _looks_like_problem(self, lowered: str, agent_mode: str) -> bool:
        if agent_mode in {"patient", "handoff"}:
            return not lowered.strip() in {"ano", "ne", "ok", "diky", "děkuji", "dekuji"}
        return any(signal in lowered for signal in PATIENT_SIGNALS)

    def _labels_for(self, data: dict[str, Any]) -> list[str]:
        return [self._label_for(key) for key, value in data.items() if value]

    def _label_for(self, key: str) -> str:
        labels = {
            "patient_name": "jmeno pacienta",
            "patient_phone": "telefon",
            "patient_email": "e-mail",
            "patient_city": "mesto",
            "urgency": "urgence",
            "problem_summary": "popis problemu",
        }
        return labels.get(key, key)

    def _normalize(self, value: str) -> str:
        return (
            value.lower()
            .replace("á", "a")
            .replace("č", "c")
            .replace("ď", "d")
            .replace("é", "e")
            .replace("ě", "e")
            .replace("í", "i")
            .replace("ň", "n")
            .replace("ó", "o")
            .replace("ř", "r")
            .replace("š", "s")
            .replace("ť", "t")
            .replace("ú", "u")
            .replace("ů", "u")
            .replace("ý", "y")
            .replace("ž", "z")
        )


@lru_cache(maxsize=4)
def get_patient_memory_store(path: str) -> PatientMemoryStore:
    return PatientMemoryStore(Path(path))
