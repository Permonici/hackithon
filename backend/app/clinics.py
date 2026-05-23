from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha1
from math import asin, cos, radians, sin, sqrt
from zoneinfo import ZoneInfo

from .schemas import AppointmentProposal, ClinicOption, TriageResult, UserInfo


PRAGUE_TZ = ZoneInfo("Europe/Prague")


@dataclass(frozen=True)
class ClinicRecord:
    name: str
    city: str
    address: str
    phone: str
    email: str
    accepting_new_patients: bool
    services: tuple[str, ...]
    map_x: float
    map_y: float
    note: str
    urgent_delay_hours: int
    normal_delay_days: int


CLINICS: tuple[ClinicRecord, ...] = (
    ClinicRecord(
        name="XDENT Smile Praha - Karlin",
        city="Praha",
        address="Krizikova 55, Praha 8",
        phone="+420 222 010 101",
        email="praha-karlin@xdent-demo.cz",
        accepting_new_patients=True,
        services=("zubni ordinace", "dentalni hygiena"),
        map_x=46,
        map_y=35,
        note="Prijima nove pacienty, akutni sloty drzi kazdy vsedni den.",
        urgent_delay_hours=3,
        normal_delay_days=2,
    ),
    ClinicRecord(
        name="XDENT Dental Care Praha - Chodov",
        city="Praha",
        address="Roztylska 18, Praha 4",
        phone="+420 222 010 202",
        email="praha-chodov@xdent-demo.cz",
        accepting_new_patients=False,
        services=("zubni ordinace",),
        map_x=48,
        map_y=42,
        note="Aktualne bere jen akutni pacienty po telefonickem potvrzeni.",
        urgent_delay_hours=5,
        normal_delay_days=8,
    ),
    ClinicRecord(
        name="XDENT Usti Dental",
        city="Usti nad Labem",
        address="Moskevska 12, Usti nad Labem",
        phone="+420 475 010 303",
        email="usti@xdent-demo.cz",
        accepting_new_patients=True,
        services=("zubni ordinace", "dentalni hygiena"),
        map_x=44,
        map_y=25,
        note="Prijima nove pacienty a ma navaznost na recepci ordinace.",
        urgent_delay_hours=4,
        normal_delay_days=3,
    ),
    ClinicRecord(
        name="XDENT Brno Centrum",
        city="Brno",
        address="Kobližna 7, Brno",
        phone="+420 542 010 404",
        email="brno@xdent-demo.cz",
        accepting_new_patients=True,
        services=("zubni ordinace", "dentalni hygiena"),
        map_x=58,
        map_y=64,
        note="Prijima nove pacienty, vhodne pro bezne i urgentni objednani.",
        urgent_delay_hours=6,
        normal_delay_days=4,
    ),
    ClinicRecord(
        name="XDENT Plzen Bory",
        city="Plzen",
        address="Klatovska 91, Plzen",
        phone="+420 377 010 505",
        email="plzen@xdent-demo.cz",
        accepting_new_patients=True,
        services=("zubni ordinace",),
        map_x=31,
        map_y=55,
        note="Prijima nove pacienty, nejrychlejsi sloty jsou dopoledne.",
        urgent_delay_hours=7,
        normal_delay_days=5,
    ),
    ClinicRecord(
        name="XDENT Ostrava Poruba",
        city="Ostrava",
        address="Hlavni trida 44, Ostrava",
        phone="+420 596 010 606",
        email="ostrava@xdent-demo.cz",
        accepting_new_patients=True,
        services=("zubni ordinace", "dentalni hygiena"),
        map_x=82,
        map_y=52,
        note="Prijima nove pacienty, akutni pripady overuje recepce.",
        urgent_delay_hours=8,
        normal_delay_days=4,
    ),
    ClinicRecord(
        name="XDENT Hygiene Praha - Letna",
        city="Praha",
        address="Milady Horakove 31, Praha 7",
        phone="+420 222 010 707",
        email="hygiena-praha@xdent-demo.cz",
        accepting_new_patients=True,
        services=("dentalni hygiena",),
        map_x=45,
        map_y=32,
        note="Dentalni hygiena prijima nove pacienty, vhodne pro preventivni navstevy.",
        urgent_delay_hours=24,
        normal_delay_days=1,
    ),
    ClinicRecord(
        name="XDENT Hygiene Brno - Veveri",
        city="Brno",
        address="Veveri 24, Brno",
        phone="+420 542 010 808",
        email="hygiena-brno@xdent-demo.cz",
        accepting_new_patients=True,
        services=("dentalni hygiena",),
        map_x=57,
        map_y=62,
        note="Dentalni hygiena prijima nove pacienty a umi rychle preventivni terminy.",
        urgent_delay_hours=24,
        normal_delay_days=2,
    ),
)


CITY_COORDS: dict[str, tuple[float, float]] = {
    "praha": (50.0755, 14.4378),
    "usti nad labem": (50.6611, 14.0323),
    "ustí nad labem": (50.6611, 14.0323),
    "brno": (49.1951, 16.6068),
    "plzen": (49.7384, 13.3736),
    "plzeň": (49.7384, 13.3736),
    "ostrava": (49.8209, 18.2625),
}


CRITICAL_KEYWORDS = (
    "otok",
    "horecka",
    "horečku",
    "krvaceni",
    "krvácení",
    "uraz",
    "úraz",
    "zlomen",
    "absces",
    "hnis",
    "dychani",
    "dýchání",
    "nesnesitelna",
    "nesnesitelná",
)
HIGH_KEYWORDS = (
    "akut",
    "boli",
    "bolĂ­",
    "silna bolest",
    "silná bolest",
    "bolest",
    "pulzuje",
    "pulzující",
    "zánět",
    "zanet",
    "nemuze spat",
    "nemůže spát",
)
LOW_KEYWORDS = ("kontrola", "preventiv", "konzultace", "dotaz", "objednat casem")


def assess_urgency(message: str, user: UserInfo | None) -> TriageResult:
    text = _normalize(" ".join(filter(None, [message, user.problem_summary if user else None])))
    provided = user.urgency if user else None
    reasons: list[str] = []

    keyword_urgency = "normal"
    if _contains_any(text, CRITICAL_KEYWORDS):
        keyword_urgency = "critical"
        reasons.append("Text obsahuje priznaky mozne akutni komplikace.")
    elif _contains_any(text, HIGH_KEYWORDS):
        keyword_urgency = "high"
        reasons.append("Text zminuje bolest nebo akutni potrebu terminu.")
    elif _contains_any(text, LOW_KEYWORDS):
        keyword_urgency = "low"
        reasons.append("Text pusobi jako planovana nebo preventivni zalezitost.")

    urgency = _max_urgency(provided or "normal", keyword_urgency)
    if provided and provided != "normal":
        reasons.append(f"Uzivatel ve formulari oznacil urgenci: {_urgency_label(provided).lower()}.")
    if not reasons:
        reasons.append("Nebyl nalezen jasny akutni signal, volim beznou prioritu.")

    recommendation = {
        "critical": "Doporucit okamzity telefonat na ordinaci nebo pohotovost; necekat na chat.",
        "high": "Nabidnout nejblizsi akutni termin a potvrdit kontakt pacienta.",
        "normal": "Nabidnout nejblizsi bezny termin podle mesta pacienta.",
        "low": "Staci standardni objednani nebo predani recepci.",
    }[urgency]
    confidence = 0.86 if keyword_urgency in {"critical", "high"} or provided else 0.68
    return TriageResult(
        urgency=urgency,
        label=_urgency_label(urgency),
        confidence=confidence,
        reasons=reasons,
        recommendation=recommendation,
        needs_immediate_care=urgency == "critical",
    )


def list_clinics(urgency: str = "normal") -> list[ClinicOption]:
    return [_clinic_option(clinic, urgency, None) for clinic in CLINICS]


def find_nearby_clinics(
    city: str | None,
    urgency: str = "normal",
    limit: int = 5,
    service_query: str | None = None,
) -> list[ClinicOption]:
    origin = _coords_for(city)
    service_text = _normalize(service_query or "")
    ranked: list[tuple[float, ClinicRecord, float | None]] = []
    for clinic in CLINICS:
        distance = _distance_km(origin, _coords_for(clinic.city)) if origin else None
        city_bonus = 0 if _same_city(city, clinic.city) else 35
        accepting_bonus = 0 if clinic.accepting_new_patients else 25
        service_bonus = _service_bonus(clinic, service_text)
        distance_rank = distance if distance is not None else 80
        ranked.append((distance_rank + city_bonus + accepting_bonus + service_bonus, clinic, distance))

    ranked.sort(key=lambda item: (item[0], not item[1].accepting_new_patients, item[1].name))
    return [
        _clinic_option(clinic, urgency, distance)
        for _, clinic, distance in ranked[:limit]
    ]


def reserve_earliest_slot(
    *,
    message: str,
    user: UserInfo | None,
    triage: TriageResult,
    clinics: list[ClinicOption],
) -> AppointmentProposal:
    contact = _best_contact(user)
    accepting = [clinic for clinic in clinics if clinic.accepting_new_patients]
    pool = accepting or clinics
    if not pool:
        return AppointmentProposal(
            status="unavailable",
            message="V demo adresari neni dostupna ordinace pro vybrane misto.",
        )

    selected = min(pool, key=lambda clinic: clinic.earliest_slot or "9999")
    if not contact:
        return AppointmentProposal(
            status="needs_contact",
            clinic_name=selected.name,
            slot_start=selected.earliest_slot,
            message="Pro predrezervaci je potreba doplnit telefon nebo e-mail pacienta.",
        )

    seed = "|".join(
        [
            user.patient_name if user and user.patient_name else "pacient",
            contact,
            selected.name,
            selected.earliest_slot or "",
            triage.urgency,
            message[:80],
        ]
    )
    reservation_id = f"XD-{sha1(seed.encode('utf-8')).hexdigest()[:8].upper()}"
    return AppointmentProposal(
        status="pre_reserved",
        clinic_name=selected.name,
        slot_start=selected.earliest_slot,
        reservation_id=reservation_id,
        message=(
            "Termin je predrezervovany v demo workflow. Recepce musi rezervaci potvrdit "
            f"pres {contact}."
        ),
    )


def _clinic_option(clinic: ClinicRecord, urgency: str, distance: float | None) -> ClinicOption:
    return ClinicOption(
        name=clinic.name,
        city=clinic.city,
        address=clinic.address,
        distance_km=round(distance, 1) if distance is not None else None,
        accepting_new_patients=clinic.accepting_new_patients,
        services=list(clinic.services),
        map_x=clinic.map_x,
        map_y=clinic.map_y,
        phone=clinic.phone,
        email=clinic.email,
        earliest_slot=_slot_for(clinic, urgency),
        note=clinic.note,
    )


def _slot_for(clinic: ClinicRecord, urgency: str) -> str:
    now = datetime.now(PRAGUE_TZ)
    if urgency in {"critical", "high"}:
        candidate = now + timedelta(hours=clinic.urgent_delay_hours)
    else:
        candidate = now + timedelta(days=clinic.normal_delay_days)
    candidate = _business_time(candidate, urgent=urgency in {"critical", "high"})
    return candidate.strftime("%d.%m.%Y %H:%M")


def _business_time(value: datetime, *, urgent: bool) -> datetime:
    hour = value.hour
    if urgent:
        if hour < 8:
            value = value.replace(hour=8, minute=0)
        elif hour >= 17:
            value = (value + timedelta(days=1)).replace(hour=8, minute=0)
    else:
        value = value.replace(hour=10 if hour < 13 else 14, minute=0)

    while value.weekday() >= 5:
        value = (value + timedelta(days=1)).replace(hour=8 if urgent else 10, minute=0)
    return value.replace(second=0, microsecond=0)


def _best_contact(user: UserInfo | None) -> str | None:
    if not user:
        return None
    for value in (user.patient_phone, user.patient_email, user.contact):
        if value and value.strip():
            return value.strip()
    return None


def _service_bonus(clinic: ClinicRecord, service_text: str) -> int:
    if not service_text:
        return 0
    wants_hygiene = any(word in service_text for word in ("hygien", "hygiena", "cisteni", "preventiv"))
    wants_dentist = any(word in service_text for word in ("zub", "bolest", "kaz", "vypln", "akut", "ordinac"))
    services = " ".join(clinic.services)
    if wants_hygiene and "hygiena" in services:
        return -30
    if wants_hygiene and "hygiena" not in services:
        return 25
    if wants_dentist and "zubni ordinace" in services:
        return -15
    return 0


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(_normalize(keyword) in text for keyword in keywords)


def _max_urgency(left: str, right: str) -> str:
    order = {"low": 0, "normal": 1, "high": 2, "critical": 3}
    return left if order.get(left, 1) >= order.get(right, 1) else right


def _urgency_label(value: str) -> str:
    return {
        "low": "Nizka",
        "normal": "Bezna",
        "high": "Vysoka",
        "critical": "Kriticka",
    }.get(value, "Bezna")


def _same_city(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return _normalize(left) == _normalize(right)


def _coords_for(city: str | None) -> tuple[float, float] | None:
    if not city:
        return None
    normalized = _normalize(city)
    for known, coords in CITY_COORDS.items():
        if known in normalized or normalized in known:
            return coords
    return None


def _distance_km(left: tuple[float, float] | None, right: tuple[float, float] | None) -> float | None:
    if not left or not right:
        return None
    lat1, lon1 = left
    lat2, lon2 = right
    radius = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * radius * asin(sqrt(a))


def _normalize(value: str) -> str:
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
