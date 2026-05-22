from __future__ import annotations

from .config import Settings
from .schemas import PriceInfoResponse


CHAT_PRICES_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5.2-chat-latest": (1.75, 14.00),
    "gpt-5.1-chat-latest": (1.25, 10.00),
    "gpt-5-chat-latest": (1.25, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o3": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
}

EMBEDDING_PRICES_PER_1M: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


def resolve_price_info(settings: Settings) -> PriceInfoResponse:
    chat_input, chat_output, chat_source = resolve_chat_prices(settings)
    embedding_price, embedding_source = resolve_embedding_price(settings)
    note = _price_note(chat_source, embedding_source)
    return PriceInfoResponse(
        currency=settings.pricing_currency,
        chat_model=settings.openai_chat_model,
        embedding_model=settings.openai_embedding_model,
        chat_input_price_per_1m=chat_input,
        chat_output_price_per_1m=chat_output,
        embedding_price_per_1m=embedding_price,
        note=note,
        reference_url=settings.pricing_reference_url,
    )


def resolve_chat_prices(settings: Settings) -> tuple[float, float, str]:
    if settings.openai_chat_input_price_per_1m > 0 or settings.openai_chat_output_price_per_1m > 0:
        return (
            settings.openai_chat_input_price_per_1m,
            settings.openai_chat_output_price_per_1m,
            ".env",
        )

    known = _known_chat_prices(settings.openai_chat_model)
    if known:
        return known[0], known[1], "known"
    return 0.0, 0.0, "missing"


def resolve_embedding_price(settings: Settings) -> tuple[float, str]:
    if settings.openai_embedding_price_per_1m > 0:
        return settings.openai_embedding_price_per_1m, ".env"

    known = EMBEDDING_PRICES_PER_1M.get(settings.openai_embedding_model.lower())
    if known is not None:
        return known, "known"
    return 0.0, "missing"


def _known_chat_prices(model: str) -> tuple[float, float] | None:
    normalized = model.lower()
    if normalized in CHAT_PRICES_PER_1M:
        return CHAT_PRICES_PER_1M[normalized]

    # Snapshot aliases usually append a date after the public model name.
    for base_model, price in sorted(CHAT_PRICES_PER_1M.items(), key=lambda item: len(item[0]), reverse=True):
        if normalized.startswith(f"{base_model}-20"):
            return price
    return None


def _price_note(chat_source: str, embedding_source: str) -> str:
    if "missing" in {chat_source, embedding_source}:
        return (
            "U některého modelu není cena známá. Doplňte OPENAI_CHAT_INPUT_PRICE_PER_1M, "
            "OPENAI_CHAT_OUTPUT_PRICE_PER_1M nebo OPENAI_EMBEDDING_PRICE_PER_1M v .env."
        )
    if ".env" in {chat_source, embedding_source}:
        return "Ceny jsou převzaté z .env a slouží pro orientační odhad nákladů v demu."
    return "Ceny jsou orientační pro známé OpenAI modely a lze je přepsat v .env."
