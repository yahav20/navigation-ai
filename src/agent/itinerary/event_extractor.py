"""LLM-based structured parser for special event data scraped from timeout.com snippets."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic schema — structured output target for the LLM
# ---------------------------------------------------------------------------

class SpecialEventRaw(BaseModel):
    """A single date-specific event extracted from timeout.com article snippets."""

    name: str = Field(
        description="Short, readable event name (e.g. 'Oktoberfest', 'Christmas Market at Old Town Square')."
    )
    description: str = Field(
        description="One or two sentence description of the event."
    )
    applicable_dates: list[str] = Field(
        description=(
            "List of YYYY-MM-DD dates (within the travel window provided) on which the event occurs. "
            "If it runs every day of a range, list each date individually."
        )
    )
    time_start: str = Field(
        description=(
            "Opening / start time in HH:MM (24-h) format. "
            "Use '' (empty string) if genuinely unknown or the event runs all day."
        )
    )
    time_end: str = Field(
        description=(
            "Closing / end time in HH:MM (24-h) format. "
            "Use '' (empty string) if genuinely unknown."
        )
    )
    cost_usd: float = Field(
        description=(
            "Estimated cost in USD per person. "
            "Convert local currency using approximate rates. "
            "For free-entry events (markets, festivals), estimate a realistic snack / "
            "browsing spend (typically $10–$30). "
            "Set 0.0 only if the event is truly free with no expected spending."
        )
    )
    suggested_visit_hours: float = Field(
        description=(
            "Realistic visit duration in hours — not the event's total operating hours. "
            "A Christmas market = 2–3 h; a day festival with multiple stages = 4–5 h."
        )
    )
    is_evening_only: bool = Field(
        description=(
            "True when the event only makes sense after 18:00 "
            "(night markets, beach parties, fireworks, New Year's Eve). "
            "False for daytime events or events that run all day."
        )
    )
    location_hint: str = Field(
        description=(
            "Most specific venue or area name found in the article — used for geocoding. "
            "Examples: 'Marienplatz', 'Old Town Square', 'Khao San Road', 'central Bangkok'. "
            "Must always be filled; use the destination city name as fallback."
        )
    )


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

def _build_prompt(city: str, date_from: str, date_to: str) -> str:
    return f"""You are extracting structured special-event data from travel article snippets.

Travel window : {date_from} to {date_to}
Destination   : {city}

Rules:
1. Only extract REAL, date-specific events (festivals, markets, concerts, seasonal celebrations).
   Do NOT extract permanent attractions, restaurants, museums, or generic tourist spots.
2. Only include events that physically take place IN {city} itself.
   Discard any event that requires travel to another city, island, or region.
3. Fill applicable_dates with every date within [{date_from}, {date_to}] on which the event runs.
4. Convert costs to USD. For free-entry events, estimate realistic snack/browsing spending.
5. suggested_visit_hours = realistic visitor stay, not total event operating hours.
6. Always fill location_hint with the most specific venue/area name from the text.
7. If no events are found, return an empty list."""


# ---------------------------------------------------------------------------
# Public extractor
# ---------------------------------------------------------------------------

def extract_special_events(
    llm,
    raw_results: list[dict],
    city: str,
    date_from: str,
    date_to: str,
) -> list[SpecialEventRaw]:
    """Parse Tavily snippets into structured SpecialEventRaw objects.

    Returns an empty list on any error (never raises).
    """
    if not raw_results:
        return []

    # Build a compact text block from the search snippets
    lines: list[str] = []
    for r in raw_results:
        title   = r.get("title", "")
        content = r.get("content", "")
        if title or content:
            lines.append(f"--- {title} ---\n{content}")
    article_text = "\n\n".join(lines).strip()
    if not article_text:
        return []

    system_prompt = _build_prompt(city, date_from, date_to)
    user_message  = f"Extract events from the following snippets:\n\n{article_text}"

    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415
        structured_llm = llm.with_structured_output(list[SpecialEventRaw])
        result = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])
        if isinstance(result, list):
            return [e for e in result if isinstance(e, SpecialEventRaw)]
        return []
    except Exception:  # noqa: BLE001
        return []
