"""Pydantic models for structured extraction by the travel agent."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TravelMetadata(BaseModel):
    """Capture core travel metadata extracted from a user message."""

    current_city: str | None = Field(default=None, description="The city the user is currently in / starting from")
    destination_city: str | None = Field(default=None, description="The city the user wants to travel to")
    budget: float | None = Field(default=None, description="The user's travel budget as a number, if mentioned")
    trip_days: int | None = Field(default=None, description="Number of days the user wants to spend on the trip, if mentioned")
    trip_start: str | None = Field(
        default=None,
        description=(
            "Approximate trip start. Use YYYY-MM-DD when the user names a specific day "
            "(e.g. 'leaving June 10' -> '2026-06-10'). Use YYYY-MM when the user only "
            "names a month or a relative period (e.g. 'in June' -> '2026-06', "
            "'next month' resolved against today's date). Return null if nothing about "
            "timing is mentioned."
        ),
    )


class UserPreferences(BaseModel):
    """Capture optional user preferences for filtering travel options."""

    min_hotel_stars: int | None = Field(default=None, description="Minimum hotel star rating preferred by the user (1-5)")
    max_hotel_price_per_night: float | None = Field(default=None, description="Maximum price per night for a hotel. ONLY extract if explicitly stated by the user. DO NOT calculate or split from the total budget.")
    max_flight_price: float | None = Field(default=None, description="Maximum acceptable flight ticket price. ONLY extract if explicitly stated by the user. DO NOT calculate or split from the total budget.")

    preferred_airline: str | None = Field(default=None, description="Preferred airline name if mentioned")

    dietary_restrictions: str | None = Field(default=None, description="Food or dietary preferences (e.g., vegan, vegetarian, kosher)")
    hotel_amenities: str | None = Field(default=None, description="Specific amenities requested for the hotel (e.g., pool, gym, balcony)")
    preferred_location: str | None = Field(default=None, description="Preferred location vibe or area (e.g., city center, near the beach)")


class RefusalDetection(BaseModel):
    """Track whether the user has refused to share specific fields."""

    refusing_origin_city: bool = Field(default=False, description="User explicitly refuses or claims to be unable to provide their origin/departure city")
    refusing_destination_city: bool = Field(default=False, description="User explicitly refuses or claims to be unable to provide their destination city")
    refusing_budget: bool = Field(default=False, description="User explicitly refuses or claims to be unable to provide their travel budget")


class TravelAdjustments(BaseModel):
    """Detect if the user explicitly wants to adjust their existing travel parameters."""

    is_adjustment: bool = Field(default=False, description="True ONLY if the user is explicitly changing their destination, origin, budget, or trip days.")
    new_destination: str | None = Field(default=None, description="The new destination city, if updated.")
    new_origin: str | None = Field(default=None, description="The new origin city, if updated.")
    new_budget: float | None = Field(default=None, description="The new total budget, if updated.")
    new_trip_days: int | None = Field(default=None, description="The new trip duration in days, if updated.")

class FlightLeg(BaseModel):
    """A single leg of a connecting-flight route."""

    from_city: str = Field(description="Origin city for this leg (from `route[i].from`)")
    to_city: str = Field(description="Destination city for this leg (from `route[i].to`)")
    airline: str | None = Field(default=None, description="Operating airline for this leg")
    flight_number: str = Field(description="Flight number for this leg (from `route[i].flight`)")


class FlightPick(BaseModel):
    """A curated flight option highlighted to the traveller."""

    label: str = Field(description="Flight number(s) from the payload, e.g. 'DL1' or 'AA123 + BA456'")
    airline: str | None = Field(default=None, description="Primary airline name from the payload")
    price: float = Field(description="Total flight price in USD from the payload")
    description: str = Field(description="One short line explaining why this flight is recommended")
    duration_minutes: int | None = Field(
        default=None,
        description=(
            "Total trip duration in minutes. Copy the source flight's `duration_minutes` "
            "when present (single-segment offers). For multi-leg routes, sum "
            "`route[i].duration_minutes` across all legs. Leave null when no leg has a duration."
        ),
    )
    departure_time: str | None = Field(
        default=None,
        description=(
            "Source flight's `departure_time` (or first leg's `departure_time` for multi-leg routes) "
            "as the ISO string from the payload. Leave null when the payload has no departure time."
        ),
    )
    destination_airport: str | None = Field(
        default=None,
        description=(
            "IATA code of the final arrival airport from the source flight's `destination_airport` "
            "field (helpful when a city has multiple airports, e.g. CDG vs ORY). Leave null when absent."
        ),
    )
    stop_airports: list[str] = Field(
        default_factory=list,
        description=(
            "IATA codes of the intermediate stop airports (origin + final destination excluded) "
            "from the source flight's `stop_airports` array. Empty for direct flights. Copy as-is."
        ),
    )
    stops: int = Field(
        default=0,
        description=(
            "Number of layovers. Copy the source flight's `transfers` value when present, "
            "otherwise set to `len(route) - 1` when the source has a `route` array, "
            "otherwise 0 for a single direct segment."
        ),
    )
    legs: list[FlightLeg] = Field(
        default_factory=list,
        description=(
            "One entry per leg ONLY when the source flight has an itemized `route` array "
            "(SQLite-backed multi-leg routes). Leave empty when the source is a single "
            "direct/connecting offer from the live API — `stops` already conveys the layover count."
        ),
    )


class FlightPairing(BaseModel):
    """A complete round-trip option: one outbound + one return + the combined price."""

    outbound: FlightPick = Field(description="The outbound (origin → destination) flight picked from the payload's `flights` array.")
    return_flight: FlightPick = Field(description="The return (destination → origin) flight picked from the payload's `return_flights` array.")
    total_price: float = Field(description="Sum of outbound.price and return_flight.price in USD.")
    description: str = Field(description="One short line on why this pairing works as a complete round trip (timing, price, layovers).")


class HotelPick(BaseModel):
    """A curated hotel option highlighted to the traveller."""

    name: str = Field(description="Hotel name from the payload")
    stars: int | None = Field(default=None, description="Hotel star rating from the payload, if known")
    price_per_night: float = Field(description="Nightly rate in USD from the payload")
    description: str = Field(description="One short line explaining why this hotel is recommended")


class ActivityPick(BaseModel):
    """A curated activity suggested to the traveller."""

    name: str = Field(description="Activity name from the payload")
    description: str = Field(description="One short line on why this activity fits the trip and the user preferences")


class RestaurantPick(BaseModel):
    """A curated restaurant suggested to the traveller."""

    name: str = Field(description="Restaurant name from the payload")
    price_tier: str | None = Field(default=None, description="Price tier label from the payload (e.g. '$', '$$', '$$$')")
    rating: float | None = Field(default=None, description="Rating from the payload, if available")
    description: str = Field(description="One short line on why this restaurant is recommended")


class TravelPlanCuration(BaseModel):
    """LLM-produced curation of a deterministic travel payload."""

    intro: str = Field(description="One-sentence opening that confirms the trip plan")
    flight_pairings: list[FlightPairing] = Field(
        default_factory=list,
        description=(
            "Exactly 3 round-trip options (or fewer only if the payload has fewer than 3 viable "
            "combinations). Each pairing combines one outbound flight from `flights` with one "
            "return flight from `return_flights`. Pick combinations that actually fit together — "
            "vary across budget/speed tradeoffs (cheapest, fastest, balanced), and prefer pairings "
            "where the airlines, timings, or layover styles complement each other."
        ),
    )
    hotels: list[HotelPick] = Field(default_factory=list, description="Exactly 3 hotel options chosen from the payload (fewer only if payload has fewer than 3). Pick at varied price points: one budget-friendly, one mid-range, one premium — so the user sees real choices.")
    activities: list[ActivityPick] = Field(default_factory=list, description="Up to 5 activities chosen from the payload, respecting user preferences")
    restaurants: list[RestaurantPick] = Field(default_factory=list, description="Up to 3 restaurant picks from the payload's restaurants list")
    sign_off: str = Field(description="One brief closing sentence")


class TravelPlan(BaseModel):
    """Final structured travel plan handed to the formatter."""

    intro: str
    sign_off: str
    flight_pairings: list[FlightPairing] = Field(default_factory=list)
    hotels: list[HotelPick]
    activities: list[ActivityPick]
    restaurants: list[RestaurantPick] = Field(default_factory=list)
    origin: str | None
    destination: str | None
    trip_days: int
    trip_start: str | None = None
    total_budget: float | None
    weather: dict[str, str]
    best_time: dict[str, Any]
    lowest_total_estimate: float | None


class IntentClassification(BaseModel):
    """Classify the user's primary intent to route them to the correct agent."""

    intent: Literal[
        "new_travel_plan",
        "update_travel_plan",
        "advisor",
        "build_itinerary",
        "out_of_scope",
    ] = Field(
        description=(
            "Classify the user's intent. When in doubt, choose 'advisor'.\n"
            "- 'advisor': The DEFAULT for ALL travel-related queries AND conversational openers. "
            "Use this for: destination ideas ('where should I go?'), activities in a city, weather, "
            "city profiles, best time to visit, trip duration advice, budget exploration, "
            "travel recommendations, currency exchange ('how much is $500 in euros?'), "
            "visa requirements ('do I need a visa for Japan?'), travel safety ('is Bangkok safe?'), "
            "packing questions ('what should I pack for Tokyo?'), local customs and etiquette, "
            "greetings ('hello', 'hi', 'thanks'), and capability questions ('what can you do?'). "
            "When in doubt, use 'advisor'.\n"
            "- 'new_travel_plan': ONLY when the user commits to a SPECIFIC destination and wants to CHECK "
            "or BOOK flights, hotels, or costs. (e.g. 'I want to fly to Rome', 'show me hotels in Madrid', "
            "'let\\'s plan a trip to Tokyo'). Do NOT use just because a city is mentioned.\n"
            "- 'update_travel_plan': Changing parameters (budget, days, destination) of an ALREADY-PLANNED trip.\n"
            "- 'build_itinerary': ONLY when user EXPLICITLY requests a day-by-day SCHEDULE. "
            "Trigger phrases: 'build an itinerary', 'plan my days', 'day-by-day schedule', 'replan'. "
            "Questions like 'how should I split my time', 'what should I do in Paris for 3 days', "
            "'give me ideas for a 5-day trip', or 'can you help me plan my trip' are 'advisor', NOT 'build_itinerary'.\n"
            "- 'out_of_scope': ONLY for queries with ZERO travel relevance — food recipes, math, "
            "coding, programming, general trivia, sports scores, anything a travel assistant should not touch. "
            "When in doubt, prefer 'advisor' over 'out_of_scope'.\n"
        )
    )
    has_explicit_destination: bool = Field(
        default=False,
        description=(
            "True ONLY if the user explicitly names a DESTINATION city (where they want TO GO or VISIT), "
            "not their origin/current/departure city. "
            "Example: 'I want to fly to Rome' → True (Rome is destination). "
            "'I\\'m flying FROM Tel Aviv' → False (Tel Aviv is origin, not destination)."
        )
    )
