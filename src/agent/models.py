"""Pydantic models for structured extraction by the travel agent."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TravelMetadata(BaseModel):
    """Capture core travel metadata extracted from a user message."""

    current_city: str | None = Field(default=None, description="The city the user is currently in / starting from")
    destination_city: str | None = Field(default=None, description="The city the user wants to travel to")
    budget: float | None = Field(default=None, description="The user's travel budget as a number, if mentioned")
    trip_days: int | None = Field(default=None, description="Number of days the user wants to spend on the trip, if mentioned")


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
    airline: str = Field(description="Operating airline for this leg")
    flight_number: str = Field(description="Flight number for this leg (from `route[i].flight`)")


class FlightPick(BaseModel):
    """A curated flight option highlighted to the traveller."""

    label: str = Field(description="Flight number(s) from the payload, e.g. 'DL1' or 'AA123 + BA456'")
    airline: str = Field(description="Primary airline name from the payload")
    price: float = Field(description="Total flight price in USD from the payload")
    description: str = Field(description="One short line explaining why this flight is recommended")
    legs: list[FlightLeg] = Field(
        default_factory=list,
        description="One entry per leg ONLY when the source flight has a `route`. Leave empty for direct flights.",
    )


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


class TravelPlanCuration(BaseModel):
    """LLM-produced curation of a deterministic travel payload."""

    intro: str = Field(description="One-sentence opening that confirms the trip plan")
    flights: list[FlightPick] = Field(default_factory=list, description="1-3 flight options chosen from the payload, best first")
    hotels: list[HotelPick] = Field(default_factory=list, description="1-3 hotel options chosen from the payload, best first")
    activities: list[ActivityPick] = Field(default_factory=list, description="Up to 5 activities chosen from the payload, respecting user preferences")
    sign_off: str = Field(description="One brief closing sentence")


class TravelPlan(BaseModel):
    """Final structured travel plan handed to the formatter."""

    intro: str
    sign_off: str
    flights: list[FlightPick]
    hotels: list[HotelPick]
    activities: list[ActivityPick]
    origin: str | None
    destination: str | None
    trip_days: int
    total_budget: float | None
    weather: dict[str, str]
    best_time: dict[str, Any]
    lowest_total_estimate: float | None


class IntentClassification(BaseModel):
    """Classify the user's primary intent to route them to the correct agent."""
    
    intent: Literal[
        "new_travel_plan",
        "update_travel_plan",
        "recommendations",
        "build_itinerary",
        "general_chat",
    ] = Field(
        description=(
            "Classify the user's intent strictly following these rules:\n"
            "- 'new_travel_plan': User wants a general travel package (flights, hotels, and a list of top activities) for a SPECIFIC destination.\n"
            "- 'update_travel_plan': Changing parameters of an EXISTING travel package (budget, dates, origin/destination).\n"
            "- 'recommendations': User wants to travel but has NO specific destination yet, looking for ideas.\n"
            "- 'build_itinerary': User explicitly asks for a DAY-BY-DAY chronological schedule or a time-based routing of attractions.\n"
            "- 'general_chat': Greetings, casual chat, or general travel questions.\n"
        )
    )
    has_explicit_destination: bool = Field(
        default=False,
        description="True ONLY if the user explicitly names a destination city or country in their message."
    )
