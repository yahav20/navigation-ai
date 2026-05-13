"""Pydantic models for structured extraction by the travel agent."""

from typing import Literal
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
    max_hotel_price_per_night: float | None = Field(default=None, description="Maximum price per night for a hotel")
    max_flight_price: float | None = Field(default=None, description="Maximum acceptable flight ticket price")
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

class IntentClassification(BaseModel):
    """Classify the user's primary intent to route them to the correct agent."""
    
    intent: Literal[
        "new_travel_plan", 
        "update_travel_plan", 
        "recommendations" #, 
        # "general_interaction", 
        # "other"
    ] = Field(
        description=(
            "Classify the user's intent:\n"
            "- 'new_travel_plan': Starting a brand new trip from scratch, OR transitioning from "
            "recommendation browsing to actual trip planning (e.g. 'plan this trip', 'let's book this', "
            "'I want to go there', 'book it', 'sounds good, let's go').\n"
            "- 'update_travel_plan': Changing parameters (budget, days, destination) of an EXISTING trip.\n"
            "- 'recommendations': Asking where to go, what to do, attractions, food, or weather. "
            "Also covers refining search parameters (budget, days) within an ACTIVE RECOMMENDATION FLOW "
            "when the user is still browsing — NOT when they want to start actual trip planning.\n"
            # "- 'general_interaction': Saying hello, thanks, or casual chat.\n"
            # "- 'other': Out of scope."
        )
    )
