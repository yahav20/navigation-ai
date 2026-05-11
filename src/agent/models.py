"""Pydantic models for structured extraction by the travel agent."""

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