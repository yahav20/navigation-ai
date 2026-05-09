from typing import Optional
from pydantic import BaseModel, Field


class TravelMetadata(BaseModel):
    current_city: Optional[str] = Field(default=None, description="The city the user is currently in / starting from")
    destination_city: Optional[str] = Field(default=None, description="The city the user wants to travel to")
    budget: Optional[float] = Field(default=None, description="The user's travel budget as a number, if mentioned")
    trip_days: Optional[int] = Field(default=None, description="Number of days the user wants to spend on the trip, if mentioned")


class UserPreferences(BaseModel):
    min_hotel_stars: Optional[int] = Field(default=None, description="Minimum hotel star rating preferred by the user (1-5)")
    max_hotel_price_per_night: Optional[float] = Field(default=None, description="Maximum price per night for a hotel")
    max_flight_price: Optional[float] = Field(default=None, description="Maximum acceptable flight ticket price")
    preferred_airline: Optional[str] = Field(default=None, description="Preferred airline name if mentioned")


class RefusalDetection(BaseModel):
    refusing_origin_city: bool = Field(default=False, description="User explicitly refuses or claims to be unable to provide their origin/departure city")
    refusing_destination_city: bool = Field(default=False, description="User explicitly refuses or claims to be unable to provide their destination city")
    refusing_budget: bool = Field(default=False, description="User explicitly refuses or claims to be unable to provide their travel budget")
