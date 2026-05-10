"""LangChain tool for calculating total trip cost."""
from langchain_core.tools import tool


@tool
def calculate_trip_cost(flight_price: float, hotel_price_per_night: float, duration_days: int) -> dict | str:
    """Calculate the total trip cost from flight price, hotel price per night, and duration in days."""
    try:
        total_hotel = float(hotel_price_per_night) * int(duration_days)
        total_grand = float(flight_price) + total_hotel
    except (ValueError, TypeError):
        return "Error: Please provide valid numbers for prices and duration."
    else:
        return {
            "breakdown": {
                "flight": flight_price,
                "hotel_total": total_hotel,
                "days": duration_days,
            },
            "total_estimate": total_grand,
            "currency": "USD",
        }
