"""Xotelo (TripAdvisor-backed) hotel API provider."""
from providers.xotelo.hotels_api import xotelo_list_hotels, xotelo_get_rates

__all__ = ["xotelo_list_hotels", "xotelo_get_rates"]
