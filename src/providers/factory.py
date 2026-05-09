from .base import BaseDataProvider
from .sqlite_provider import SQLiteDataProvider
from .json_provider import JSONDataProvider

def create_data_provider(provider_type: str = "sqlite") -> BaseDataProvider:
    """
    Factory method to create a data provider instance.

    Args:
        provider_type: Type of provider to create ("sqlite" or "json")

    Returns:
        An instance of the requested data provider
    """
    if provider_type == "sqlite":
        return SQLiteDataProvider()
    elif provider_type == "json":
        return JSONDataProvider()
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
