"""Data provider extended with recommendation-specific queries."""
from providers.sqlite.provider import SQLiteDataProvider
from recommendation.queries.destination_queries import RecommendationQueriesMixin


class RecommendationDataProvider(RecommendationQueriesMixin, SQLiteDataProvider):
    """Extends the core SQLiteDataProvider with queries needed by the recommendation agent."""
