"""LangChain tools for Wikipedia and Wikivoyage enrichment."""
from langchain_core.tools import tool

from providers.wikipedia import fetch_wiki_summary, fetch_wikidata_id


@tool
def fetch_attraction_details(attraction_name: str) -> dict:
    """Fetch Wikipedia details about an attraction.

    Returns a one-liner description, Wikipedia URL, and image URL if available.
    """
    if not attraction_name:
        return {}

    wiki_data = fetch_wiki_summary(attraction_name)
    if not wiki_data:
        return {"error": f"No Wikipedia article found for '{attraction_name}'"}

    return {
        "name": attraction_name,
        "description": wiki_data.get("extract"),
        "wikipedia_url": wiki_data.get("url"),
        "image_url": wiki_data.get("image_url"),
        "wikidata_id": fetch_wikidata_id(attraction_name),
    }


@tool
def lookup_wikidata(attraction_name: str) -> dict:
    """Look up a Wikidata Q-ID for an attraction by Wikipedia title.

    Useful for cross-referencing with Wikivoyage and other Wikidata-backed systems.
    """
    if not attraction_name:
        return {}

    qid = fetch_wikidata_id(attraction_name)
    if not qid:
        return {"error": f"No Wikidata entry found for '{attraction_name}'"}

    return {
        "name": attraction_name,
        "wikidata_id": qid,
        "wikidata_url": f"https://www.wikidata.org/wiki/{qid}",
    }
