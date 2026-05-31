"""Wikipedia and Wikivoyage enrichment helpers."""
from __future__ import annotations

import requests

_WIKIPEDIA_BASE = "https://en.wikipedia.org/api/rest_v1"
_WIKIDATA_BASE = "https://www.wikidata.org/w/api.php"
_TIMEOUT = 10

_wiki_cache: dict[str, dict] = {}
_wikidata_cache: dict[str, str] = {}


def fetch_wiki_summary(title: str) -> dict | None:
    """Fetch a Wikipedia summary for an article.

    Args:
        title: Wikipedia article title (e.g., "Louvre")

    Returns: Dict with extract, url, image_url, or None on error
    """
    if not title:
        return None

    key = title.lower().strip()
    if key in _wiki_cache:
        return _wiki_cache[key]

    try:
        r = requests.get(
            f"{_WIKIPEDIA_BASE}/page/summary/{title}",
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json() or {}
    except (requests.RequestException, ValueError):
        return None

    if data.get("type") == "not-found":
        return None

    result = {
        "title": data.get("title"),
        "extract": data.get("extract"),
        "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
        "image_url": data.get("thumbnail", {}).get("source"),
    }

    _wiki_cache[key] = result
    return result


def fetch_wikidata_id(wikipedia_title: str) -> str | None:
    """Resolve a Wikipedia article title to a Wikidata Q-ID.

    Args:
        wikipedia_title: Wikipedia article title

    Returns: Wikidata ID (e.g., "Q7270") or None
    """
    if not wikipedia_title:
        return None

    key = wikipedia_title.lower().strip()
    if key in _wikidata_cache:
        return _wikidata_cache[key]

    try:
        r = requests.get(
            _WIKIDATA_BASE,
            params={
                "action": "query",
                "titles": wikipedia_title,
                "prop": "pageprops",
                "format": "json",
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json() or {}
    except (requests.RequestException, ValueError):
        return None

    pages = data.get("query", {}).get("pages") or {}
    for page in pages.values():
        qid = page.get("pageprops", {}).get("wikibase_item")
        if qid:
            _wikidata_cache[key] = qid
            return qid

    return None


def fetch_wikidata_info(qid: str) -> dict | None:
    """Fetch full Wikidata entity info.

    Args:
        qid: Wikidata Q-ID (e.g., "Q7270")

    Returns: Dict with label, description, image, or None
    """
    if not qid:
        return None

    try:
        r = requests.get(
            f"{_WIKIDATA_BASE}",
            params={
                "action": "wbgetentities",
                "ids": qid,
                "format": "json",
                "languages": "en",
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json() or {}
    except (requests.RequestException, ValueError):
        return None

    entities = data.get("entities") or {}
    entity = entities.get(qid) or {}

    labels = entity.get("labels") or {}
    descriptions = entity.get("descriptions") or {}
    claims = entity.get("claims") or {}

    image_url = None
    image_props = claims.get("P18") or []
    if image_props and len(image_props) > 0:
        image_url = image_props[0].get("mainsnak", {}).get("datavalue", {}).get("value")

    return {
        "qid": qid,
        "label": labels.get("en", {}).get("value"),
        "description": descriptions.get("en", {}).get("value"),
        "image_url": image_url,
    }
