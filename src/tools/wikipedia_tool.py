import json

import requests
from langchain_core.tools import tool


@tool
def get_wikipedia_summary(topic: str) -> str:
    """Fetch a short, factual overview of any travel-related topic from Wikipedia.

    Use this when the user asks about:
    - A specific city or country ("tell me about Kyoto", "what is Amsterdam known for?")
    - A specific attraction or landmark ("what is the Louvre?", "tell me about the Colosseum")
    - A historical site, museum, or natural wonder ("what is Machu Picchu?")

    Do NOT use for questions about flights, hotels, prices, or trip planning.
    """
    # Normalise spaces to underscores for the Wikipedia URL
    formatted = topic.strip().replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted}"

    try:
        response = requests.get(url, timeout=5, headers={"User-Agent": "navigation-ai/1.0"})

        if response.status_code == 200:
            data = response.json()
            return json.dumps({
                "topic": topic,
                "summary": data.get("extract", "No summary available."),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            })
        elif response.status_code == 404:
            return json.dumps({
                "error": f"No Wikipedia page found for '{topic}'. Try a more specific name or different spelling."
            })
        else:
            return json.dumps({"error": f"Wikipedia API returned status {response.status_code}"})

    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Request failed: {e}"})
