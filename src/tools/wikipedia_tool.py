import requests
import json
from langchain_core.tools import tool

@tool
def get_city_wikipedia_summary(city: str) -> str:
    """Fetch a short, factual overview of a city or destination from Wikipedia.
    Use this tool when the user asks for general information, history, or background about a specific place.
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{city}"
    
    try:
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return json.dumps({
                "city": city,
                "summary": data.get("extract", "No summary available.")
            })
        elif response.status_code == 404:
            return json.dumps({"error": f"Could not find a Wikipedia page for '{city}'. Try a different spelling."})
        else:
            return json.dumps({"error": f"Wikipedia API returned status code {response.status_code}"})
            
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"API request failed: {str(e)}"})
