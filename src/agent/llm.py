"""Construct chat models bound to travel-agent tools."""
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from tools import core_tools
from tools.rec_tools import rec_tools

_TOOL_SETS = {
    "travel": core_tools,
    "recommendation": rec_tools,
}

def get_models(provider: str = "google", mode: str = "travel") -> tuple[Runnable, BaseChatModel]:
    """Return (model_with_tools, extraction_model) for the chosen provider and mode.

    mode="travel"         — binds core planning tools
    mode="recommendation" — binds rec discovery tools (used for tool binding; rec path
                            uses Plan-and-Execute so the tools-bound model is not invoked
                            in the rec loop)
    """
    provider = provider.lower()
    bound_tools = _TOOL_SETS[mode]

    if provider == "groq":
        base = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    elif provider == "ollama":
        base = ChatOllama(model="gpt-oss:120b-cloud", temperature=0)
    elif provider == "openai":
        base = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        base = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    return base.bind_tools(bound_tools), base
