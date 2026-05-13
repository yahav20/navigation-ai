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

    mode="travel"         — binds core planning tools, no token cap
    mode="recommendation" — binds rec discovery tools, caps tokens at 1500 to prevent
                            runaway loops when tool results repeat across multiple calls
    """
    provider = provider.lower()
    bound_tools = _TOOL_SETS[mode]
    # Token cap only needed for the rec agent
    cap = 1500 if mode == "recommendation" else None

    if provider == "groq":
        base = ChatGroq(model="llama-3.1-8b-instant", temperature=0, **({"max_tokens": cap} if cap else {}))
    elif provider == "ollama":
        base = ChatOllama(model="gpt-oss:120b-cloud", temperature=0, **({"num_predict": cap} if cap else {}))
    elif provider == "openai":
        base = ChatOpenAI(model="gpt-4o-mini", temperature=0, **({"max_tokens": cap} if cap else {}))
    else:
        base = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, **({"max_output_tokens": cap} if cap else {}))

    return base.bind_tools(bound_tools), base
