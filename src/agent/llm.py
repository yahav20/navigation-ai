"""Construct chat models for travel and recommendation paths."""
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from tools.rec_tools import rec_tools


def get_models(provider: str = "google", mode: str = "travel") -> tuple[Runnable, BaseChatModel]:
    """Return (response_or_agent_model, extraction_model) for provider/mode.

    mode="travel"         — returns an unbound response model, no token cap
    mode="recommendation" — binds rec discovery tools, caps tokens at 1500 to prevent
                            runaway loops when tool results repeat across multiple calls
    """
    provider = provider.lower()
    if mode not in {"travel", "recommendation"}:
        msg = f"Unknown model mode: {mode}"
        raise ValueError(msg)

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

    if mode == "recommendation":
        return base.bind_tools(rec_tools), base

    return base, base
