"""Construct chat models for travel and recommendation paths."""
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from tools import core_tools
from tools.advisor_tools import advisor_tools

_TOOL_SETS = {
    "travel": core_tools,
    "advisor": advisor_tools,
}

def get_models(provider: str = "google", mode: str = "travel") -> tuple[Runnable, BaseChatModel]:
    """Return (response_or_agent_model, extraction_model) for provider/mode.

    mode="travel"   — returns an unbound response model, no tool binding
    mode="advisor"  — binds advisor discovery tools; used by general chat node and
                      for tool schema reference (advisor path itself uses Plan-and-Execute)
    """
    provider = provider.lower()
    if mode not in _TOOL_SETS:
        msg = f"Unknown model mode: {mode!r}"
        raise ValueError(msg)

    if provider == "groq":
        base = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    elif provider == "ollama":
        base = ChatOllama(model="gpt-oss:120b-cloud", temperature=0)
    elif provider == "openai":
        base = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        base = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    if mode == "advisor":
        return base.bind_tools(advisor_tools), base

    return base, base
