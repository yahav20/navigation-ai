"""Construct chat models for travel and recommendation paths."""
import os

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from tools import core_tools
from tools.destinations import advisor_tools

_TOOL_SETS = {
    "travel": core_tools,
    "advisor": advisor_tools,
}

# Tag recognized by LangGraph's `messages`/`messages-tuple` stream mode: any LLM
# run carrying it is skipped, so its tokens never reach the chat UI. Use it for
# internal reasoning/extraction calls (structured JSON, rolling summaries,
# ReAct scratchpads). Only the user-facing formatter and chat nodes should
# stream — everything else would otherwise leak raw JSON into the GUI.
#
# The exact string MUST be "nostream" — that is langgraph.constants.TAG_NOSTREAM,
# checked in langgraph/pregel/_messages.py. (The "langsmith:" prefix belongs to
# TAG_HIDDEN = "langsmith:hidden", a different tag — do not confuse the two.)
NO_STREAM_TAG = "nostream"


def silent(runnable: Runnable) -> Runnable:
    """Wrap an LLM/runnable so its tokens are NOT streamed to the client UI."""
    return runnable.with_config(tags=[NO_STREAM_TAG])

def get_models(provider: str = "google", mode: str = "travel") -> tuple[Runnable, BaseChatModel]:
    """Return (response_or_agent_model, extraction_model) for provider/mode.

    mode="travel"   — returns an unbound response model, no tool binding
    mode="advisor"  — binds advisor discovery tools for tool schema reference
                      (the advisor path itself uses Plan-and-Execute, not tool binding)
    """
    provider = provider.lower()
    if mode not in _TOOL_SETS:
        msg = f"Unknown model mode: {mode!r}"
        raise ValueError(msg)

    if provider == "groq":
        # max_retries=6: the free Groq tier caps tokens-per-minute at 12k, so bursts
        # of graph nodes hit 429s with a retry-after of a few seconds. The SDK's
        # default 2 retries gets exhausted before the minute window frees up.
        base = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0, max_retries=6)
    elif provider == "ollama":
        base = ChatOllama(model="gpt-oss:120b-cloud", temperature=0)
    elif provider == "openai":
        model = "gpt-5.4-mini" if mode == "advisor" else "gpt-4o-mini"
        base = ChatOpenAI(model=model, temperature=0)
    else:
        base = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    if mode == "advisor":
        return base.bind_tools(advisor_tools), base

    return base, base


def get_generation_model(provider: str = "google") -> BaseChatModel:
    """Return the chat model for the user-facing generation nodes (travel_agent,
    formatter).

    On OpenAI these two nodes use gpt-5.4-mini, which benchmarked ~2x faster than
    gpt-4o-mini at equal output quality for the curation/formatting prompts
    (~12s -> ~6.5s median on the slim travel payload). Other providers reuse
    their default response model.
    """
    if provider.lower() == "openai":
        return ChatOpenAI(model="gpt-5.4-mini", temperature=0)
    response_model, _ = get_models(provider)
    return response_model
