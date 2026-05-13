"""Construct chat models bound to travel-agent tools."""
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from tools import core_tools as tools
from tools.rec_tools import rec_tools


def get_models(provider: str = "google") -> tuple[Runnable, BaseChatModel]:
    """Return a tuple of (model_with_tools, extraction_model) for the chosen provider."""
    if provider.lower() == "groq":
        model = ChatGroq(model="llama-3.1-8b-instant", temperature=0).bind_tools(tools)
        extraction_model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    elif provider.lower() == "ollama":
        model = ChatOllama(model="gpt-oss:120b-cloud", temperature=0).bind_tools(tools)
        extraction_model = ChatOllama(model="gpt-oss:120b-cloud", temperature=0)
    elif provider.lower() == "openai":
        model = ChatOpenAI(model="gpt-5.4-mini", temperature=0).bind_tools(tools)
        extraction_model = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
    else:
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools)
        extraction_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    return model, extraction_model


def get_rec_models(provider: str = "openai") -> tuple[Runnable, BaseChatModel]:
    """Return (model_with_tools, extraction_model) for the recommendation agent.

    max_tokens is capped to prevent runaway generation loops observed when tool
    results repeat the same data across multiple calls.
    """
    provider = provider.lower()

    if provider == "groq":
        base = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=1500)
    elif provider == "ollama":
        base = ChatOllama(model="gpt-oss:120b-cloud", temperature=0, num_predict=1500)
    elif provider == "google":
        base = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_output_tokens=1500)
    else:
        base = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1500)

    model_with_tools = base.bind_tools(rec_tools)
    return model_with_tools, base
