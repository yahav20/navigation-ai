"""Model factory for the recommendation agent — imports providers lazily to avoid missing-package errors."""
from langchain_core.language_models import BaseChatModel

from recommendation.tools.rec_tools import rec_tools


def get_rec_models(provider: str = "openai") -> tuple[BaseChatModel, BaseChatModel]:
    """Return (model_with_tools, extraction_model) for the recommendation agent.

    max_tokens is capped to prevent runaway generation loops that have been observed
    with smaller models when the same data appears in tool results multiple times.
    """
    provider = provider.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        base = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=1500)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        base = ChatOllama(model="gpt-oss:120b-cloud", temperature=0, num_predict=1500)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        base = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1500)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        base = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_output_tokens=1500)

    model_with_tools = base.bind_tools(rec_tools)
    return model_with_tools, base
