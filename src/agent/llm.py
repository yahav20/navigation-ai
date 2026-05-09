from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from tools import core_tools as tools

def get_models(provider: str = "google"):
    """
    Returns a tuple of (model_with_tools, extraction_model) based on the chosen provider.
    """
    if provider.lower() == "groq":
        print(" Initializing Groq (Llama 3)...")
        model = ChatGroq(model="llama-3.1-8b-instant", temperature=0).bind_tools(tools)
        extraction_model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    elif provider.lower() == "ollama":
        print(f" Initializing Ollama (local)...")
        model = ChatOllama(model="gpt-oss:120b-cloud", temperature=0).bind_tools(tools)
        extraction_model = ChatOllama(model="gpt-oss:120b-cloud", temperature=0)
    else:
        print(" Initializing Google (Gemini 2.5 Flash)...")
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools)
        extraction_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    return model, extraction_model