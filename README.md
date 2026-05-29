# Agentic Travel Planner - Atlas AI

A smart autonomous travel planning agent built with **Gemini 2.5 Flash** and **LangChain/LangGraph**, developed as part of the Software Engineering Project Workshop (83538-01) at Bar-Ilan University.

## Overview

This project moves beyond basic Prompt Engineering into full **Agentic AI architecture** — an autonomous system capable of:

- Planning and executing complex travel tasks
- Searching and retrieving data from external tools
- Performing self-reflection and self-correction
- Coordinating multiple specialized sub-agents

## Tech Stack

| Component | Technology |
|---|---|
| Language Model | Gemini 2.5 Flash (Google AI Studio) |
| Orchestration & State | LangChain & LangGraph |
| Search & Retrieval | Tavily API, SQLite |
| Language | Python |

## Setup

### 1. Get a Gemini API Key

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with a Gmail account
3. Click **Get API key** → **Create API key in a new project**
4. Make sure the quota tier is **Free tier**

Free tier limits:
- **RPM**: 10 requests/minute (sufficient for development)
- **RPD**: 250 requests/day
- **Context Window**: up to 250,000 tokens

### 2. Create a Virtual Environment

```bash
python -m venv taenv
.\taenv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install python-dotenv
pip install -U langchain-google-genai
pip install -q -U google-generativeai
pip install langgraph
```

### 4. Configure Environment Variables

Create a `.env` file in the project root (do **not** commit this file):

```
GOOGLE_API_KEY=your_key_here
```

### 5. Test the Connection

Run `test_connection.py` to verify everything works:

```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

try:
    print("--- Testing Connection ---")
    response = llm.invoke("Say 'System Online' if you can hear me.")
    print(f"Success! Response: {response.content}")
except Exception as e:
    print(f"--- Connection Failed ---\nError details: {e}")
```

## Web UI — connect [agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui)

The graph is exposed as a LangGraph server via `langgraph.json` (graph id `agent`,
factory `src/agent/graph.py:make_graph`), so the official `agent-chat-ui` can drive it.

The dev server needs `langgraph-cli`, whose backend (`langgraph-api`) is best kept in
**its own virtualenv** — installing it alongside the main app would otherwise force
`protobuf>=6` and clash with the Gemini SDK (`protobuf<6`). Setup (uses [uv](https://docs.astral.sh/uv/)):

```bash
uv venv .lgenv --python 3.12
uv pip install --python .lgenv/bin/python -r requirements-server.txt
.lgenv/bin/langgraph dev --allow-blocking          # http://127.0.0.1:2024
```

`--allow-blocking` is required because graph nodes read local files (e.g. `data/travel_db.json`)
synchronously. Then point the UI at it:

```bash
npx create-agent-chat-app          # or clone agent-chat-ui and `npm run dev`
# Deployment URL: http://127.0.0.1:2024
# Assistant / Graph ID: agent
```

The CLI (`python src/main.py`) is unchanged and keeps using its own `SqliteSaver`.

> **Security note:** input validation (`validate_input`) runs *inside* the graph via
> `security_gate`, so it protects the UI path too. But `scan_output` (output secret-scanning)
> and the per-session turn cap (`MAX_TURNS_PER_SESSION`) live only in the CLI loop
> (`src/main.py`) and are **not** enforced over the web UI. Move them into the graph (e.g. a
> post-`summary` scan node) if the UI is exposed beyond local dev.

## Project Structure

```
navigation-ai/
├── .env                  # API keys (not committed)
├── tools.py              # Agent tools (flights, hotels, cost calculation)
├── travel_db.json        # Local travel database
├── test_connection.py    # Connection test script
└── README.md
```

## Tools (The Toolbox)

The agent uses three core tools defined in `tools.py` with the `@tool` decorator. Each tool must have a clear English docstring so the LLM knows when and how to use it.

| Tool | Input | Output |
|---|---|---|
| `fetch_flights` | origin city, destination city | List of flights with prices and times (JSON) |
| `fetch_hotels` | destination city | List of hotels with availability and ratings |
| `calculate_trip_cost` | flight price, hotel price/night, number of nights | Total cost + 10% service fee |

### Tool Guidelines
- **Data**: Tools read from `travel_db.json` — make sure the file is properly structured.
- **Empty results**: Return a clear plain-text message (e.g., `"No available hotels found in Paris for the given criteria"`) rather than an empty list, so the agent can reason about it.
- **Docstrings**: Must be in English and accurately describe the tool's behavior.

## Resources
- [LangChain Documentation](https://docs.langchain.com/)
- Antonio Gullí, *Agentic Design Pattern - A Hands-On Guide to Building Intelligent Systems*, Springer 2025
