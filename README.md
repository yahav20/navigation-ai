# Atlas AI — Agentic Travel Planner

An autonomous multi-agent travel planner built with **LangChain / LangGraph**, developed for the Software Engineering Project Workshop (83538-01) at Bar-Ilan University.

The agent plans complete trips end-to-end: it routes each request to specialized sub-agents (flight search, day-by-day itinerary planning, travel advisor, destination exploration), calls external tools (flights, hotels, attractions, restaurants, live events), and self-corrects with critic/replanner loops.

## Tech Stack

| Component | Technology |
|---|---|
| Language model | OpenAI `gpt-4o-mini` / `gpt-5.4-mini` (default) — Groq, Gemini and Ollama also supported |
| Orchestration & state | LangChain & LangGraph (multi-node graph with checkpointing) |
| Data & retrieval | SQLite (`data/travel_agency.db`), Travelpayouts, Google Maps, Xotelo, Wikipedia, Tavily |
| Web UI | LangGraph dev server + `atlas-web` (Next.js chat UI, vendored from [agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui)) |
| Language | Python 3.12, TypeScript |

## Quick Start

### Prerequisites

- **Python 3.12**
- **Node.js 18+** (for the web UI)
- A `.env` file in the project root (see below)

### 1. Configure environment variables

Copy `.env.example` to `.env` in the project root and fill in the keys
(if you were given a ready `.env` file, just place it in the project root):

```bash
cp .env.example .env
```

| Variable | Required? | Purpose |
|---|---|---|
| `MODEL_PROVIDER` | yes | Which LLM to use: `openai` (default), `groq`, `google`, or `ollama` |
| `OPENAI_API_KEY` | yes (with `openai`) | API key for the chosen provider |
| `GOOGLE_MAPS_API_KEY` | yes | Attractions, restaurants, geocoding |
| `TAVILY_API_KEY` | yes | Concert / live-event search |
| `TRAVELPAYOUTS_API_KEY` | no | Live flight prices — without it, flight search falls back to the bundled SQLite database |

### 2. Install Python dependencies

macOS / Linux:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-server.txt
```

Windows:

```bat
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements-server.txt
```

> pip may print a `protobuf` dependency-conflict warning during install — it is
> harmless (the Gemini SDK pins `protobuf<6`, the LangGraph dev server wants
> `>=6`; the default OpenAI provider is unaffected).

### 3. Start the backend (LangGraph dev server)

macOS / Linux:

```bash
.venv/bin/langgraph dev --allow-blocking
```

Windows:

```bat
.venv\Scripts\langgraph dev --allow-blocking
```

This serves the graph at `http://127.0.0.1:2024`. `--allow-blocking` is required
because graph nodes read local files (e.g. the SQLite database) synchronously.

### 4. Start the web UI

In a **second terminal**:

```bash
cd atlas-web
npm install
npm run dev
```

Open **http://localhost:3000** — `atlas-web/.env` (committed, no secrets) already
points the UI at the local server (`NEXT_PUBLIC_API_URL=http://localhost:2024`,
assistant id `agent`), so it connects on load. Type a request like
*"Plan a 4-day trip from Tel Aviv to Rome in October for 2 adults"*.

### Alternative: run the CLI (no web UI needed)

macOS / Linux:

```bash
.venv/bin/python src/main.py
```

Windows:

```bat
.venv\Scripts\python src\main.py
```

An interactive terminal chat with the same agent graph. Sessions are checkpointed
to `data/checkpoints.db` and can be resumed with `--session <name>`.

## Running Tests

```bash
.venv/bin/pip install pytest
.venv/bin/pytest -m unit        # fast: pure Python, no LLM or network calls
.venv/bin/pytest -m integration # requires a valid LLM API key in .env
```

## Project Structure

```
navigation-ai/
├── .env.example          # Template for required API keys (copy to .env)
├── langgraph.json        # LangGraph server config (graph id: "agent")
├── requirements.txt      # Core app dependencies
├── requirements-server.txt  # Core deps + LangGraph dev server
├── data/
│   ├── travel_agency.db  # SQLite: cities, flights, hotels, activities + API caches
│   └── init_db.py        # Script that builds/seeds the database
├── src/
│   ├── main.py           # CLI entry point
│   ├── security.py       # Input validation, output scanning, audit log
│   ├── config/           # Provider selection (MODEL_PROVIDER) & session naming
│   ├── agent/
│   │   ├── core/         # Main graph: security gate → router → sub-agents
│   │   ├── itinerary/    # Day-by-day planner/executor/critic/replanner loop
│   │   ├── advisor/      # Travel-advisor sub-agent (plan → execute → replan)
│   │   ├── explore/      # Destination-exploration graph
│   │   ├── travel/       # Trip curation agent
│   │   └── shared/       # Shared nodes and utilities
│   ├── providers/        # Data access: SQLite DAL + Travelpayouts / Google Maps /
│   │                     #   Xotelo / Wikipedia APIs with SQLite caching & fallback
│   └── tools/            # LangChain @tool wrappers (flights, hotels, activities,
│                         #   concerts, maps, cost calculator, ...)
├── tests/                # pytest suite (unit + integration markers)
└── atlas-web/            # Next.js chat UI, pre-configured for the local server
```

## How It Works

1. **Security gate** — every message passes `validate_input` (prompt-injection guard) before reaching the graph.
2. **Router** — an LLM router classifies the request (trip planning, itinerary, advice, exploration, out-of-scope) and dispatches to the matching sub-graph.
3. **Sub-agents** — e.g. the itinerary pipeline runs *plan → execute → critic → replan* until the schedule passes its checks; multi-destination trips are split into legs and recombined.
4. **Tools & data** — providers try live APIs first (Travelpayouts, Google Maps, Xotelo), cache results into SQLite, and fall back to the bundled database when an API or key is unavailable.
5. **Checkpointing** — conversations are checkpointed (SQLite for the CLI, in-memory for the dev server) so multi-turn refinements keep full context.

> **Security note:** input validation runs *inside* the graph (`security_gate`), so it
> protects the web UI path too. Output secret-scanning and the per-session turn cap live
> only in the CLI loop (`src/main.py`) and are not enforced over the web UI.

## Resources

- [LangChain Documentation](https://docs.langchain.com/)
- Antonio Gullí, *Agentic Design Patterns — A Hands-On Guide to Building Intelligent Systems*, Springer 2025
