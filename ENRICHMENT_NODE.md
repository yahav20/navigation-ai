# Enrichment Node — Feature Summary

## Purpose

Before this feature, the graph sent every user message directly to the AI agent without any validation. This caused two common problems:

1. **Missing information** — the agent would try to plan a trip without knowing the origin city, destination, or budget, often producing a poor or incomplete response.
2. **Too many options** — when the database contained many flights or hotels for a route, the agent had no guidance on which ones to highlight, leading to overwhelming or unfocused responses.

The **enrichment node** is a quality gate inserted into the graph before the agent runs. It ensures the agent only receives a request once that request is both complete and focused enough to produce a high-quality response.

---

## What It Does and How

The enrichment node runs through up to five sequential phases on every user message. If any phase fails its check, it asks the user a follow-up question and halts the graph for that turn. The user's answer is picked up on the next turn (conversation history is preserved via `MemorySaver`), and the node re-evaluates from the beginning.

### Phase 1 — Required fields check
Reads `current_city`, `destination_city`, and `total_budget` from the state (populated by the preceding `extract_metadata` node). If any field is missing, the node uses the LLM to ask the user for the missing information in a friendly, conversational way.

**Refusal handling:** If the node has previously asked for a field and the user still hasn't provided it, a single `RefusalDetection` LLM call checks whether the user is actively refusing:
- **Mandatory fields** (`current_city`, `destination_city`): refusal triggers a firm but polite message explaining that the information is required and offering the user the option to type `exit`.
- **Optional field** (`total_budget`): refusal sets `budget_optional = True` in state and the node continues without a budget constraint — the user is never asked again.

This check is pure Python logic — no database queries, no LLM reasoning beyond the single refusal-detection call.

### Phase 2 — Country destination check
Checks whether `destination_city` is actually a country name rather than a specific city. It queries the database for destination cities in that country that have available flights from the user's origin. If more than one city is found, the user is asked to choose. If exactly one city is found, the destination is silently corrected and the node continues.

### Phase 3 — Option count check
Queries the database to count real flight and hotel options for the route. If both counts are within `OPTION_THRESHOLD` (currently 2), the node passes the request through to the agent immediately.

### Phase 4 — Preference extraction
If the option count is too high, the node tries to extract filtering preferences the user may have already stated anywhere in the conversation history (e.g. "I prefer 4-star hotels"). This uses structured LLM output (`UserPreferences` Pydantic model) to pull out up to four fields: `min_hotel_stars`, `max_hotel_price_per_night`, `max_flight_price`, `preferred_airline`.

If preferences are found, they are applied as a filter against the fetched options:
- If filtered results are non-empty → preferences are saved to state and the node passes through.
- If filtered results are empty (the user's preferences match nothing) → the node passes through without any filtering, so the agent sees all options.

### Phase 5 — Enrichment question mini-agent
If no preferences have been stated yet, a small dedicated agent is invoked. This agent has access to two targeted tools:

- **`get_hotel_filter_options(city)`** — runs a `SELECT DISTINCT stars, MIN/MAX price` SQL query. Returns only the star ratings that exist and the price range, without fetching full hotel records.
- **`get_flight_filter_options(origin, destination)`** — runs a `SELECT DISTINCT airline, MIN/MAX price` SQL query. Returns only the airlines that operate the route and the price range.

The agent calls whichever tools are relevant, receives the dimension data back as tool messages, and then formulates one targeted question for the user. Because it sees the actual available values, it only asks about dimensions that have real variety — for example, if all hotels are 3-star, it will not ask about star rating.

### Graph routing
After the enrichment node runs, a conditional edge (`after_enrichment`) routes the graph:
- `enrichment_complete = True` → continues to the `agent` node
- `enrichment_complete = False` → routes to `END`, surfacing the question to the user

When the user answers, `MemorySaver` ensures the full conversation history (including the question and the answer) is available in the next turn, so `extract_metadata` can pick up new information without the user having to repeat anything.

---

## Code Changes

### `src/agent/state.py`
- Added `enrichment_complete: bool` — tracks whether the enrichment gate has been passed in the current turn.
- Added `user_preferences: dict` — stores filtering preferences extracted from the user (e.g. min star rating, max price).
- Added `enrichment_asked_fields: list` — tracks which fields have already been requested so refusal detection is only triggered on follow-up turns.
- Added `budget_optional: bool` — set to `True` when the user explicitly declines to provide a budget; prevents re-asking.

### `src/agent/models.py` *(new)*
- `TravelMetadata` — Pydantic model used by `extract_metadata` to pull origin, destination, and budget from conversation history.
- `UserPreferences` — Pydantic model with four optional fields: `min_hotel_stars`, `max_hotel_price_per_night`, `max_flight_price`, `preferred_airline`.
- `RefusalDetection` — Pydantic model with three boolean flags used to detect when the user is actively refusing to supply a required field.

### `src/tools/enrichment_tools.py` *(new)*
- `get_hotel_filter_options(city)` — `@tool`; calls `get_hotel_dimensions()` on the active provider.
- `get_flight_filter_options(origin, destination)` — `@tool`; calls `get_flight_dimensions()` on the active provider.
- `enrichment_tools` — list exported for model binding.
- `enrichment_tool_map` — name → function dict used by the mini-agent's manual tool loop.

### `src/agent/enrichment.py` *(new)*
- `OPTION_THRESHOLD = 2` — maximum number of flights or hotels before preferences are solicited.
- `_count_travel_options(origin, destination)` — fetches real flights and hotels from the active provider, filtering out "no results" messages.
- `_get_country_cities(destination, origin)` — checks if the destination is a country name and returns the available cities.
- `_apply_pref_filter(flights, hotels, prefs)` — filters fetched options against the user's stated preferences.
- `make_check_enrichment(extraction_model, enrichment_question_model)` — factory that returns the `check_enrichment` node, closing over the injected LLM instances. Contains the complete five-phase logic described above.

### `src/agent/edge.py`
- Added `after_enrichment(state)` — conditional edge function that returns `"agent"` when `enrichment_complete` is `True` and `END` otherwise.

### `src/agent/node.py`
- Imports `TravelMetadata` from `agent.models`, `make_check_enrichment` from `agent.enrichment`, and `enrichment_tools` from `tools.enrichment_tools`. All enrichment-specific code has been removed.
- `get_models(provider)` — factory returning `(model_with_tools, extraction_model)` for Google Gemini or Groq Llama.
- `extract_travel_data(state)` — utility used by `formatter` to parse tool-call results out of message history.
- `create_nodes(provider)` — builds `enrichment_question_model` by binding `enrichment_tools` to `extraction_model`, delegates enrichment node creation to `make_check_enrichment`, and defines `extract_metadata`, `call_model`, and `formatter` inline. Returns all four as a tuple.
- `call_model` system prompt now handles three budget states: a known number, `budget_optional=True` ("Not specified — no budget constraint"), and unknown.

### `src/main.py`
- Imported `MemorySaver` and `after_enrichment`.
- Added `"enrichment"` node to the graph builder.
- Changed the edge from `extract_metadata` to go to `"enrichment"` instead of directly to `"agent"`.
- Added `add_conditional_edges("enrichment", after_enrichment)`.
- Compiled the graph with `checkpointer=MemorySaver()` to persist conversation history across turns.

### `src/providers/base.py`
- Added three new abstract methods: `get_hotel_dimensions`, `get_flight_dimensions`, `get_cities_in_country`.

### `src/providers/sqlite_provider.py`
- Implemented `get_hotel_dimensions` — uses `SELECT DISTINCT stars, MIN/MAX price_per_night GROUP BY stars`.
- Implemented `get_flight_dimensions` — uses `SELECT DISTINCT airline, MIN/MAX price GROUP BY airline`.
- Implemented `get_cities_in_country` — joins `flights`, `cities`, and `countries` to return destination cities with available flights in a given country, optionally filtered by origin.

### `src/providers/json_provider.py`
- Implemented `get_hotel_dimensions` — processes the in-memory JSON data to return the same structure.
- Implemented `get_flight_dimensions` — same approach for flights.
- Implemented `get_cities_in_country` — returns an empty list (the JSON database has no country-level data).
- Updated `fetch_hotels` signature to accept `max_price: int = None` (to match the updated base class).

### `data/travel_db.json`
- Added three more Paris hotels (de Crillon 5★, Ibis 3★, Hotel du Louvre 4★) and two more TLV→Paris flights (Transavia, Wizz Air) to give the enrichment node enough options to trigger.
- Renamed all `"TLV"` origin entries to `"Tel Aviv"` to match what the LLM naturally extracts from user messages.
