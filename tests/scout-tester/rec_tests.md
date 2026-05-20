# Recommendation Agent Test Suite

Tests for the Plan-and-Execute recommendation agent (`rec_planner → rec_executor → rec_formatter`).

Each test specifies:
- **Input** — the user message(s)
- **Expected plan** — which tools the planner should select and with what args
- **Expected behavior** — what the formatted response must contain or do
- **Pass criteria** — specific, checkable outcomes
- **Status** — T (pass) / F (fail) / ? (not yet run)

---

## Section A — Single-Turn, Single-Tool
*Tests planner precision: the right one tool is selected, no over-calling.*

---

### A1 — Best time to visit (specific city)
**Input:** `When is the best time to visit Tokyo?`
**Expected plan:** `get_best_time_to_visit(city="Tokyo")` — 1 tool
**Expected behavior:** Response names the recommended months and the reason (weather / festivals).
**Pass criteria:**
- Exactly 1 tool in plan
- Response includes months and a reason
- No invented landmarks or activities
- `rec_shown_cities` contains "Tokyo"
**Status:** T

---

### A2 — Trip duration for a single city
**Input:** `How many days should I spend in London?`
**Expected plan:** `get_trip_duration_recommendation(city="London")` — 1 tool
**Expected behavior:** Response gives a min–max day range with an explanation of what each covers.
**Pass criteria:**
- Exactly 1 tool in plan
- Response includes a day range (e.g. "5–7 days")
- `rec_shown_cities` contains "London"
**Status:** T

---

### A3 — Weather in a specific season
**Input:** `What's the weather like in Paris in summer?`
**Expected plan:** `get_average_weather(city="Paris", season="Summer")` — 1 tool
**Expected behavior:** Response gives a temperature figure for summer in Paris.
**Pass criteria:**
- Exactly 1 tool in plan; season arg is exactly "Summer"
- Response mentions temperature
- No other seasons hallucinated
**Status:** T

---

### A4 — Vibe/lifestyle tag search, no origin
**Input:** `I want a romantic destination — where should I go?`
**Expected plan:** `find_destinations_by_tag(tag="romantic")` — 1 tool
**Expected behavior:** Lists only cities returned by the tool. Ends with an invitation to check flights once origin is known.
**Pass criteria:**
- Exactly 1 tool; tag="romantic"
- `get_reachable_destinations` NOT called (no origin given)
- All mentioned cities appear in `rec_shown_cities`
- Response ends with a question about the user's origin or a closer
**Status:** T

---

### A5 — Activity-category search, no origin
**Input:** `Where should I go for a nature trip?`
**Expected plan:** `find_destinations_by_vibe(category="Nature")` — 1 tool
**Expected behavior:** Lists only cities returned by the tool.
**Pass criteria:**
- Exactly 1 tool; category="Nature"
- `find_destinations_by_tag` NOT called
- Cities listed match tool output exactly
**Status:** T

---

### A6 — Activities for a named city the user is already visiting
**Input:** `I'm going to Tokyo — what activities are there?`
**Expected plan:** `fetch_activities(city="Tokyo")` — 1 tool
**Expected behavior:** Lists activities by category. Does not suggest alternative destinations.
**Pass criteria:**
- Exactly 1 tool; no destination-discovery tools called
- Response is activity-focused, not destination-recommendation-focused
- `rec_shown_cities` contains "Tokyo"
**Status:** T

---

### A7 — Budget + origin, no trip duration
**Input:** `I'm flying from Tel Aviv and my total budget is $800. Where can I go?`
**Expected plan:** `find_destinations_within_budget_auto(origin="Tel Aviv", total_budget=800)` — 1 tool
**Expected behavior:** Lists affordable destinations with flight + hotel cost breakdown. Mentions recommended stay duration per city (from tool).
**Pass criteria:**
- Exactly 1 tool; `find_destinations_within_budget` NOT used (no days given)
- `get_reachable_destinations` NOT called alongside the budget tool
- Response includes cost figures
- All cities in `rec_shown_cities`
**Status:** T

---

### A8 — Budget + origin + explicit days
**Input:** `I'm in Tel Aviv with a budget of $800 for exactly 7 days. Where can I realistically fly?`
**Expected plan:** `find_destinations_within_budget(origin="Tel Aviv", total_budget=800, trip_days=7)` — 1 tool
**Expected behavior:** Only destinations whose `cheapest_flight + cheapest_hotel × 7` fits $800 are mentioned.
**Pass criteria:**
- Exactly 1 tool; `find_destinations_within_budget_auto` NOT used
- trip_days arg is 7
- No city presented that exceeds budget
**Status:** T

---

### A9 — Short-haul reachability, no budget
**Input:** `I'm flying from Tel Aviv. I hate long flights — where should I go?`
**Expected plan:** `get_reachable_destinations(origin="Tel Aviv", max_flight_hours=2.5)` — 1 tool
**Expected behavior:** Lists only cities reachable within ~2–3 hours from Tel Aviv with prices and flight duration.
**Pass criteria:**
- Exactly 1 tool; max_flight_hours is ~2.5 (short-haul threshold)
- Budget tool NOT called
- Only tool-returned cities mentioned
**Status:** T

---

## Section B — Single-Turn, Multi-Tool
*Tests planner strategy: the right combination of tools is selected, and they remain within the 3-tool cap.*

---

### B1 — Full city profile (overview + activities)
**Input:** `Tell me everything about Berlin — what kind of city is it, what can I do there, and when should I visit?`
**Expected plan:** `get_city_overview(city="Berlin")` + `fetch_activities(city="Berlin")` — 2 tools
**Expected behavior:** Covers activity types, best visit months, weather, and a named activity list from the DB.
**Pass criteria:**
- Exactly 2 tools; both target "Berlin"
- No invented attractions (only tool-returned ones)
- `rec_shown_cities` contains "Berlin"
**Status:** T

---

### B2 — Origin + vibe, no budget
**Input:** `I'm flying from Tel Aviv and I want a foodie destination. Where should I go?`
**Expected plan:** `get_reachable_destinations(origin="Tel Aviv")` + `find_destinations_by_tag(tag="foodie")` — 2 tools
**Expected behavior:** Leads with cities that appear in BOTH results (reachable AND foodie-tagged). Other reachable cities listed separately.
**Pass criteria:**
- Exactly 2 tools
- Budget tool NOT called
- Cross-referencing visible in response (reachable + matching tag highlighted first)
- All mentioned cities in `rec_shown_cities`
**Status:** T

---

### B3 — Nightlife search (tag + vibe)
**Input:** `I want a city with great nightlife — any recommendations?`
**Expected plan:** `find_destinations_by_tag(tag="nightlife")` + `find_destinations_by_vibe(category="Nightlife")` — 2 tools
**Expected behavior:** Combines results from both tools; cities appearing in both are highlighted.
**Pass criteria:**
- Both tools called with the nightlife/Nightlife arg
- No origin-based filtering (no origin given)
- All mentioned cities in `rec_shown_cities`
**Status:** T

---

### B4 — Multi-city duration split
**Input:** `I want to visit both Rome and Amsterdam. How should I split my time?`
**Expected plan:** `get_trip_duration_recommendation(city="Rome")` + `get_trip_duration_recommendation(city="Amsterdam")` — 2 tools
**Expected behavior:** Gives a day range for each city and a suggested split.
**Pass criteria:**
- Exactly 2 tools, one per city
- Response suggests specific day ranges for both
- `rec_shown_cities` contains both "Rome" and "Amsterdam"
**Status:** T

---

### B5 — Specific city: when to go + weather
**Input:** `I'm thinking of going to Tokyo. What's the weather like in summer and when's the best time to visit?`
**Expected plan:** `get_city_overview(city="Tokyo")` — 1 tool (covers both questions in one call), OR `get_best_time_to_visit(city="Tokyo")` + `get_average_weather(city="Tokyo", season="Summer")` — 2 tools
**Expected behavior:** Answers both the seasonal weather and the best-time question.
**Pass criteria:**
- At most 2 tools; no destination-discovery tools called
- Both questions answered from tool data only
- `rec_shown_cities` contains "Tokyo"
**Status:** T

---

### B6 — Family trip with specific request
**Input:** `I'm planning a family trip with kids aged 7 and 10. Where should we go and what activities would suit them?`
**Expected plan:** `find_destinations_by_vibe(category="Family")` + optionally `find_destinations_by_tag(tag="cultural")` or similar — 2 tools max
**Expected behavior:** Lists family-friendly destinations with relevant activities (no nightlife, no adult-only venues).
**Pass criteria:**
- At least 1 tool is find_destinations_by_vibe(category="Family")
- `fetch_activities` NOT called (user didn't name a specific city)
- Response filters out adult-oriented activities
- All mentioned cities in `rec_shown_cities`
**Status:** T

---

### B7 — Budget + vibe (budget tool only — no vibe tool)
**Input:** `I'm from Tel Aviv, I have $1000, and I want a romantic city. Where can I go?`
**Expected plan:** `find_destinations_within_budget_auto(origin="Tel Aviv", total_budget=1000)` — 1 tool only
**Expected behavior:** Lists affordable destinations; highlights which of those match a romantic vibe based on tool data. Does NOT call `find_destinations_by_tag` separately.
**Pass criteria:**
- Exactly 1 tool (budget already filters by reachability — adding a tag tool would introduce out-of-budget cities)
- Response notes which affordable cities have a romantic feel IF that appears in the data
- No city suggested that wasn't returned by the budget tool
**Status:** T

---

## Section C — Multi-Turn
*Tests follow-up detection, state accumulation, and context carry-over.*

---

### C1 — Follow-up: detail on a previously shown city
**Turn 1:** `I want a romantic destination — where should I go?`
**Turn 2:** `Tell me more about Paris — when should I visit and how many days?`

**Expected Turn 1 plan:** `find_destinations_by_tag(tag="romantic")`
**Expected Turn 2 plan:** `get_city_overview(city="Paris")` — classified as follow-up; `rec_shown_cities` from Turn 1 provides context
**Pass criteria:**
- Turn 2 is detected as a follow-up (summary passed to planner)
- Turn 2 plan does NOT re-call `find_destinations_by_tag`
- `rec_shown_cities` after Turn 2 contains cities from both turns
- Turn 2 response answers only the Paris question
**Status:** ?

---

### C2 — Fresh question after a previous turn (no context bleed)
**Turn 1:** `Tell me about Tokyo — what's the weather like in summer?`
**Turn 2:** `What are the best family destinations in Europe?`

**Expected Turn 1 plan:** `get_average_weather(city="Tokyo", season="Summer")`
**Expected Turn 2 plan:** `find_destinations_by_vibe(category="Family")` — classified as fresh; Tokyo context NOT carried into Turn 2
**Pass criteria:**
- Turn 2 detected as fresh (is_followup=False)
- Turn 2 response makes no reference to Tokyo
- `find_destinations_by_vibe` called without any Tokyo filter
- `rec_shown_cities` accumulates cities from both turns
**Status:** ?

---

### C3 — Refining a budget within an active rec session
**Turn 1:** `I'm from Tel Aviv. I want a romantic city — where should I go?`
**Turn 2:** `My budget is $900. Which of those could I afford?`

**Expected Turn 1 plan:** `get_reachable_destinations(origin="Tel Aviv")` + `find_destinations_by_tag(tag="romantic")`
**Expected Turn 2 plan:** `find_destinations_within_budget_auto(origin="Tel Aviv", total_budget=900)` — follow-up; origin carried from state; budget tool used since budget is now known
**Pass criteria:**
- Turn 2 is_followup=True
- Planner picks up origin="Tel Aviv" from state context (not re-stated by user)
- Budget tool used, not reachability tool again
- Response only presents cities within $900
**Status:** ?

---

### C4 — Avoid re-fetching already seen data
**Turn 1:** `What's the weather like in Amsterdam in summer?`
**Turn 2:** `And what about Amsterdam in winter?`

**Expected Turn 1 plan:** `get_average_weather(city="Amsterdam", season="Summer")`
**Expected Turn 2 plan:** `get_average_weather(city="Amsterdam", season="Winter")` — follow-up; planner sees `rec_last_tool_results` showing summer already fetched; does NOT re-fetch summer
**Pass criteria:**
- Turn 2 plan has exactly 1 tool with season="Winter"
- Turn 2 response answers winter only (no duplication of summer data)
- `rec_shown_cities` contains "Amsterdam" after both turns
**Status:** ?

---

### C5 — Transition from rec to planning (rec_shown_cities handoff)
**Turn 1:** `I'm from Tel Aviv with $800. Where can I go?`
**Turn 2:** `Sounds great, let's plan a trip to [first city returned]!`

**Expected Turn 1 plan:** `find_destinations_within_budget_auto(origin="Tel Aviv", total_budget=800)`
**Expected Turn 2 routing:** Router classifies as `new_travel_plan` (transition intent); `rec_shown_cities` from Turn 1 is available so the planning path can resolve the city reference
**Pass criteria:**
- Turn 2 intent = "new_travel_plan" (not "recommendations")
- Planning path activated (enrichment node, not rec_planner)
- `rec_shown_cities` persists so destination can be resolved
**Status:** ?

---

## Section D — Edge Cases
*Tests fallback, country resolution, no-origin guard, and no-hallucination.*

---

### D1 — Fallback: tag with no DB match
**Input:** `I want to go somewhere known for its jungles — where should I go?`
**Expected plan:** `find_destinations_by_tag(tag="nature-nearby")` or similar — if result is empty, executor falls back to `find_destinations_by_vibe(category="Sightseeing")`
**Pass criteria:**
- Fallback fires if the original tag returns empty
- Response is transparent ("no exact match — here are some general options")
- Fallback cities appear in `rec_shown_cities`
- No invented jungle destinations
**Status:** ?

---

### D2 — Country resolution: origin
**Input:** `I'm from Israel. I want somewhere with great food — where should I go?`
**Expected plan:** `get_reachable_destinations(origin="Tel Aviv")` + `find_destinations_by_tag(tag="foodie")` — "Israel" resolved to "Tel Aviv" in tool args
**Pass criteria:**
- Tool args use "Tel Aviv", not "Israel"
- No error from the provider
- Results reflect actual flights from Tel Aviv
**Status:** T

---

### D3 — No origin guard: no reachability tool without an origin
**Input:** `I want a beach holiday — where should I fly?`
**Expected plan:** `find_destinations_by_tag(tag="beach")` — 1 tool only; `get_reachable_destinations` NOT called
**Expected behavior:** Lists beach cities; ends with an offer to check flights once the user shares their origin.
**Pass criteria:**
- `get_reachable_destinations` NOT in plan
- No "no flights" language (user hasn't shared origin)
- Response invites user to share origin
- `rec_shown_cities` populated with beach cities
**Status:** T

---

### D4 — No-hallucination: only tool-returned cities in response
**Input:** `I'm flying from New York. What are my options?`
**Expected plan:** `get_reachable_destinations(origin="New York City")` — 1 tool
**Pass criteria:**
- Every city mentioned in response appears in the tool's result set
- No city added from training knowledge ("Amsterdam is also worth considering...")
- If tool returns 0 results, response says so honestly
**Status:** T

---

### D5 — No budget tool without an origin
**Input:** `I have a $500 budget — where can I go?`
**Expected plan:** `find_destinations_by_tag(tag="budget-friendly")` — 1 tool; budget tools NOT used (no origin given)
**Expected behavior:** Lists budget-friendly cities by tag; notes that costs depend on origin and offers to check once origin is shared.
**Pass criteria:**
- `find_destinations_within_budget_auto` NOT called (no origin)
- `find_destinations_within_budget` NOT called
- Response does not present specific flight/hotel prices (no origin means no real cost data)
**Status:** ?

---

### D6 — rec_last_tool_results prevents redundant re-fetch
**Turn 1:** `What can I do in Amsterdam?`
**Turn 2:** `What about nightlife there specifically?`

**Expected Turn 1 plan:** `fetch_activities(city="Amsterdam")`
**Expected Turn 2 plan:** *(no new tool call needed — activities already in rec_last_tool_results)* — planner should note activities were already fetched and filter the existing data rather than re-fetching
**Pass criteria:**
- Turn 2 plan either has 0 tool calls (uses cached data) OR calls a targeted tool like `find_destinations_by_tag(tag="nightlife")` for cross-reference — but does NOT re-call `fetch_activities("Amsterdam")`
- Turn 2 response is narrowed to nightlife activities from the previous result
**Status:** ?

---

## Summary Table

| ID  | Description                                | Tools | Multi-turn | Status |
|-----|--------------------------------------------|-------|------------|--------|
| A1  | Best time to visit Tokyo                   | 1     | N          | T      |
| A2  | Trip duration for London                   | 1     | N          | T      |
| A3  | Weather in Paris in summer                 | 1     | N          | T      |
| A4  | Romantic destination, no origin            | 1     | N          | T      |
| A5  | Nature trip, no origin                     | 1     | N          | T      |
| A6  | Activities in Tokyo (named city)           | 1     | N          | T      |
| A7  | Budget + origin, no days                   | 1     | N          | T      |
| A8  | Budget + origin + 7 days                   | 1     | N          | T      |
| A9  | Short-haul from Tel Aviv                   | 1     | N          | T      |
| B1  | Full Berlin profile                        | 2     | N          | T      |
| B2  | Tel Aviv origin + foodie vibe              | 2     | N          | T      |
| B3  | Nightlife (tag + vibe)                     | 2     | N          | T      |
| B4  | Rome + Amsterdam duration split            | 2     | N          | T      |
| B5  | Tokyo weather + best time                  | 1–2   | N          | T      |
| B6  | Family trip with activities                | 2     | N          | T      |
| B7  | Budget + romantic (budget tool only)       | 1     | N          | T      |
| C1  | Follow-up: detail on shown city            | 1     | Y          | ?      |
| C2  | Fresh question, no context bleed           | 1     | Y          | ?      |
| C3  | Budget refinement in active rec session    | 1     | Y          | ?      |
| C4  | Avoid re-fetching same city+season         | 1     | Y          | ?      |
| C5  | Transition from rec to planning            | —     | Y          | ?      |
| D1  | Fallback: tag with no DB match             | 1+fb  | N          | ?      |
| D2  | Country resolution (Israel -> Tel Aviv)    | 2     | N          | T      |
| D3  | No reachability tool without origin        | 1     | N          | T      |
| D4  | No-hallucination: NYC options              | 1     | N          | T      |
| D5  | Budget stated, no origin                   | 1     | N          | ?      |
| D6  | rec_last_tool_results prevents re-fetch    | 0–1   | Y          | ?      |
