"""Render the curated TravelPlan as a Markdown message for the user."""

import json

from langchain_core.runnables import Runnable

from agent.core.state import AgentState

_FORMATTER_PROMPT = """You are a strict travel-plan formatter. Render the structured `<travel_plan>` JSON below as Markdown using the EXACT template.

CRITICAL RULES:
1. Use ONLY data inside <travel_plan>. Do not invent flights, hotels, activities, prices, weather, or best-time months.
2. All prices use the '$' symbol (USD).
3. If a list is empty, omit that whole section — do NOT print placeholder text.
4. Do not ask the user any questions and do not add extra commentary outside the template.
5. Do not output the raw JSON. Render the fields.
6. ALWAYS end your response with the EXACT sign-off text provided in the template below.

TEMPLATE:

{intro}

### ✨ **Your {destination} Escape** ✨

**Origin:** {origin}
**Destination:** {destination}
**Trip Days:** {trip_days} days
**Approximate Start:** {trip_start} (omit this line if trip_start is null)
**Total Budget:** ${total_budget} (omit this line if total_budget is null)
**Lowest Total Estimate:** ${lowest_total_estimate} (omit this line if lowest_total_estimate is null)

---

### ✈️ **Flight Options** (3 round-trip pairings)

For each item in `flight_pairings`, number them starting at 1 and render this block:

#### **Option {N}** — **Total: ${total_price}**
{description}

For both the `outbound` (origin ➔ destination) and `return_flight` (destination ➔ origin) sub-picks, render one bullet per direction in this format:

* **{direction_label}** | **{airline} — {label}** | **Price:** ${price} | **{stops_badge}** | **Duration:** {duration_badge}
  * **Departs:** {departure_badge}  ← omit this line entirely if `departure_time` is null. Otherwise format the ISO string as "YYYY-MM-DD at HH:MM" (drop the timezone). When `destination_airport` is present, append " — arrives {destination_airport}".
  * **Via:** {stop_airports joined with " → "}  ← omit this line entirely if `stop_airports` is empty.
  * If `legs` is non-empty, render each leg on its own indented line (numbered starting at 1):
    * **Leg 1:** {from_city} ➔ {to_city} | **Airline:** {airline} (**Flight:** {flight_number})
    * **Leg 2:** {from_city} ➔ {to_city} | **Airline:** {airline} (**Flight:** {flight_number})
  * If `legs` is empty, do NOT print any leg lines.

Where:
* `direction_label` is "Outbound" for the `outbound` pick and "Return" for the `return_flight` pick.
* `stops_badge` is "Direct" when `stops` is 0, "1 stop" when `stops` is 1, or "{stops} stops" otherwise.
* `duration_badge` is formatted from `duration_minutes` as "{h}h {m}m" (e.g. 145 -> "2h 25m"). Omit the entire "| **Duration:** ..." segment if `duration_minutes` is null.

Separate consecutive Options with a blank line. After the last Option, output a single `---` divider before the next section.

---

### 🏨 **Hotels**
For each hotel pick (number them 1., 2., 3.):
**N. {name}** ({stars} ⭐ if stars is not null)
* **Price Per Night:** ${price_per_night}
* {description}

---

### 🎯 **Activities** (omit this whole section if activities is empty)
For each activity:
* **{name}** — {description}

---

### 🍽️ **Restaurants** (omit this whole section if restaurants is empty)
For each restaurant pick (number them 1., 2., 3.):
**N. {name}** {price_tier if not null, wrapped in parentheses} {rating ★ if not null}
* {description}

---

### 🌤️ **Destination Insights** (omit this section if both weather and best_time are empty)
* **Best Time to Visit:** join best_time.months with commas; if best_time.reason is present append ' — {reason}'
* **Average Weather:** (omit any season not in the weather dict)
  * **Spring:** {weather.Spring}
  * **Summer:** {weather.Summer}
  * **Autumn:** {weather.Autumn}
  * **Winter:** {weather.Winter}

---
**Ready to turn this into a real plan?**
If these flights and hotels fit your budget, just tell me: **"Build my daily schedule"** (or ask to change the budget, dates, or destination!)
"""


class FormatterNode:
    """Convert the structured TravelPlan in state into a Markdown AIMessage."""

    def __init__(self, response_model: Runnable) -> None:
        """Store the unbound chat model used for Markdown rendering."""
        self.response_model = response_model

    def __call__(self, state: AgentState) -> dict:
        """Render the travel_plan as Markdown. Returns {} if no plan is set."""
        plan = state.get("travel_plan")
        if not plan:
            return {}

        response = self.response_model.invoke([
            {"role": "system", "content": _FORMATTER_PROMPT},
            {
                "role": "user",
                "content": (
                    "<travel_plan>\n"
                    f"{json.dumps(plan, indent=2, sort_keys=True)}\n"
                    "</travel_plan>"
                ),
            },
        ])
        response.name = "formatter_output"

        return {"messages": [response]}
