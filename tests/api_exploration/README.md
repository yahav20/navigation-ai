# API exploration scripts

Scratchpad for testing third-party data sources and deciding which of the
SQLite-backed tools in `src/tools/` can be replaced with live data. Each
script is runnable on its own from the repo root and reads secrets from
`.env`.

```bash
tavenv/bin/python tests/api_exploration/<script>.py
```

Output files are written to `out/` at the project root.

---

## Probes — what does this API actually return?

### `test_travelpayouts.py`
Probes the Travelpayouts (Aviasales) flight + Hotellook endpoints.
Verifies that flight endpoints (`aviasales/v3/prices_for_dates`,
`v2/prices/latest`, autocomplete) work, and confirms that every documented
Hotellook hotel endpoint now returns 404. Ends with a replacement matrix
mapping each existing tool to whether Travelpayouts can back it.
Requires `TRAVELPAYOUTS_API_KEY` in `.env`.

### `test_google_maps.py`
Probes Google Maps Platform (legacy endpoints): Geocoding, Places Text
Search, Nearby Search, Place Details, Place Photo, Distance Matrix, and
a Text Search for hotels. Prints a replacement matrix at the end.
Requires `GOOGLE_MAPS_API_KEY` in `.env`.

### `probe_flights.py`
Deep-dive on the Travelpayouts flight API. Hits eight different
endpoints (`prices_for_dates`, `grouped_prices`, `get_latest_prices`,
`v2/prices/latest`, `v2/prices/month-matrix`, `v1/prices/cheap`,
`v1/prices/calendar`, `v1/prices/direct`) with several date scopes
(exact day vs. whole month) and `direct=true/false` variations, then
runs per-airport queries against CDG / ORY / BVA. Decodes airline IATA
codes via `data/en/airlines.json` so "IZ" is shown as "Arkia". Used to
diagnose why exact-day queries return so few rows and pick the best
combination for the build script.

---

## Builders — produce the example output

### `build_paris_sample.py`
Pulls a complete travel-planning sample for a TLV → Paris trip on
2026-06-10 and writes:

- `out/paris_sample.json` — structured data
- `out/paris_sample.md`   — readable markdown tables

Sections:

| Section | Source | Notes |
|---|---|---|
| Flights | Travelpayouts Aviasales v3 | Direct + cheapest connecting per Paris airport + monthly cheapest-per-day calendar |
| Hotels  | Xotelo (TripAdvisor, no auth) | `/list` for ratings + USD price range + lat/lng. `/rates` for live OTA breakdown is **opt-in** (`with_rates=True`) because it stalls 8–10s per hotel when TripAdvisor has no cache |
| Attractions | Google Places (Nearby) + Wikipedia + Wikivoyage | Calls `enrich_attractions()` from `enrich_attractions_wiki.py` to add one-line descriptions and admission prices |
| Restaurants | Google Places API **v1** | Uses new `places:searchText` endpoint for the `priceRange` field (actual EUR amounts), filters out `lodging` types |

Requires `TRAVELPAYOUTS_API_KEY` and `GOOGLE_MAPS_API_KEY` in `.env`.
Typical runtime ~12s (Wikivoyage page fetches dominate).

---

## Enrichers — add fields to an existing sample

### `enrich_attractions_wiki.py`
For each attraction in `out/paris_sample.json`, adds:
- `wiki_description` — one-liner from Wikipedia REST summary
- `wiki_url`         — link to the Wikipedia article
- `wikidata_id`      — Wikidata Q-id (used to match Wikivoyage listings)
- `wikivoyage_admission` — admission price string parsed from
  Wikivoyage `{{see}}` / `{{do}}` listings on each Paris arrondissement
  sub-page

`enrich_attractions(attractions)` and `attractions_markdown_section(attractions)`
are reusable helpers — `build_paris_sample.py` imports and calls them
directly, so running this script standalone is no longer required. CLI
mode (reads the JSON, mutates, writes back, patches the markdown) still
works.

### `enrich_restaurants.py`
Place Details follow-up over each restaurant in `out/paris_sample.json`.
For every row with `price_level == None` from the initial Nearby Search,
it issues a Place Details call with `fields=price_level,website,…` to
see if Details returns more data. Documents the finding (Google strips
`price_level` for places typed as `lodging`, so this pass mostly comes
up empty). Kept around as a record of the experiment — `build_paris_sample.py`
no longer needs it because it now uses the new Places API v1
(`priceRange`) for restaurants.

---

## Output layout

Both files live at the project root:

```
out/
├── paris_sample.json   # structured — feed to a router/planner
└── paris_sample.md     # readable summary — open in any markdown viewer
```
