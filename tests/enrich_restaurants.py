"""Place Details pass over the restaurants in out/paris_sample.json.

For every restaurant with `price_level == None`, call the Place Details endpoint
to see whether Details returns a price_level (Text Search / Nearby Search
sometimes omit it even when Google has the data).

Writes the enriched data back to:
    out/paris_sample.json     (restaurants list updated in-place)
    out/paris_sample.md       (restaurants table re-rendered)

Run:
    tavenv/bin/python tests/enrich_restaurants.py
"""
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

GM_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not GM_KEY:
    print("Missing GOOGLE_MAPS_API_KEY")
    sys.exit(1)

OUT_DIR = ROOT / "out"
JSON_PATH = OUT_DIR / "paris_sample.json"
MD_PATH = OUT_DIR / "paris_sample.md"

if not JSON_PATH.exists():
    print(f"Run tests/build_paris_sample.py first — {JSON_PATH} missing")
    sys.exit(1)

PRICE_LEVEL = {
    0: "Free",
    1: "$ – inexpensive",
    2: "$$ – moderate",
    3: "$$$ – expensive",
    4: "$$$$ – very expensive",
}

sample = json.loads(JSON_PATH.read_text())
restaurants = sample["restaurants_in_paris"]

DETAIL_FIELDS = "price_level,website,formatted_phone_number,editorial_summary"


def details(place_id: str) -> dict:
    r = requests.get(
        "https://maps.googleapis.com/maps/api/place/details/json",
        params={"place_id": place_id, "fields": DETAIL_FIELDS, "key": GM_KEY},
        timeout=15,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("status") != "OK":
        return {"_status": body.get("status"), "_error": body.get("error_message")}
    return body.get("result", {})


print(f"Enriching {len(restaurants)} restaurants…\n")
stats = {"filled": 0, "still_null": 0, "already_had": 0, "errors": 0}

for r in restaurants:
    name = r["name"]
    if r.get("price_level") is not None:
        stats["already_had"] += 1
        print(f"  ✓ already had tier   {r['price_tier']:<26s} {name}")
        continue

    try:
        d = details(r["place_id"])
    except requests.RequestException as e:
        stats["errors"] += 1
        print(f"  ! request error                          {name}: {e}")
        continue

    if "_status" in d:
        stats["errors"] += 1
        print(f"  ! API status {d['_status']:<10s}              {name}")
        continue

    pl = d.get("price_level")
    r["website"] = d.get("website")
    r["phone"] = d.get("formatted_phone_number")
    summary = (d.get("editorial_summary") or {}).get("overview")
    if summary:
        r["editorial_summary"] = summary

    if pl is not None:
        r["price_level"] = pl
        r["price_tier"] = PRICE_LEVEL.get(pl)
        stats["filled"] += 1
        print(f"  + filled from Details {r['price_tier']:<26s} {name}")
    else:
        stats["still_null"] += 1
        print(f"  · no price in Details either              {name}")

    time.sleep(0.05)  # be polite

print(f"\nSummary: {stats}")

# Re-write JSON
JSON_PATH.write_text(json.dumps(sample, indent=2, ensure_ascii=False))
print(f"Updated {JSON_PATH}")

# Regenerate the markdown table for restaurants only
md = MD_PATH.read_text()
lines = md.splitlines()

# Locate the restaurants table — from its heading to next blank/heading
start = next(i for i, ln in enumerate(lines) if ln.startswith("## Top-reviewed restaurants"))
# find end: the next "##" or "_Note_" or EOF
end = len(lines)
for i in range(start + 1, len(lines)):
    if lines[i].startswith("## ") or lines[i].startswith("_Note"):
        end = i
        break

new_section = [
    "## Top-reviewed restaurants near Paris center  (Nearby Search + Place Details)",
    "",
    "| Name | Rating ★ | Reviews | Price tier | Lat | Lng | Map |",
    "|---|---:|---:|---|---:|---:|---|",
]
for r in restaurants:
    new_section.append(
        f"| {r['name']} | {r['rating']} | {r['review_count']} | "
        f"{r['price_tier'] or '—'} | "
        f"{r['lat']:.4f} | {r['lng']:.4f} | "
        f"[map]({r['google_maps_url']}) |"
    )
new_section.append("")  # trailing blank before next section

MD_PATH.write_text("\n".join(lines[:start] + new_section + lines[end:]))
print(f"Updated {MD_PATH}")
