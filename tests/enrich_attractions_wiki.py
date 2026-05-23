"""Wikipedia + Wikivoyage enrichment for the attractions in out/paris_sample.json.

For each attraction we add:
  - wiki_description : one-line description (Wikipedia REST summary `description`)
  - wiki_url         : link to the Wikipedia article
  - wikidata_id      : Wikidata Q-id (used to match Wikivoyage listings)
  - wikivoyage_admission : admission/ticket price text harvested from Wikivoyage

Wikivoyage's `{{see}}` and `{{do}}` listings have a structured `price=` field —
purpose-built for travel-guide ticket info. Far more reliable than regex over
Wikipedia article text.

Run:
    tavenv/bin/python tests/enrich_attractions_wiki.py
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
JSON_PATH = OUT / "paris_sample.json"
MD_PATH = OUT / "paris_sample.md"

if not JSON_PATH.exists():
    print("Run tests/build_paris_sample.py first")
    sys.exit(1)

WP_API = "https://en.wikipedia.org/w/api.php"
WP_REST = "https://en.wikipedia.org/api/rest_v1"
WV_API = "https://en.wikivoyage.org/w/api.php"
UA = "navigation-ai/0.1 (educational; ido.rozin@darwingov.com)"
HEADERS = {"User-Agent": UA, "Accept": "application/json"}


# ---------------------------------------------------------------------------
# Wikipedia helpers
# ---------------------------------------------------------------------------
def wiki_search_title(name: str) -> str | None:
    r = requests.get(
        WP_API,
        params={"action": "opensearch", "search": name, "limit": 1, "format": "json"},
        headers=HEADERS,
        timeout=10,
    )
    r.raise_for_status()
    arr = r.json()
    return arr[1][0] if arr[1] else None


def wiki_summary(title: str) -> dict:
    r = requests.get(f"{WP_REST}/page/summary/{title}", headers=HEADERS, timeout=10)
    return r.json() if r.status_code == 200 else {}


def first_sentence(text: str) -> str:
    if not text:
        return ""
    m = re.search(r".+?[.!?](?=\s|$)", text)
    return (m.group(0) if m else text)[:200]


# ---------------------------------------------------------------------------
# Wikivoyage — build a global {Q-id: price} map across all Paris sub-pages
# ---------------------------------------------------------------------------
LISTING_RE = re.compile(r"\{\{(?:see|do)\s*\|(.*?)\n\}\}", re.DOTALL | re.IGNORECASE)


def kv_from_listing(body: str) -> dict:
    """Parse `name=foo | price=bar | wikidata=Q123` pairs.

    Wikivoyage listings put multiple `key=value` pairs on one line, separated
    by `|`. A value ends at the next `|` OR at end-of-line, whichever comes
    first. (Don't use a single `.*?` — same-line pipes wouldn't terminate.)
    """
    body = "|" + body
    pairs = re.findall(
        r"\|\s*([a-z_]+)\s*=\s*([^|\n]*)",
        body,
        flags=re.IGNORECASE,
    )
    return {k.lower(): v.strip() for k, v in pairs}


def fetch_wv(page: str) -> str:
    r = requests.get(
        WV_API,
        params={"action": "parse", "page": page, "prop": "wikitext",
                "format": "json", "redirects": 1},
        headers=HEADERS,
        timeout=20,
    )
    if r.status_code != 200:
        return ""
    parse = r.json().get("parse") or {}
    return (parse.get("wikitext") or {}).get("*", "")


def build_paris_price_map() -> dict[str, dict]:
    """Walk every per-arrondissement Wikivoyage sub-page and harvest priced listings."""
    main = fetch_wv("Paris")
    # Sub-page names look like `Paris/1st arrondissement`, `Paris/La Défense`, etc.
    sub_pages = sorted({
        m.group(1)
        for m in re.finditer(r"\[\[(Paris/[^#\]|]+)#Q\d+", main)
    })
    print(f"  Wikivoyage Paris has {len(sub_pages)} sub-pages")

    price_map: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(fetch_wv, p): p for p in sub_pages}
        for fut in as_completed(futures):
            page = futures[fut]
            wt = fut.result()
            n = 0
            for m in LISTING_RE.finditer(wt):
                kv = kv_from_listing(m.group(1))
                q = kv.get("wikidata")
                price = kv.get("price", "").strip()
                if q and price and re.search(r"\d", price):
                    price_map[q] = {
                        "price": price,
                        "name": kv.get("name"),
                        "source_page": page,
                    }
                    n += 1
            print(f"    {page:<40s} {n} priced listings")
    return price_map


# ---------------------------------------------------------------------------
# Enrich
# ---------------------------------------------------------------------------
def clip_price(s: str) -> str:
    # Wikivoyage prices can be long ("€12-16; under 18, free; EU-residents, 18-26 free; …"). Keep the first clause.
    s = re.split(r"[;.]", s, maxsplit=1)[0]
    s = re.sub(r"\s+", " ", s).strip()
    return s[:80]


sample = json.loads(JSON_PATH.read_text())
attractions = sample["attractions_in_paris"]

print("Building Wikivoyage price map…")
price_map = build_paris_price_map()
print(f"  -> {len(price_map)} priced listings indexed\n")

print(f"Enriching {len(attractions)} attractions…\n")
for a in attractions:
    name = a["name"]
    title = wiki_search_title(name) or name
    summary = wiki_summary(title)

    description = summary.get("description") or first_sentence(summary.get("extract", ""))
    q_id = summary.get("wikibase_item")  # e.g., "Q19675"
    wiki_url = ((summary.get("content_urls") or {}).get("desktop") or {}).get("page")

    a["wiki_title"] = summary.get("title") or title
    a["wiki_url"] = wiki_url
    a["wiki_description"] = description
    a["wikidata_id"] = q_id

    wv = price_map.get(q_id) if q_id else None
    if wv:
        a["wikivoyage_admission"] = clip_price(wv["price"])
        a["wikivoyage_source"] = wv["source_page"]
    else:
        a["wikivoyage_admission"] = None
        a["wikivoyage_source"] = None

    print(f"  · {name:<40s} {q_id or '-':<10s}  {description[:50]}")
    print(f"      admission: {a['wikivoyage_admission']}")
    time.sleep(0.1)


JSON_PATH.write_text(json.dumps(sample, indent=2, ensure_ascii=False))
print(f"\nUpdated {JSON_PATH}")

# Re-render the attractions section of the markdown.
md_lines = MD_PATH.read_text().splitlines()
start = next(i for i, ln in enumerate(md_lines) if ln.startswith("## Top tourist attractions"))
end = len(md_lines)
for i in range(start + 1, len(md_lines)):
    if md_lines[i].startswith("## ") or md_lines[i].startswith("_Note"):
        end = i
        break

new_section = [
    "## Top tourist attractions near Paris center  (Google Nearby + Wikipedia + Wikivoyage)",
    "",
    "| Name | ★ | Reviews | Description | Admission | Lat | Lng | Wiki |",
    "|---|---:|---:|---|---|---:|---:|---|",
]
for a in attractions:
    desc = (a.get("wiki_description") or "—").replace("|", "\\|")[:80]
    adm = (a.get("wikivoyage_admission") or "—").replace("|", "\\|")
    wiki = f"[wiki]({a['wiki_url']})" if a.get("wiki_url") else "—"
    new_section.append(
        f"| {a['name']} | {a['rating']} | {a['review_count']} | {desc} | {adm} | "
        f"{a['lat']:.4f} | {a['lng']:.4f} | {wiki} |"
    )
new_section.append("")
MD_PATH.write_text("\n".join(md_lines[:start] + new_section + md_lines[end:]))
print(f"Updated {MD_PATH}")
