"""Build a SQLite DB of countries and cities from public GitHub CSVs.

Sources:
  - country-codes.csv (datasets/country-codes)
  - world_cities.csv  (joelacus/world-cities)

Schema:
  countries(id, alpha2 UNIQUE, alpha3 UNIQUE, numeric UNIQUE, name, region, subregion)
  cities(id, country_id FK, name, lat, lng)

Cities whose ISO alpha-2 code is not in the country-codes dataset are skipped
and reported (e.g. XK / Kosovo, which has no official ISO 3166-1 assignment).
"""

import csv
import io
import os
import sqlite3
import sys
import urllib.request

COUNTRY_CODES_URL = "https://raw.githubusercontent.com/datasets/country-codes/main/data/country-codes.csv"
WORLD_CITIES_URL = "https://raw.githubusercontent.com/joelacus/world-cities/main/world_cities.csv"
DB_PATH = "cities.db"

SCHEMA = """
CREATE TABLE countries (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    alpha2    TEXT NOT NULL UNIQUE,
    alpha3    TEXT NOT NULL UNIQUE,
    numeric   TEXT NOT NULL UNIQUE,
    name      TEXT NOT NULL,
    region    TEXT,
    subregion TEXT
);

CREATE TABLE cities (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    country_id INTEGER NOT NULL REFERENCES countries(id),
    name       TEXT NOT NULL,
    lat        REAL NOT NULL,
    lng        REAL NOT NULL
);

CREATE INDEX idx_cities_country_id ON cities(country_id);
"""


def fetch_csv(url: str) -> list[dict]:
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def main() -> int:
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    print("fetching country codes...")
    countries = fetch_csv(COUNTRY_CODES_URL)
    print("fetching world cities...")
    cities = fetch_csv(WORLD_CITIES_URL)

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    country_rows = [
        (
            row["ISO3166-1-Alpha-2"],
            row["ISO3166-1-Alpha-3"],
            row["ISO3166-1-numeric"],
            row.get("CLDR display name") or row.get("official_name_en") or "",
            row.get("Region Name") or None,
            row.get("Sub-region Name") or None,
        )
        for row in countries
        if row.get("ISO3166-1-Alpha-2") and row.get("ISO3166-1-Alpha-3") and row.get("ISO3166-1-numeric")
    ]
    conn.executemany(
        "INSERT INTO countries (alpha2, alpha3, numeric, name, region, subregion) VALUES (?, ?, ?, ?, ?, ?)",
        country_rows,
    )

    alpha2_to_id = {alpha2: cid for cid, alpha2 in conn.execute("SELECT id, alpha2 FROM countries")}

    inserted = 0
    skipped: dict[str, int] = {}
    city_rows = []
    for city in cities:
        cid = alpha2_to_id.get(city["country"])
        if cid is None:
            skipped[city["country"]] = skipped.get(city["country"], 0) + 1
            continue
        city_rows.append((cid, city["name"], float(city["lat"]), float(city["lng"])))
    conn.executemany(
        "INSERT INTO cities (country_id, name, lat, lng) VALUES (?, ?, ?, ?)",
        city_rows,
    )
    inserted = len(city_rows)

    conn.commit()
    conn.close()

    print(f"wrote {DB_PATH}")
    print(f"  countries: {len(country_rows)}")
    print(f"  cities:    {inserted}")
    if skipped:
        print(f"  skipped:   {sum(skipped.values())} cities with codes {sorted(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
