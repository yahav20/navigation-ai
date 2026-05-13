"""
Migration v2 — adds city_tags, recommended_duration, and flights.duration_hours.

Safe to re-run: each step checks whether the change already exists before applying it.

Run from the project root:  python data/migrate_v2.py
"""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "travel_agency.db")


def resolve_city(conn: sqlite3.Connection, name: str, alpha2: str) -> int:
    row = conn.execute(
        """SELECT cities.id FROM cities
             JOIN countries ON cities.country_id = countries.id
            WHERE LOWER(cities.name) = LOWER(?) AND countries.alpha2 = ?
            LIMIT 1""",
        (name, alpha2),
    ).fetchone()
    if row is None:
        raise ValueError(f"city not found: {name!r} ({alpha2})")
    return row[0]


def add_duration_hours(conn: sqlite3.Connection) -> None:
    existing = [c[1] for c in conn.execute("PRAGMA table_info(flights)").fetchall()]
    if "duration_hours" in existing:
        print("  flights.duration_hours already exists — skipping.")
        return

    conn.execute("ALTER TABLE flights ADD COLUMN duration_hours REAL")

    # Actual flight durations in hours (door-to-door block time, rounded to nearest 0.5h)
    durations = {
        "LY321":  4.5,   # Tel Aviv → Paris (El Al)
        "AF123":  4.5,   # Tel Aviv → Paris (Air France)
        "BA164":  5.0,   # Tel Aviv → London (British Airways)
        "VS100":  5.0,   # Tel Aviv → London (Virgin Atlantic)
        "LY091": 11.5,   # Tel Aviv → Tokyo (El Al direct)
        "EK312": 13.0,   # Tel Aviv → Tokyo (Emirates via Dubai)
        "UA445": 11.0,   # Tel Aviv → New York (United)
        "LH909":  3.5,   # Tel Aviv → Berlin (Lufthansa)
        "FR101":  3.5,   # Tel Aviv → Berlin (Ryanair)
        "KL456":  4.5,   # Tel Aviv → Amsterdam (KLM)
        "AF124":  1.25,  # London → Paris (Air France)
        "JL402": 12.0,   # London → Tokyo (JAL)
        "VS001":  7.5,   # London → New York (Virgin Atlantic)
        "VS002":  7.0,   # New York → London (Virgin Atlantic, tailwind)
        "AF200":  7.5,   # New York → Paris (Air France)
    }

    for flight_number, hours in durations.items():
        conn.execute(
            "UPDATE flights SET duration_hours = ? WHERE flight_number = ?",
            (hours, flight_number),
        )

    updated = conn.execute(
        "SELECT COUNT(*) FROM flights WHERE duration_hours IS NOT NULL"
    ).fetchone()[0]
    print(f"  flights.duration_hours added and populated ({updated} rows).")


def add_city_tags(conn: sqlite3.Connection) -> None:
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    if "city_tags" in tables:
        print("  city_tags already exists — skipping.")
        return

    conn.executescript("""
        CREATE TABLE city_tags (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            city_id INTEGER NOT NULL REFERENCES cities(id),
            tag     TEXT    NOT NULL,
            UNIQUE(city_id, tag)
        );
        CREATE INDEX idx_city_tags_city ON city_tags(city_id);
        CREATE INDEX idx_city_tags_tag  ON city_tags(tag);
    """)

    ids = {
        "tel_aviv":  resolve_city(conn, "Tel Aviv",      "IL"),
        "paris":     resolve_city(conn, "Paris",         "FR"),
        "london":    resolve_city(conn, "London",        "GB"),
        "tokyo":     resolve_city(conn, "Tokyo",         "JP"),
        "new_york":  resolve_city(conn, "New York City", "US"),
        "berlin":    resolve_city(conn, "Berlin",        "DE"),
        "amsterdam": resolve_city(conn, "Amsterdam",     "NL"),
    }

    # (city_key, tag)
    tags = [
        # Tel Aviv
        ("tel_aviv", "beach"),
        ("tel_aviv", "mediterranean"),
        ("tel_aviv", "nightlife"),
        ("tel_aviv", "foodie"),
        ("tel_aviv", "sunny"),
        ("tel_aviv", "modern"),
        ("tel_aviv", "city-break"),

        # Paris
        ("paris", "romantic"),
        ("paris", "city-break"),
        ("paris", "cultural"),
        ("paris", "luxury"),
        ("paris", "foodie"),
        ("paris", "historic"),
        ("paris", "fashion"),
        ("paris", "walkable"),

        # London
        ("london", "city-break"),
        ("london", "cultural"),
        ("london", "historic"),
        ("london", "multicultural"),
        ("london", "theatre"),
        ("london", "foodie"),
        ("london", "rainy"),

        # Tokyo
        ("tokyo", "modern"),
        ("tokyo", "foodie"),
        ("tokyo", "unique"),
        ("tokyo", "technology"),
        ("tokyo", "historic"),
        ("tokyo", "safe"),
        ("tokyo", "city-break"),
        ("tokyo", "nature-nearby"),

        # New York City
        ("new_york", "city-break"),
        ("new_york", "shopping"),
        ("new_york", "entertainment"),
        ("new_york", "iconic"),
        ("new_york", "multicultural"),
        ("new_york", "foodie"),
        ("new_york", "expensive"),

        # Berlin
        ("berlin", "nightlife"),
        ("berlin", "budget-friendly"),
        ("berlin", "cultural"),
        ("berlin", "historic"),
        ("berlin", "alternative"),
        ("berlin", "art"),
        ("berlin", "city-break"),
        ("berlin", "student-friendly"),

        # Amsterdam
        ("amsterdam", "cycling"),
        ("amsterdam", "canals"),
        ("amsterdam", "romantic"),
        ("amsterdam", "city-break"),
        ("amsterdam", "cultural"),
        ("amsterdam", "historic"),
        ("amsterdam", "liberal"),
        ("amsterdam", "walkable"),
    ]

    conn.executemany(
        "INSERT OR IGNORE INTO city_tags (city_id, tag) VALUES (?, ?)",
        [(ids[c], t) for c, t in tags],
    )
    print(f"  city_tags created and seeded ({len(tags)} rows).")


def add_recommended_duration(conn: sqlite3.Connection) -> None:
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    if "recommended_duration" in tables:
        print("  recommended_duration already exists — skipping.")
        return

    conn.executescript("""
        CREATE TABLE recommended_duration (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            city_id  INTEGER NOT NULL UNIQUE REFERENCES cities(id),
            min_days INTEGER NOT NULL,
            max_days INTEGER NOT NULL,
            notes    TEXT    NOT NULL
        );
    """)

    ids = {
        "tel_aviv":  resolve_city(conn, "Tel Aviv",      "IL"),
        "paris":     resolve_city(conn, "Paris",         "FR"),
        "london":    resolve_city(conn, "London",        "GB"),
        "tokyo":     resolve_city(conn, "Tokyo",         "JP"),
        "new_york":  resolve_city(conn, "New York City", "US"),
        "berlin":    resolve_city(conn, "Berlin",        "DE"),
        "amsterdam": resolve_city(conn, "Amsterdam",     "NL"),
    }

    # (city_key, min_days, max_days, notes)
    durations = [
        ("tel_aviv",  3, 5,
         "3 days covers the beach, Jaffa Old City, and the main food markets. "
         "Add 2 more days for day trips to Jerusalem or the Dead Sea."),

        ("paris",     3, 7,
         "3 days for the iconic sights (Eiffel Tower, Louvre, Notre-Dame). "
         "5–7 days lets you explore neighborhoods like Montmartre and take a day trip to Versailles."),

        ("london",    4, 7,
         "4 days covers the top landmarks and a few museums. "
         "7 days gives you time for day trips to Oxford, Stonehenge, or the Cotswolds."),

        ("tokyo",     5, 10,
         "5 days for the main districts (Shibuya, Shinjuku, Asakusa) and a day trip to Mt. Fuji. "
         "8–10 days to also explore Kyoto, Osaka, or Hakone."),

        ("new_york",  5, 10,
         "5 days to cover Manhattan highlights and Brooklyn. "
         "7–10 days to explore all five boroughs and do a day trip to the Hamptons or Philadelphia."),

        ("berlin",    3, 5,
         "3 days is plenty for the main historical and cultural sites. "
         "5 days lets you enjoy the nightlife scene and explore surrounding neighborhoods at a relaxed pace."),

        ("amsterdam", 2, 4,
         "2–3 days is ideal for this compact city — you can walk or cycle almost everywhere. "
         "4 days lets you do a day trip to Keukenhof (tulip season) or Delft."),
    ]

    conn.executemany(
        "INSERT INTO recommended_duration (city_id, min_days, max_days, notes) VALUES (?, ?, ?, ?)",
        [(ids[c], mn, mx, n) for c, mn, mx, n in durations],
    )
    print(f"  recommended_duration created and seeded ({len(durations)} rows).")


def run() -> None:
    if not os.path.exists(DB_PATH):
        print(f"ERROR: database not found at {DB_PATH}. Run data/init_db.py first.")
        raise SystemExit(1)

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        print("Applying migration v2...")
        add_duration_hours(conn)
        add_city_tags(conn)
        add_recommended_duration(conn)
        conn.commit()
        print("Migration v2 complete.")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
