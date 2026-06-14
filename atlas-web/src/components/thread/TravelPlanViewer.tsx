/**
 * components/thread/TravelPlanViewer.tsx
 *
 * Generative-UI component rendered inside LoadExternalComponent.
 * Receives props from the backend via the "TravelPlanViewer" UI message.
 *
 * Sections:
 *   1. Hero       — destination, route, dates, budget, travelers
 *   2. Flights    — 3 option cards, cheapest highlighted as "Best Value"
 *   3. Hotels     — horizontal-scroll cards with visual stars
 *   4. Activities & Restaurants — two-column pill grid
 *   5. Insights   — best-time banner + seasonal weather chips
 *   6. CTA        — "Build my daily schedule" → submits to the stream
 */

import { type FC, type CSSProperties } from "react";
import { v4 as uuidv4 } from "uuid";
import { useStreamContext } from "@/providers/Stream";

// ─── Types ────────────────────────────────────────────────────────────────────

interface FlightLeg {
  airline?: string;
  label?: string;
  price?: number;
  stops?: number | null;
  duration_minutes?: number | null;
  departure_time?: string;
  destination_airport?: string;
}

interface FlightPairing {
  total_price: number;
  description?: string;
  outbound: FlightLeg;
  return_flight: FlightLeg;
}

interface Hotel {
  name: string;
  stars?: number | null;
  price_per_night: number;
  description?: string;
}

interface Activity {
  name: string;
  description?: string;
}

interface Restaurant {
  name: string;
  price_tier?: string;
  rating?: number | null;
  description?: string;
}

interface TravelPlanViewerProps {
  destination?: string;
  origin?: string;
  trip_days?: number | null;
  trip_start?: string;
  total_budget?: number | null;
  travelers_label?: string;
  lowest_group_estimate?: number | null;
  flight_pairings?: FlightPairing[];
  hotels?: Hotel[];
  activities?: Activity[];
  restaurants?: Restaurant[];
  weather?: Record<string, string>;
  best_time?: { months?: string[]; reason?: string };
}

// ─── Pure helpers ─────────────────────────────────────────────────────────────

function fmtDuration(minutes: number | null | undefined): string {
  if (!minutes) return "";
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return h ? `${h}h ${m.toString().padStart(2, "0")}m` : `${m}m`;
}

function fmtIso(iso: string | undefined): string {
  if (!iso) return "";
  try {
    if (iso.includes("T")) {
      const [date, time] = iso.split("T");
      const [, month, day] = date.split("-");
      const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"];
      return `${parseInt(day)} ${MONTHS[parseInt(month) - 1]}, ${time.slice(0, 5)}`;
    }
    return iso.slice(0, 5);
  } catch {
    return iso;
  }
}

function stopsLabel(stops: number | null | undefined): string {
  if (stops == null) return "";
  if (stops === 0) return "Direct";
  return stops === 1 ? "1 stop" : `${stops} stops`;
}

// ─── Stars ────────────────────────────────────────────────────────────────────

const Stars: FC<{ count?: number | null }> = ({ count }) => {
  if (!count) return null;
  const n = Math.round(count);
  return (
    <span style={{ color: "#f5a623", fontSize: 13, letterSpacing: 1 }}>
      {"★".repeat(n)}{"☆".repeat(Math.max(0, 5 - n))}
    </span>
  );
};

// ─── 1. Hero ──────────────────────────────────────────────────────────────────

const Hero: FC<{
  destination: string;
  origin: string;
  trip_days?: number | null;
  trip_start?: string;
  total_budget?: number | null;
  travelers_label?: string;
  lowest_group_estimate?: number | null;
}> = ({ destination, origin, trip_days, trip_start, total_budget,
        travelers_label, lowest_group_estimate }) => (
  <div className="tpv-hero" style={S.hero}>
    <div style={S.heroOverlay} />
    <div style={S.heroContent}>
      {(origin || destination) && (
        <div style={S.heroRoute}>
          {origin && destination ? `${origin}  →  ${destination}` : (origin || destination)}
        </div>
      )}
      <h1 style={S.heroTitle}>{destination || "Your Trip"}</h1>
      <div style={S.heroBadges}>
        {!!trip_days && (
          <span style={S.badge}>✈  {trip_days} days</span>
        )}
        {trip_start && (
          <span style={S.badge}>📅  {trip_start}</span>
        )}
        {total_budget != null && (
          <span style={{ ...S.badge, ...S.badgeTeal }}>
            💰  ${total_budget.toLocaleString()} budget
          </span>
        )}
        {travelers_label && travelers_label !== "1 adult" && (
          <span style={S.badge}>👥  {travelers_label}</span>
        )}
        {lowest_group_estimate != null &&
          travelers_label && travelers_label !== "1 adult" && (
          <span style={{ ...S.badge, ...S.badgeIndigo }}>
            Est. ${lowest_group_estimate.toLocaleString()} group total
          </span>
        )}
      </div>
    </div>
  </div>
);

// ─── 2. Flight option card ────────────────────────────────────────────────────

const FlightLegBlock: FC<{ leg: FlightLeg; direction: "outbound" | "return" }> = ({
  leg, direction,
}) => {
  if (!leg || !Object.keys(leg).length) return null;
  const stops = leg.stops ?? null;
  const isDirect = stops === 0;
  return (
    <div style={S.flightLegBlock}>
      <div style={S.flightLegDir}>
        {direction === "outbound" ? "🛫 Outbound" : "🛬 Return"}
      </div>
      {(leg.airline || leg.label) && (
        <div style={S.flightLegAirline}>
          {[leg.airline, leg.label].filter(Boolean).join(" · ")}
        </div>
      )}
      <div style={S.flightLegMeta}>
        {stops != null && (
          <span
            className={isDirect ? "tpv-stops-direct" : undefined}
            style={{ ...S.stopsBadge, ...(isDirect ? S.stopsDirect : {}) }}
          >
            {stopsLabel(stops)}
          </span>
        )}
        {!!leg.duration_minutes && (
          <span style={S.metaChip}>{fmtDuration(leg.duration_minutes)}</span>
        )}
      </div>
      {leg.departure_time && (
        <div style={S.flightDepart}>Departs {fmtIso(leg.departure_time)}</div>
      )}
      {!!leg.price && (
        <div style={S.flightLegPrice}>${leg.price.toLocaleString()}</div>
      )}
    </div>
  );
};

const FlightCard: FC<{
  pairing: FlightPairing;
  index: number;
  isCheapest: boolean;
}> = ({ pairing, index, isCheapest }) => (
  <div style={{ ...S.flightCard, ...(isCheapest ? S.flightCardBest : {}) }}>
    <div style={S.flightCardTop}>
      <span style={S.flightOptionNo}>Option {index + 1}</span>
      {isCheapest && (
        <span className="tpv-best-value" style={S.bestBadge}>Best Value</span>
      )}
    </div>
    <div style={S.flightTotal}>${pairing.total_price.toLocaleString()}</div>
    <div style={S.flightTotalSub}>total round-trip</div>

    <FlightLegBlock leg={pairing.outbound} direction="outbound" />
    <div style={S.flightDivider} />
    <FlightLegBlock leg={pairing.return_flight} direction="return" />
  </div>
);

// ─── 3. Hotel card ────────────────────────────────────────────────────────────

const HotelCard: FC<{ hotel: Hotel }> = ({ hotel }) => (
  <div style={S.hotelCard}>
    <div style={S.hotelName}>{hotel.name}</div>
    <Stars count={hotel.stars} />
    <div style={S.hotelPrice}>
      ${hotel.price_per_night.toLocaleString()}
      <span style={S.hotelPriceUnit}>/night</span>
    </div>
    {hotel.description && (
      <div style={S.pillDesc}>{hotel.description}</div>
    )}
  </div>
);

// ─── 4. Activity / Restaurant pills ──────────────────────────────────────────

const ActivityPill: FC<{ item: Activity }> = ({ item }) => (
  <div style={S.pill}>
    <div style={S.pillName}><span style={S.pillIcon}>🎯</span>{item.name}</div>
    {item.description && <div style={S.pillDesc}>{item.description}</div>}
  </div>
);

const RestaurantPill: FC<{ item: Restaurant }> = ({ item }) => (
  <div style={S.pill}>
    <div style={S.pillName}><span style={S.pillIcon}>🍽️</span>{item.name}</div>
    <div style={S.pillMeta}>
      {item.price_tier && <span style={S.pillTag}>{item.price_tier}</span>}
      {item.rating != null && <span style={S.pillTag}>{item.rating}★</span>}
    </div>
    {item.description && <div style={S.pillDesc}>{item.description}</div>}
  </div>
);

// ─── 5. Destination Insights ─────────────────────────────────────────────────

const SEASON_ICONS: Record<string, string> = {
  Spring: "🌸", Summer: "☀️", Autumn: "🍂", Winter: "❄️",
};

const InsightsSection: FC<{
  weather: Record<string, string>;
  best_time: { months?: string[]; reason?: string };
}> = ({ weather, best_time }) => {
  const hasBestTime = !!(best_time?.months?.length || best_time?.reason);
  const seasons = (["Spring", "Summer", "Autumn", "Winter"] as const)
    .filter((s) => weather[s]);
  if (!hasBestTime && !seasons.length) return null;

  return (
    <div style={S.insightsCard}>
      <div style={S.sectionLabel}>🌤  Destination Insights</div>

      {hasBestTime && (
        <div style={S.bestTimeBanner}>
          <div style={S.bestTimeTitle}>Best time to visit</div>
          {best_time.months && best_time.months.length > 0 && (
            <div style={S.bestTimeMonths}>
              {best_time.months.map((m) => (
                <span key={m} style={S.bestTimeChip}>{m}</span>
              ))}
            </div>
          )}
          {best_time.reason && (
            <div style={S.bestTimeReason}>{best_time.reason}</div>
          )}
        </div>
      )}

      {seasons.length > 0 && (
        <div className="tpv-season-grid" style={S.seasonGrid}>
          {seasons.map((season) => (
            <div key={season} style={S.seasonChip}>
              <span style={S.seasonIcon}>{SEASON_ICONS[season]}</span>
              <div>
                <div style={S.seasonName}>{season}</div>
                <div style={S.seasonVal}>{weather[season]}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ─── Main export ──────────────────────────────────────────────────────────────

export default function TravelPlanViewer({
  destination = "",
  origin = "",
  trip_days,
  trip_start,
  total_budget,
  travelers_label = "1 adult",
  lowest_group_estimate,
  flight_pairings = [],
  hotels = [],
  activities = [],
  restaurants = [],
  weather = {},
  best_time = {},
}: TravelPlanViewerProps) {
  const stream = useStreamContext();

  const handleSchedule = () => {
    stream.submit(
      {
        messages: [{
          id:      uuidv4(),
          type:    "human",
          content: [{ type: "text", text: "Build my daily schedule" }],
        }],
      },
      {
        config:          { recursion_limit: 100 },
        streamMode:      ["values"],
        streamSubgraphs: true,
        streamResumable: true,
      },
    );
  };

  const cheapestIdx = flight_pairings.length > 0
    ? flight_pairings.reduce(
        (minI, p, i, arr) => (p.total_price < arr[minI].total_price ? i : minI),
        0,
      )
    : -1;

  return (
    <div style={S.root} className="tpv-root">

      {/* ── 1. Hero ── */}
      <Hero
        destination={destination}
        origin={origin}
        trip_days={trip_days}
        trip_start={trip_start}
        total_budget={total_budget}
        travelers_label={travelers_label}
        lowest_group_estimate={lowest_group_estimate}
      />

      {/* ── 2. Flights ── */}
      {flight_pairings.length > 0 && (
        <div style={S.section}>
          <div style={S.sectionLabel}>✈  Flight Options</div>
          <div className="tpv-flight-grid" style={S.flightGrid}>
            {flight_pairings.map((p, i) => (
              <FlightCard
                key={i}
                pairing={p}
                index={i}
                isCheapest={i === cheapestIdx}
              />
            ))}
          </div>
        </div>
      )}

      {/* ── 3. Hotels ── */}
      {hotels.length > 0 && (
        <div style={S.section}>
          <div style={S.sectionLabel}>🏨  Hotels</div>
          <div className="tpv-hotel-scroll" style={S.hotelScroll}>
            {hotels.map((h, i) => (
              <HotelCard key={i} hotel={h} />
            ))}
          </div>
        </div>
      )}

      {/* ── 4. Activities & Restaurants ── */}
      {(activities.length > 0 || restaurants.length > 0) && (
        <div className="tpv-two-col" style={S.twoCol}>
          {activities.length > 0 && (
            <div style={S.section}>
              <div style={S.sectionLabel}>🎯  Activities</div>
              <div style={S.pillList}>
                {activities.map((a, i) => <ActivityPill key={i} item={a} />)}
              </div>
            </div>
          )}
          {restaurants.length > 0 && (
            <div style={S.section}>
              <div style={S.sectionLabel}>🍽  Restaurants</div>
              <div style={S.pillList}>
                {restaurants.map((r, i) => <RestaurantPill key={i} item={r} />)}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── 5. Insights ── */}
      <InsightsSection weather={weather} best_time={best_time} />

      {/* ── 6. CTA ── */}
      <div style={S.ctaWrap}>
        <button className="tpv-cta" style={S.ctaBtn} onClick={handleSchedule}>
          Build my daily schedule →
        </button>
        <div style={S.ctaHint}>
          Starts the full day-by-day itinerary builder
        </div>
      </div>

    </div>
  );
}

// ─── Inline style map ─────────────────────────────────────────────────────────

const S: Record<string, CSSProperties> = {
  root: {
    fontFamily: "var(--font-sans, system-ui, sans-serif)",
    padding:    "0.75rem 0",
    display:    "flex",
    flexDirection: "column",
    gap: 16,
  },

  // ── Hero ──
  hero: {
    borderRadius: 16,
    overflow:     "hidden",
    position:     "relative",
    padding:      "2.5rem 2rem 2rem",
    background:   "linear-gradient(135deg, #0a6e72 0%, #0f5e4a 40%, #2d2680 100%)",
    backgroundSize: "200% 200%",
    animation:    "tpv-shift 10s ease infinite",
  },
  heroOverlay: {
    position: "absolute",
    inset:    0,
    background: "rgba(0,0,0,0.22)",
  },
  heroContent: {
    position: "relative",
    zIndex:   1,
  },
  heroRoute: {
    fontSize:   13,
    color:      "rgba(255,255,255,0.72)",
    fontWeight: 500,
    letterSpacing: "0.04em",
    marginBottom: 6,
  },
  heroTitle: {
    fontSize:      38,
    fontWeight:    900,
    color:         "#ffffff",
    margin:        0,
    letterSpacing: "-0.02em",
    lineHeight:    1.1,
    marginBottom:  18,
    textTransform: "uppercase",
  },
  heroBadges: {
    display:   "flex",
    flexWrap:  "wrap",
    gap:       8,
  },
  badge: {
    fontSize:   12,
    fontWeight: 500,
    padding:    "5px 12px",
    borderRadius: 20,
    background: "rgba(255,255,255,0.17)",
    color:      "#ffffff",
    border:     "1px solid rgba(255,255,255,0.28)",
    backdropFilter: "blur(4px)",
  },
  badgeTeal: {
    background: "rgba(29,158,117,0.5)",
    border:     "1px solid rgba(29,158,117,0.65)",
  },
  badgeIndigo: {
    background: "rgba(99,91,221,0.5)",
    border:     "1px solid rgba(99,91,221,0.65)",
  },

  // ── Generic section wrapper ──
  section: {
    display:       "flex",
    flexDirection: "column",
    gap:           10,
  },
  sectionLabel: {
    fontSize:      11,
    fontWeight:    600,
    color:         "var(--color-text-tertiary, #999)",
    textTransform: "uppercase",
    letterSpacing: "0.07em",
  },

  // ── Flights ──
  flightGrid: {
    display:             "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap:                 10,
  },
  flightCard: {
    background:    "var(--color-background-primary, #fff)",
    border:        "1px solid var(--color-border-tertiary, rgba(0,0,0,0.12))",
    borderRadius:  12,
    padding:       "1rem",
    display:       "flex",
    flexDirection: "column",
    gap:           6,
  },
  flightCardBest: {
    border:     "1.5px solid #1D9E75",
    boxShadow:  "0 0 0 3px rgba(29,158,117,0.1)",
  },
  flightCardTop: {
    display:        "flex",
    alignItems:     "center",
    justifyContent: "space-between",
    marginBottom:   2,
  },
  flightOptionNo: {
    fontSize:      11,
    fontWeight:    600,
    color:         "var(--color-text-tertiary, #999)",
    textTransform: "uppercase",
    letterSpacing: "0.06em",
  },
  bestBadge: {
    fontSize:      10,
    fontWeight:    700,
    color:         "#0F6E56",
    background:    "#E1F5EE",
    padding:       "2px 8px",
    borderRadius:  10,
    textTransform: "uppercase",
    letterSpacing: "0.04em",
  },
  flightTotal: {
    fontSize:   28,
    fontWeight: 800,
    color:      "var(--color-text-primary, #111)",
    lineHeight: 1.1,
  },
  flightTotalSub: {
    fontSize:     11,
    color:        "var(--color-text-tertiary, #aaa)",
    marginBottom: 6,
  },
  flightLegBlock: {
    display:       "flex",
    flexDirection: "column",
    gap:           3,
  },
  flightLegDir: {
    fontSize:   12,
    fontWeight: 600,
    color:      "var(--color-text-secondary, #666)",
  },
  flightLegAirline: {
    fontSize:   13,
    fontWeight: 500,
    color:      "var(--color-text-primary, #111)",
  },
  flightLegMeta: {
    display:    "flex",
    gap:        5,
    flexWrap:   "wrap",
    alignItems: "center",
  },
  stopsBadge: {
    fontSize:     11,
    fontWeight:   500,
    padding:      "2px 7px",
    borderRadius: 4,
    background:   "var(--color-background-secondary, #f0efec)",
    color:        "var(--color-text-secondary, #666)",
  },
  stopsDirect: {
    background: "#E1F5EE",
    color:      "#0F6E56",
  },
  metaChip: {
    fontSize: 11,
    color:    "var(--color-text-secondary, #777)",
  },
  flightDepart: {
    fontSize: 11,
    color:    "var(--color-text-secondary, #888)",
  },
  flightLegPrice: {
    fontSize:   13,
    fontWeight: 600,
    color:      "#1D9E75",
    marginTop:  2,
  },
  flightDivider: {
    height:     1,
    background: "var(--color-border-tertiary, rgba(0,0,0,0.08))",
    margin:     "4px 0",
  },

  // ── Hotels ──
  hotelScroll: {
    display:       "flex",
    gap:           12,
    overflowX:     "auto",
    paddingBottom: 6,
    scrollbarWidth: "thin" as any,
  },
  hotelCard: {
    flex:          "0 0 220px",
    background:    "var(--color-background-primary, #fff)",
    border:        "0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.12))",
    borderRadius:  12,
    padding:       "1rem",
    display:       "flex",
    flexDirection: "column",
    gap:           6,
  },
  hotelName: {
    fontSize:   14,
    fontWeight: 600,
    color:      "var(--color-text-primary, #111)",
    lineHeight: 1.3,
  },
  hotelPrice: {
    fontSize:   20,
    fontWeight: 700,
    color:      "#1D9E75",
    marginTop:  4,
  },
  hotelPriceUnit: {
    fontSize:   12,
    fontWeight: 400,
    color:      "var(--color-text-tertiary, #aaa)",
    marginLeft: 2,
  },

  // ── Two-column grid (Activities + Restaurants) ──
  twoCol: {
    display:             "grid",
    gridTemplateColumns: "1fr 1fr",
    gap:                 16,
  },
  pillList: {
    display:       "flex",
    flexDirection: "column",
    gap:           8,
  },
  pill: {
    background:    "var(--color-background-secondary, #f5f5f3)",
    borderRadius:  10,
    padding:       "10px 12px",
    display:       "flex",
    flexDirection: "column",
    gap:           4,
  },
  pillName: {
    fontSize:   13,
    fontWeight: 600,
    color:      "var(--color-text-primary, #111)",
    display:    "flex",
    alignItems: "center",
  },
  pillIcon: {
    marginRight: 6,
    fontSize:    14,
    flexShrink:  0,
  },
  pillMeta: {
    display:  "flex",
    gap:      5,
    flexWrap: "wrap",
  },
  pillTag: {
    fontSize:     11,
    fontWeight:   500,
    padding:      "1px 6px",
    borderRadius: 4,
    background:   "var(--color-background-primary, #fff)",
    color:        "var(--color-text-secondary, #777)",
    border:       "0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.1))",
  },
  pillDesc: {
    fontSize:   12,
    color:      "var(--color-text-secondary, #777)",
    lineHeight: 1.4,
  },

  // ── Insights ──
  insightsCard: {
    background:    "var(--color-background-primary, #fff)",
    border:        "0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.12))",
    borderRadius:  12,
    padding:       "1rem 1.25rem",
    display:       "flex",
    flexDirection: "column",
    gap:           14,
  },
  bestTimeBanner: {
    background:    "linear-gradient(135deg, rgba(29,158,117,0.07) 0%, rgba(29,100,158,0.05) 100%)",
    border:        "1px solid rgba(29,158,117,0.2)",
    borderRadius:  10,
    padding:       "12px 14px",
    display:       "flex",
    flexDirection: "column",
    gap:           8,
  },
  bestTimeTitle: {
    fontSize:      11,
    fontWeight:    700,
    color:         "#0F6E56",
    textTransform: "uppercase",
    letterSpacing: "0.06em",
  },
  bestTimeMonths: {
    display:  "flex",
    gap:      6,
    flexWrap: "wrap",
  },
  bestTimeChip: {
    fontSize:     12,
    fontWeight:   600,
    color:        "#0F6E56",
    background:   "rgba(29,158,117,0.12)",
    padding:      "3px 10px",
    borderRadius: 12,
  },
  bestTimeReason: {
    fontSize:   13,
    color:      "var(--color-text-secondary, #555)",
    lineHeight: 1.4,
  },
  seasonGrid: {
    display:             "grid",
    gridTemplateColumns: "repeat(4, 1fr)",
    gap:                 8,
  },
  seasonChip: {
    background:    "var(--color-background-secondary, #f5f5f3)",
    borderRadius:  10,
    padding:       "10px 12px",
    display:       "flex",
    alignItems:    "flex-start",
    gap:           8,
  },
  seasonIcon: {
    fontSize:   18,
    flexShrink: 0,
    marginTop:  2,
  },
  seasonName: {
    fontSize:      11,
    fontWeight:    700,
    color:         "var(--color-text-tertiary, #999)",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  seasonVal: {
    fontSize:   12,
    color:      "var(--color-text-primary, #111)",
    marginTop:  2,
    lineHeight: 1.3,
  },

  // ── CTA ──
  ctaWrap: {
    display:       "flex",
    flexDirection: "column",
    alignItems:    "center",
    gap:           8,
    paddingTop:    4,
  },
  ctaBtn: {
    width:        "100%",
    padding:      "14px 24px",
    fontSize:     15,
    fontWeight:   700,
    color:        "#ffffff",
    background:   "linear-gradient(135deg, #1D9E75 0%, #1a60a8 100%)",
    border:       "none",
    borderRadius: 12,
    cursor:       "pointer",
    letterSpacing: "0.01em",
    transition:   "opacity 0.2s, transform 0.1s",
  },
  ctaHint: {
    fontSize:  11,
    color:     "var(--color-text-tertiary, #aaa)",
    textAlign: "center" as const,
  },
};

// ─── Injected styles (animation + dark mode + responsive) ────────────────────

if (typeof document !== "undefined") {
  const STYLES = [
    // Hero gradient animation
    "@keyframes tpv-shift {",
    "  0%   { background-position: 0%   50%; }",
    "  50%  { background-position: 100% 50%; }",
    "  100% { background-position: 0%   50%; }",
    "}",

    // Dark-mode CSS variables (Tailwind .dark class strategy, same as ItineraryViewer)
    ".dark .tpv-root {",
    "  --color-background-primary:  #1c1c1e;",
    "  --color-background-secondary: #2c2c2e;",
    "  --color-text-primary:  #f2f2f7;",
    "  --color-text-secondary: #aeaeb2;",
    "  --color-text-tertiary:  #636366;",
    "  --color-border-secondary: rgba(255,255,255,0.2);",
    "  --color-border-tertiary:  rgba(255,255,255,0.08);",
    "}",

    // Dark-mode badge overrides
    ".dark .tpv-root .tpv-best-value   { background: #0d3028 !important; color: #4fc99e !important; }",
    ".dark .tpv-root .tpv-stops-direct { background: #0d3028 !important; color: #4fc99e !important; }",
    ".dark .tpv-root .tpv-cta          { opacity: 0.93; }",

    // Hotel scrollbar
    ".tpv-hotel-scroll::-webkit-scrollbar              { height: 4px; }",
    ".tpv-hotel-scroll::-webkit-scrollbar-track        { background: transparent; }",
    ".tpv-hotel-scroll::-webkit-scrollbar-thumb        { background: rgba(0,0,0,0.18); border-radius: 2px; }",
    ".dark .tpv-hotel-scroll::-webkit-scrollbar-thumb  { background: rgba(255,255,255,0.18); }",

    // CTA hover feedback (can't do :hover in inline styles)
    ".tpv-cta:hover  { opacity: 0.88; transform: translateY(-1px); }",
    ".tpv-cta:active { opacity: 1;    transform: translateY(0); }",

    // Mobile: stack grids vertically on narrow screens
    "@media (max-width: 640px) {",
    "  .tpv-flight-grid { grid-template-columns: 1fr !important; }",
    "  .tpv-two-col     { grid-template-columns: 1fr !important; }",
    "  .tpv-season-grid { grid-template-columns: repeat(2, 1fr) !important; }",
    "}",
  ].join("\n");

  const existing = document.getElementById("travelplan-styles");
  if (existing) {
    existing.textContent = STYLES;
  } else {
    const style = document.createElement("style");
    style.id    = "travelplan-styles";
    style.textContent = STYLES;
    document.head.appendChild(style);
  }
}
