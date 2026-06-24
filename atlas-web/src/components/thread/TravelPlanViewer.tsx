/**
 * components/thread/TravelPlanViewer.tsx
 *
 * Generative-UI component rendered inside LoadExternalComponent.
 * Receives props from the backend via the "TravelPlanViewer" UI message.
 *
 * Visual: Google Flights × Airbnb × Linear — premium, editorial, zero decorative emoji.
 * Logic, interfaces, and submit call are unchanged from the previous version.
 */

import { useState, type FC, type CSSProperties, type ReactNode } from "react";

// ─── Types (unchanged) ────────────────────────────────────────────────────────

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

interface SpecialEvent {
  name: string;
  description?: string;
  dates?: string;
  location?: string;
  cost?: number;
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
  special_events?: SpecialEvent[];
}

// ─── Pure helpers (unchanged) ─────────────────────────────────────────────────

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

// ─── SVG icons — stroke style, monochrome, no emoji ──────────────────────────

const IconPlane: FC = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17.8 19.2 16 11l3.5-3.5C21 6 21 4 19 2c-2-2-4-2-5.5-.5L10 5 1.8 6.2l2.9 2.9
             4-1.4 3.4 3.4-6.3 6.3 1.4 1.4 5-5 3.4 3.4-1.4 4 2.9 2.9z"/>
  </svg>
);

const IconBed: FC = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2 4v16"/><path d="M2 8h18a2 2 0 0 1 2 2v10"/>
    <path d="M2 17h20"/><path d="M6 8v9"/>
  </svg>
);

const IconMapPin: FC = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/>
    <circle cx="12" cy="10" r="3"/>
  </svg>
);

const IconUtensils: FC = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 2v7c0 1.1.9 2 2 2h4a2 2 0 0 0 2-2V2"/>
    <path d="M7 2v20"/>
    <path d="M21 15V2a5 5 0 0 0-5 5v6c0 1.1.9 2 2 2h3Zm0 0v7"/>
  </svg>
);

const IconCalendar: FC = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
    <line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/>
    <line x1="3" y1="10" x2="21" y2="10"/>
  </svg>
);

const IconSun: FC = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="4"/>
    <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41
             M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>
  </svg>
);

// ─── Square pixel stars (6×6 squares, filled vs unfilled) ────────────────────

const SquareStars: FC<{ count?: number | null }> = ({ count }) => {
  if (!count) return null;
  const n = Math.min(5, Math.round(count));
  return (
    <div style={{ display: "flex", gap: 3, alignItems: "center" }}>
      {Array.from({ length: 5 }, (_, i) => (
        <div key={i} style={{
          width: 6, height: 6, borderRadius: 1,
          background: i < n
            ? "var(--tpv-blue, #103076)"
            : "var(--color-border-secondary, #CBD5E1)",
        }} />
      ))}
    </div>
  );
};

// ─── Rating dots ──────────────────────────────────────────────────────────────

const RatingDots: FC<{ rating: number }> = ({ rating }) => {
  const filled = Math.round(Math.min(5, rating));
  return (
    <div style={{ display: "flex", gap: 3, alignItems: "center" }}>
      {Array.from({ length: 5 }, (_, i) => (
        <div key={i} style={{
          width: 5, height: 5, borderRadius: "50%",
          background: i < filled
            ? "var(--tpv-blue, #103076)"
            : "var(--color-border-secondary, #CBD5E1)",
        }} />
      ))}
    </div>
  );
};

// ─── Section header ───────────────────────────────────────────────────────────

const SectionHead: FC<{ icon: ReactNode; title: string }> = ({ icon, title }) => (
  <div style={S.sectionHead}>
    <span style={S.sectionIcon}>{icon}</span>
    <span style={S.sectionTitle}>{title}</span>
  </div>
);

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
        travelers_label, lowest_group_estimate }) => {

  const subtitleParts: string[] = [];
  if (origin && destination) subtitleParts.push(`${origin} → ${destination}`);
  else if (origin || destination) subtitleParts.push(origin || destination);
  if (trip_days) subtitleParts.push(`${trip_days} days`);
  if (trip_start) subtitleParts.push(trip_start);

  const showGroup = lowest_group_estimate != null
    && travelers_label && travelers_label !== "1 adult";

  return (
    <div style={S.hero}>
      {/* Thin blue accent line across the bottom edge */}
      <div style={S.heroAccentLine} />
      <div style={S.heroContent}>
        {subtitleParts.length > 0 && (
          <p style={S.heroSubtitle}>{subtitleParts.join(" · ")}</p>
        )}
        <h1 style={S.heroTitle}>{destination || "Your Trip"}</h1>
        <div style={S.heroPills}>
          {total_budget != null && (
            <span style={S.heroPill}>Budget ${total_budget.toLocaleString()}</span>
          )}
          {showGroup && (
            <span style={S.heroPill}>
              Group ${lowest_group_estimate!.toLocaleString()} · {travelers_label}
            </span>
          )}
        </div>
      </div>
      <div style={S.heroRule} />
    </div>
  );
};

// ─── 2. Flights — tab switcher ────────────────────────────────────────────────

const FlightRow: FC<{ leg: FlightLeg; label: string }> = ({ leg, label }) => {
  if (!leg || !Object.keys(leg).length) return null;
  const stops     = leg.stops ?? null;
  const isDirect  = stops === 0;
  const isLayover = stops != null && stops > 0;

  return (
    <div style={S.flightRow}>
      {/* Left: direction label, airline, depart time */}
      <div style={S.flightRowLeft}>
        <span style={S.flightRowDir}>{label}</span>
        <div style={S.flightRowAirline}>
          {leg.airline || "—"}
          {leg.label && (
            <span className="tpv-flight-num" style={S.flightNum}>{leg.label}</span>
          )}
        </div>
        {leg.departure_time && (
          <span style={S.flightDepart}>{fmtIso(leg.departure_time)}</span>
        )}
      </div>

      {/* Centre: duration + stops */}
      <div style={S.flightRowMeta}>
        {!!leg.duration_minutes && (
          <span style={S.metaDur}>{fmtDuration(leg.duration_minutes)}</span>
        )}
        {stops != null && (
          <span
            className={isDirect ? "tpv-stops-direct" : isLayover ? "tpv-stops-amber" : undefined}
            style={{
              ...S.stopsBadge,
              ...(isDirect  ? S.stopsDirect : {}),
              ...(isLayover ? S.stopsAmber  : {}),
            }}
          >
            {stopsLabel(stops)}
          </span>
        )}
      </div>

      {/* Right: price */}
      {!!leg.price && (
        <div style={S.flightRowPrice}>${leg.price.toLocaleString()}</div>
      )}
    </div>
  );
};

const FlightsSection: FC<{ pairings: FlightPairing[]; cheapestIdx: number }> = ({
  pairings, cheapestIdx,
}) => {
  const [tab, setTab] = useState(0);
  const pairing = pairings[tab];
  if (!pairing) return null;

  return (
    <div>
      {/* Tab bar */}
      <div style={S.tabBar}>
        {pairings.map((_, i) => (
          <button
            key={i}
            onClick={() => setTab(i)}
            className={`tpv-tab${tab === i ? " tpv-tab-active" : ""}`}
            style={{ ...S.tab, ...(tab === i ? S.tabActive : {}) }}
          >
            Option {i + 1}
            {i === cheapestIdx && <span style={S.tabDot} />}
          </button>
        ))}
      </div>

      {/* Panel */}
      <div style={S.flightPanel}>
        <FlightRow leg={pairing.outbound}      label="Outbound" />
        <div style={S.flightPanelDivider} />
        <FlightRow leg={pairing.return_flight} label="Return" />
        <div style={S.flightTotalRow}>
          <span style={S.flightTotalLabel}>Round-trip total</span>
          <span style={S.flightTotalPrice}>${pairing.total_price.toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
};

// ─── 3. Hotel card ────────────────────────────────────────────────────────────

const HotelCard: FC<{ hotel: Hotel; isCheapest: boolean }> = ({ hotel, isCheapest }) => (
  <div style={{ ...S.hotelCard, ...(isCheapest ? S.hotelCardCheapest : {}) }}>
    <div style={S.hotelName}>{hotel.name}</div>
    <SquareStars count={hotel.stars} />
    <div style={S.hotelPriceRow}>
      <span style={S.hotelPrice}>${hotel.price_per_night.toLocaleString()}</span>
      <span style={S.hotelPriceUnit}>/night</span>
    </div>
    {hotel.description && (
      <div style={S.hotelDesc}>{hotel.description}</div>
    )}
  </div>
);

// ─── 4a. Activity row ─────────────────────────────────────────────────────────

const ActivityRow: FC<{ item: Activity; index: number }> = ({ item, index }) => (
  <div style={S.activityRow}>
    <span style={S.activityNum}>{index + 1}</span>
    <div style={S.activityBody}>
      <div style={S.activityName}>{item.name}</div>
      {item.description && (
        <div style={S.activityDesc}>{item.description}</div>
      )}
    </div>
  </div>
);

// ─── 4b. Restaurant card ──────────────────────────────────────────────────────

const RestaurantCard: FC<{ item: Restaurant }> = ({ item }) => (
  <div style={S.restCard}>
    <div style={S.restName}>{item.name}</div>
    <div style={S.restMeta}>
      {item.rating != null && (
        <>
          <span style={S.restRating}>{item.rating}</span>
          <RatingDots rating={item.rating} />
        </>
      )}
      {item.price_tier && (
        <span className="tpv-price-tier" style={S.priceTier}>{item.price_tier}</span>
      )}
    </div>
    {item.description && (
      <div style={S.restDesc}>{item.description}</div>
    )}
  </div>
);

// ─── 5. Events ───────────────────────────────────────────────────────────────

const EventsSection: FC<{ events: SpecialEvent[] }> = ({ events }) => {
  if (!events?.length) return null;
  return (
    <div style={S.sectionCard}>
      <SectionHead icon={<span style={{ color: "#8A6A00", display: "flex" }}><IconCalendar /></span>} title="Events" />
      <div style={S.eventList}>
        {events.map((ev, i) => {
          const meta = [ev.dates, ev.location].filter(Boolean).join("  ·  ");
          return (
            <div key={i} style={S.eventCard}>
              <div style={S.eventName}>{ev.name}</div>
              {(meta || (ev.cost ?? 0) > 0) && (
                <div style={S.eventMeta}>
                  {meta && <span>{meta}</span>}
                  {(ev.cost ?? 0) > 0 && (
                    <span style={S.eventCost}>${Math.round(ev.cost as number)}</span>
                  )}
                </div>
              )}
              {ev.description && <div style={S.eventDesc}>{ev.description}</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ─── 6. Insights ─────────────────────────────────────────────────────────────

const SEASONS = ["Spring", "Summer", "Autumn", "Winter"] as const;

const InsightsSection: FC<{
  weather: Record<string, string>;
  best_time: { months?: string[]; reason?: string };
}> = ({ weather, best_time }) => {
  const hasBestTime = !!(best_time?.months?.length || best_time?.reason);
  const seasons     = SEASONS.filter((s) => weather[s]);
  if (!hasBestTime && !seasons.length) return null;

  return (
    <div style={S.insightsCard}>
      <SectionHead icon={<IconSun />} title="Insights" />

      {hasBestTime && (
        <div style={S.bestTimeBanner}>
          <span style={S.bestTimeSpark}>✦</span>
          <span style={S.bestTimeBody}>
            <span style={S.bestTimeHead}>Best time: </span>
            {best_time.months && best_time.months.length > 0 && (
              <span style={S.bestTimeMonths}>{best_time.months.join(", ")}</span>
            )}
            {best_time.reason && (
              <span style={S.bestTimeReason}> — {best_time.reason}</span>
            )}
          </span>
        </div>
      )}

      {seasons.length > 0 && (
        <div className="tpv-season-grid" style={S.seasonGrid}>
          {seasons.map((season) => (
            <div
              key={season}
              style={{
                ...S.seasonCard,
                ...(season === "Summer" ? S.seasonCardHighlight : {}),
              }}
            >
              <div style={S.seasonName}>{season}</div>
              <div style={{
                ...S.seasonVal,
                ...(season === "Summer" ? S.seasonValHighlight : {}),
              }}>
                {weather[season]}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ─── Main export — logic unchanged ───────────────────────────────────────────

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
  special_events = [],
}: TravelPlanViewerProps) {
  const cheapestIdx = flight_pairings.length > 0
    ? flight_pairings.reduce(
        (minI, p, i, arr) => (p.total_price < arr[minI].total_price ? i : minI),
        0,
      )
    : -1;

  const cheapestHotelIdx = hotels.length > 0
    ? hotels.reduce(
        (minI, h, i, arr) => (h.price_per_night < arr[minI].price_per_night ? i : minI),
        0,
      )
    : -1;

  const budgetPct =
    lowest_group_estimate != null && total_budget != null && total_budget > 0
      ? Math.min(100, (lowest_group_estimate / total_budget) * 100)
      : null;

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
        <div style={S.sectionCard}>
          <SectionHead icon={<IconPlane />} title="Flights" />
          <FlightsSection pairings={flight_pairings} cheapestIdx={cheapestIdx} />
        </div>
      )}

      {/* ── 3. Hotels ── */}
      {hotels.length > 0 && (
        <div style={S.section}>
          <SectionHead icon={<IconBed />} title="Hotels" />
          <div className="tpv-hotel-scroll" style={S.hotelScroll}>
            {hotels.map((h, i) => (
              <HotelCard key={i} hotel={h} isCheapest={i === cheapestHotelIdx} />
            ))}
          </div>
        </div>
      )}

      {/* ── 4. Activities & Restaurants ── */}
      {(activities.length > 0 || restaurants.length > 0) && (
        <div className="tpv-two-col" style={S.twoCol}>
          {activities.length > 0 && (
            <div style={S.sectionCard}>
              <SectionHead icon={<IconMapPin />} title="Activities" />
              <div style={S.activityList}>
                {activities.map((a, i) => (
                  <ActivityRow key={i} item={a} index={i} />
                ))}
              </div>
            </div>
          )}
          {restaurants.length > 0 && (
            <div style={S.sectionCard}>
              <SectionHead icon={<IconUtensils />} title="Restaurants" />
              <div style={S.restList}>
                {restaurants.map((r, i) => (
                  <RestaurantCard key={i} item={r} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── 5. Events ── */}
      {special_events.length > 0 && (
        <EventsSection events={special_events} />
      )}

      {/* ── 6. Insights ── */}
      <InsightsSection weather={weather} best_time={best_time} />

      {/* ── 7. Budget ── */}
      {(total_budget != null || lowest_group_estimate != null) && (
        <div style={S.ctaBudget}>
          <div style={S.ctaBudgetText}>
            {lowest_group_estimate != null && total_budget != null
              ? `Estimated group cost: $${lowest_group_estimate.toLocaleString()} of $${total_budget.toLocaleString()} budget`
              : total_budget != null
                ? `Budget: $${total_budget.toLocaleString()}`
                : `Estimated cost: $${lowest_group_estimate?.toLocaleString()}`}
          </div>
          {budgetPct !== null && (
            <div style={S.progressTrack}>
              <div style={{ ...S.progressFill, width: `${budgetPct}%` }} />
            </div>
          )}
        </div>
      )}

    </div>
  );
}

// ─── Style map ────────────────────────────────────────────────────────────────

const S: Record<string, CSSProperties> = {
  root: {
    fontFamily:    "var(--font-sans, system-ui, -apple-system, sans-serif)",
    padding:       "0.5rem 0 1rem",
    display:       "flex",
    flexDirection: "column",
    gap:           12,
    color:         "var(--color-text-primary, #0F172A)",
  },

  // ── Hero ──────────────────────────────────────────────────────────────────
  hero: {
    borderRadius: 14,
    overflow:     "hidden",
    position:     "relative",
    background:   "linear-gradient(150deg, #0F172A 0%, #1E293B 55%, #0c1629 100%)",
    padding:      "2.25rem 2rem 0",
  },
  heroAccentLine: {
    position:   "absolute",
    bottom:     0,
    left:       "6%",
    right:      "6%",
    height:     1,
    background: "linear-gradient(90deg, transparent, rgba(37,99,235,0.65), transparent)",
  },
  heroContent: {
    position: "relative",
    zIndex:   1,
  },
  heroSubtitle: {
    margin:        0,
    fontSize:      12,
    fontWeight:    500,
    color:         "rgba(255,255,255,0.45)",
    letterSpacing: "0.03em",
    marginBottom:  8,
  },
  heroTitle: {
    margin:        0,
    fontSize:      48,
    fontWeight:    800,
    color:         "#ffffff",
    letterSpacing: "-0.03em",
    lineHeight:    1.05,
    marginBottom:  20,
    overflowWrap:  "anywhere",
  },
  heroPills: {
    display:      "flex",
    gap:          8,
    flexWrap:     "wrap",
    marginBottom: 24,
  },
  heroPill: {
    fontSize:       12,
    fontWeight:     500,
    color:          "rgba(255,255,255,0.82)",
    background:     "rgba(255,255,255,0.1)",
    border:         "1px solid rgba(255,255,255,0.18)",
    borderRadius:   20,
    padding:        "5px 13px",
    backdropFilter: "blur(6px)",
    letterSpacing:  "0.01em",
  },
  heroRule: {
    height:     1,
    background: "rgba(255,255,255,0.07)",
    margin:     "0 -2rem",
    position:   "relative",
    zIndex:     1,
  },

  // ── Generic containers ─────────────────────────────────────────────────────
  section: {
    display:       "flex",
    flexDirection: "column",
    gap:           10,
  },
  sectionCard: {
    background:    "var(--color-background-primary, #ffffff)",
    border:        "1px solid var(--color-border-tertiary, #E2E8F0)",
    borderRadius:  12,
    padding:       "1.25rem",
    display:       "flex",
    flexDirection: "column",
    gap:           16,
  },
  sectionHead: {
    display:    "flex",
    alignItems: "center",
    gap:        7,
    color:      "var(--color-text-tertiary, #64748B)",
  },
  sectionIcon: {
    display:    "flex",
    alignItems: "center",
    flexShrink: 0,
  },
  sectionTitle: {
    fontSize:      11,
    fontWeight:    600,
    letterSpacing: "0.07em",
    textTransform: "uppercase" as const,
  },

  // ── Flights: tab bar ───────────────────────────────────────────────────────
  tabBar: {
    display:      "flex",
    borderBottom: "1px solid var(--color-border-tertiary, #E2E8F0)",
    marginBottom: 4,
  },
  tab: {
    position:   "relative",
    padding:    "8px 16px",
    fontSize:   13,
    fontWeight: 500,
    color:      "var(--color-text-secondary, #64748B)",
    background: "none",
    border:     "none",
    cursor:     "pointer",
    display:    "flex",
    alignItems: "center",
    gap:        6,
    outline:    "none",
    transition: "color 0.15s",
  },
  tabActive: {
    color:      "var(--tpv-blue, #103076)",
    fontWeight: 600,
  },
  tabDot: {
    width:        5,
    height:       5,
    borderRadius: "50%",
    background:   "var(--tpv-blue, #103076)",
    flexShrink:   0,
  },

  // ── Flights: rows ──────────────────────────────────────────────────────────
  flightPanel: {
    display:       "flex",
    flexDirection: "column",
  },
  flightRow: {
    display:    "flex",
    alignItems: "flex-start",
    gap:        12,
    padding:    "14px 0",
  },
  flightRowLeft: {
    flex:          1,
    display:       "flex",
    flexDirection: "column",
    gap:           3,
    minWidth:      0,
  },
  flightRowDir: {
    fontSize:      10,
    fontWeight:    700,
    color:         "var(--color-text-tertiary, #94A3B8)",
    textTransform: "uppercase" as const,
    letterSpacing: "0.08em",
  },
  flightRowAirline: {
    fontSize:    14,
    fontWeight:  600,
    color:       "var(--color-text-primary, #0F172A)",
    display:     "flex",
    alignItems:  "center",
    gap:         7,
    flexWrap:    "wrap" as any,
    lineHeight:  1.3,
  },
  flightNum: {
    fontSize:    11,
    fontFamily:  "ui-monospace, 'SF Mono', Monaco, monospace",
    color:       "var(--color-text-tertiary, #94A3B8)",
    background:  "var(--color-background-secondary, #F1F5F9)",
    padding:     "1px 6px",
    borderRadius: 3,
    fontWeight:  400,
    letterSpacing: "0.04em",
  },
  flightDepart: {
    fontSize: 11,
    color:    "var(--color-text-tertiary, #94A3B8)",
  },
  flightRowMeta: {
    display:    "flex",
    alignItems: "center",
    gap:        6,
    flexShrink: 0,
  },
  metaDur: {
    fontSize:   12,
    fontWeight: 500,
    color:      "var(--color-text-secondary, #475569)",
  },
  stopsBadge: {
    fontSize:     11,
    fontWeight:   600,
    padding:      "2px 8px",
    borderRadius: 4,
    background:   "var(--color-background-secondary, #F1F5F9)",
    color:        "var(--color-text-secondary, #475569)",
  },
  stopsDirect: {
    background: "#F0FDF4",
    color:      "#16A34A",
  },
  stopsAmber: {
    background: "#FFFBEB",
    color:      "#D97706",
  },
  flightRowPrice: {
    fontSize:           16,
    fontWeight:         700,
    color:              "var(--color-text-primary, #0F172A)",
    fontVariantNumeric: "tabular-nums" as any,
    flexShrink:         0,
    minWidth:           52,
    textAlign:          "right" as const,
  },
  flightPanelDivider: {
    height:     1,
    background: "var(--color-border-tertiary, #F1F5F9)",
  },
  flightTotalRow: {
    display:         "flex",
    alignItems:      "center",
    justifyContent:  "space-between",
    paddingTop:      14,
    marginTop:       6,
    borderTop:       "1px solid var(--color-border-tertiary, #E2E8F0)",
  },
  flightTotalLabel: {
    fontSize:   12,
    fontWeight: 500,
    color:      "var(--color-text-tertiary, #94A3B8)",
  },
  flightTotalPrice: {
    fontSize:           22,
    fontWeight:         800,
    color:              "var(--tpv-blue, #103076)",
    fontVariantNumeric: "tabular-nums" as any,
    letterSpacing:      "-0.02em",
  },

  // ── Hotels ─────────────────────────────────────────────────────────────────
  hotelScroll: {
    display:        "flex",
    gap:            10,
    overflowX:      "auto",
    paddingBottom:  4,
    scrollbarWidth: "thin" as any,
  },
  hotelCard: {
    flex:          "0 0 215px",
    background:    "var(--color-background-primary, #ffffff)",
    border:        "1px solid var(--color-border-tertiary, #E2E8F0)",
    borderRadius:  10,
    padding:       "1rem",
    display:       "flex",
    flexDirection: "column",
    gap:           7,
  },
  hotelCardCheapest: {
    borderLeft: "3px solid var(--tpv-blue, #103076)",
  },
  hotelName: {
    fontSize:   14,
    fontWeight: 600,
    color:      "var(--color-text-primary, #0F172A)",
    lineHeight: 1.3,
  },
  hotelPriceRow: {
    display:    "flex",
    alignItems: "baseline",
    gap:        3,
    marginTop:  2,
  },
  hotelPrice: {
    fontSize:           22,
    fontWeight:         800,
    color:              "var(--tpv-blue, #103076)",
    letterSpacing:      "-0.02em",
    fontVariantNumeric: "tabular-nums" as any,
  },
  hotelPriceUnit: {
    fontSize: 12,
    color:    "var(--color-text-tertiary, #94A3B8)",
  },
  hotelDesc: {
    fontSize:   12,
    color:      "var(--color-text-secondary, #64748B)",
    lineHeight: 1.45,
  },

  // ── Activities + Restaurants side-by-side ──────────────────────────────────
  twoCol: {
    display:             "grid",
    gridTemplateColumns: "1fr 1fr",
    gap:                 12,
  },
  activityList: {
    display:       "flex",
    flexDirection: "column",
  },
  activityRow: {
    display:       "flex",
    gap:           12,
    padding:       "10px 0",
    borderBottom:  "1px solid var(--color-border-tertiary, #F1F5F9)",
    alignItems:    "flex-start",
  },
  activityNum: {
    fontSize:   12,
    fontWeight: 700,
    color:      "var(--tpv-blue, #103076)",
    minWidth:   16,
    paddingTop: 1,
    flexShrink: 0,
  },
  activityBody: {
    flex:     1,
    minWidth: 0,
  },
  activityName: {
    fontSize:   13,
    fontWeight: 600,
    color:      "var(--color-text-primary, #0F172A)",
    lineHeight: 1.3,
  },
  activityDesc: {
    fontSize:   12,
    color:      "var(--color-text-secondary, #64748B)",
    lineHeight: 1.4,
    marginTop:  2,
  },
  restList: {
    display:       "flex",
    flexDirection: "column",
    gap:           8,
  },
  restCard: {
    background:    "var(--color-background-secondary, #F8FAFC)",
    border:        "1px solid var(--color-border-tertiary, #E2E8F0)",
    borderRadius:  8,
    padding:       "10px 12px",
    display:       "flex",
    flexDirection: "column",
    gap:           5,
  },
  restName: {
    fontSize:   13,
    fontWeight: 600,
    color:      "var(--color-text-primary, #0F172A)",
  },
  restMeta: {
    display:    "flex",
    alignItems: "center",
    gap:        7,
    flexWrap:   "wrap" as any,
  },
  restRating: {
    fontSize:           13,
    fontWeight:         700,
    color:              "var(--tpv-blue, #103076)",
    fontVariantNumeric: "tabular-nums" as any,
  },
  priceTier: {
    fontSize:   12,
    fontWeight: 600,
    color:      "var(--tpv-blue, #103076)",
  },
  restDesc: {
    fontSize:   11,
    color:      "var(--color-text-secondary, #64748B)",
    lineHeight: 1.4,
  },

  // ── Events ─────────────────────────────────────────────────────────────────
  eventList: { display: "flex", flexDirection: "column" as const, gap: 8 },
  eventCard: {
    background:    "rgba(234,224,207,0.5)",
    borderLeft:    "3px solid #EAE0CF",
    borderRadius:  8,
    padding:       "10px 12px",
    display:       "flex",
    flexDirection: "column" as const,
    gap:           4,
  },
  eventName: { fontSize: 13, fontWeight: 600, color: "var(--color-text-primary, #0F172A)", lineHeight: 1.3 },
  eventMeta: {
    display:    "flex",
    alignItems: "center",
    gap:        8,
    flexWrap:   "wrap" as const,
    fontSize:   12,
    fontWeight: 500,
    color:      "#8A6A00",
  },
  eventCost: {
    fontSize:   11,
    fontWeight: 700,
    color:      "#8A6A00",
    background: "rgba(234,224,207,0.8)",
    padding:    "1px 7px",
    borderRadius: 4,
  },
  eventDesc: { fontSize: 11, color: "var(--color-text-secondary, #64748B)", lineHeight: 1.45 },

  // ── Insights ───────────────────────────────────────────────────────────────
  insightsCard: {
    background:    "var(--color-background-primary, #ffffff)",
    border:        "1px solid var(--color-border-tertiary, #E2E8F0)",
    borderRadius:  12,
    padding:       "1.25rem",
    display:       "flex",
    flexDirection: "column",
    gap:           14,
  },
  bestTimeBanner: {
    display:      "flex",
    alignItems:   "flex-start",
    gap:          8,
    background:   "var(--tpv-blue-bg, rgba(37,99,235,0.05))",
    border:       "1px solid var(--tpv-blue-border, rgba(37,99,235,0.15))",
    borderRadius: 8,
    padding:      "10px 13px",
  },
  bestTimeSpark: {
    fontSize:   12,
    color:      "var(--tpv-blue, #103076)",
    flexShrink: 0,
    marginTop:  2,
    fontWeight: 700,
  },
  bestTimeBody: {
    fontSize:   13,
    lineHeight: 1.55,
  },
  bestTimeHead: {
    fontWeight: 600,
    color:      "var(--color-text-primary, #0F172A)",
  },
  bestTimeMonths: {
    fontWeight: 600,
    color:      "var(--tpv-blue, #1a45a3)",
  },
  bestTimeReason: {
    fontWeight: 400,
    color:      "var(--color-text-secondary, #475569)",
  },
  seasonGrid: {
    display:             "grid",
    gridTemplateColumns: "repeat(4, 1fr)",
    gap:                 8,
  },
  seasonCard: {
    background:    "var(--color-background-secondary, #F8FAFC)",
    border:        "1px solid var(--color-border-tertiary, #E2E8F0)",
    borderRadius:  8,
    padding:       "10px 12px",
    display:       "flex",
    flexDirection: "column",
    gap:           4,
  },
  seasonCardHighlight: {
    background: "var(--tpv-blue-bg, rgba(37,99,235,0.06))",
    border:     "1px solid var(--tpv-blue-border, rgba(37,99,235,0.2))",
  },
  seasonName: {
    fontSize:      10,
    fontWeight:    700,
    color:         "var(--color-text-tertiary, #94A3B8)",
    textTransform: "uppercase" as const,
    letterSpacing: "0.07em",
  },
  seasonVal: {
    fontSize:   13,
    fontWeight: 500,
    color:      "var(--color-text-primary, #0F172A)",
    lineHeight: 1.3,
  },
  seasonValHighlight: {
    color:      "var(--tpv-blue, #103076)",
    fontWeight: 600,
  },

  // ── CTA ────────────────────────────────────────────────────────────────────
  ctaSection: {
    display:       "flex",
    flexDirection: "column",
    gap:           10,
    paddingTop:    4,
  },
  ctaBtn: {
    width:         "100%",
    height:        48,
    fontSize:      15,
    fontWeight:    600,
    color:         "#ffffff",
    background:    "var(--tpv-blue, #103076)",
    border:        "none",
    borderRadius:  10,
    cursor:        "pointer",
    letterSpacing: "-0.01em",
    transition:    "box-shadow 0.2s, transform 0.12s",
  },
  ctaBudget: {
    display:       "flex",
    flexDirection: "column",
    gap:           6,
  },
  ctaBudgetText: {
    fontSize:  12,
    color:     "var(--color-text-secondary, #64748B)",
    textAlign: "center" as const,
  },
  progressTrack: {
    height:       4,
    borderRadius: 2,
    background:   "var(--color-border-tertiary, #E2E8F0)",
    overflow:     "hidden",
  },
  progressFill: {
    height:     "100%",
    borderRadius: 2,
    background:   "var(--tpv-blue, #103076)",
    transition:   "width 0.5s ease",
  },
};

// ─── Injected CSS — dark mode, pseudo-elements, hover, responsive ─────────────

if (typeof document !== "undefined") {
  const STYLES = [
    // Tab active underline via ::after (can't do pseudo in inline styles)
    ".tpv-tab { position: relative; }",
    ".tpv-tab:hover { color: var(--color-text-primary, #0F172A); }",
    ".tpv-tab-active::after {",
    "  content: '';",
    "  position: absolute;",
    "  bottom: -1px; left: 0; right: 0;",
    "  height: 2px;",
    "  background: var(--tpv-blue, #103076);",
    "  border-radius: 1px 1px 0 0;",
    "}",

    // CTA hover glow
    ".tpv-cta:hover  { box-shadow: 0 0 0 4px rgba(16,48,118,0.22); transform: translateY(-1px); }",
    ".tpv-cta:active { box-shadow: none; transform: translateY(0); }",

    // Dark mode CSS variable overrides (Tailwind .dark class — same as ItineraryViewer)
    ".dark .tpv-root {",
    "  --tpv-blue:        #3B82F6;",
    "  --tpv-blue-bg:     rgba(59,130,246,0.08);",
    "  --tpv-blue-border: rgba(59,130,246,0.22);",
    "  --color-background-primary:    #111827;",
    "  --color-background-secondary:  #1F2937;",
    "  --color-text-primary:    #F1F5F9;",
    "  --color-text-secondary:  #94A3B8;",
    "  --color-text-tertiary:   #4B5563;",
    "  --color-border-secondary: rgba(255,255,255,0.14);",
    "  --color-border-tertiary:  rgba(255,255,255,0.08);",
    "}",

    // Dark: badge overrides
    ".dark .tpv-root .tpv-stops-direct { background: rgba(22,163,74,0.15)  !important; color: #4ADE80 !important; }",
    ".dark .tpv-root .tpv-stops-amber  { background: rgba(217,119,6,0.15)  !important; color: #FCD34D !important; }",
    ".dark .tpv-root .tpv-flight-num   { background: rgba(255,255,255,0.07) !important; color: #4B5563 !important; }",
    ".dark .tpv-root .tpv-price-tier   { color: #FCD34D !important; }",
    ".dark .tpv-root .tpv-cta:hover    { box-shadow: 0 0 0 4px rgba(59,130,246,0.28); }",

    // Hotel scrollbar
    ".tpv-hotel-scroll::-webkit-scrollbar              { height: 3px; }",
    ".tpv-hotel-scroll::-webkit-scrollbar-track        { background: transparent; }",
    ".tpv-hotel-scroll::-webkit-scrollbar-thumb        { background: rgba(0,0,0,0.14); border-radius: 2px; }",
    ".dark .tpv-hotel-scroll::-webkit-scrollbar-thumb  { background: rgba(255,255,255,0.12); }",

    // Responsive: stack columns on narrow screens
    "@media (max-width: 640px) {",
    "  .tpv-two-col     { grid-template-columns: 1fr !important; }",
    "  .tpv-season-grid { grid-template-columns: repeat(2, 1fr) !important; }",
    "}",
  ].join("\n");

  const existing = document.getElementById("travelplan-styles");
  if (existing) {
    existing.textContent = STYLES;
  } else {
    const style = document.createElement("style");
    style.id          = "travelplan-styles";
    style.textContent = STYLES;
    document.head.appendChild(style);
  }
}