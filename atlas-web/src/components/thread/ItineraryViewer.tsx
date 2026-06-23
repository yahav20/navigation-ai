/**
 * components/thread/ItineraryViewer.tsx
 *
 * Generative-UI component — rendered inside LoadExternalComponent.
 * Receives props from the backend via the "ItineraryViewer" UI message.
 *
 * Layout:
 *   ┌──────────────────────────┬───────────────┐
 *   │  Flights & Hotels card   │               │
 *   ├──────────────────────────┤  Leaflet Map  │
 *   │  Day nav + timeline      │  (sticky)     │
 *   └──────────────────────────┴───────────────┘
 */

import { useState, useEffect, useRef, type FC, type CSSProperties } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

type SlotType = "activity" | "meal" | "transport" | "rest" | "checkin";

interface TimeSlot {
  time: string;
  end_time?: string;
  duration_minutes?: number;
  slot_type: SlotType;
  name: string;
  description?: string;
  estimated_cost?: number;
  cost_per_person?: number;
  group_cost?: number;
  transport_mode?: "walk" | "taxi";
  distance_km?: number;
  lat?: number;
  lng?: number;
  is_special?: boolean;
}

interface FlightInfo {
  direction: "outbound" | "return";
  airline?: string;
  flight_number?: string;
  route: string;
  datetime: string;
  price: string;
}

interface HotelInfo {
  name: string;
  address?: string;
  stars?: number;
  price_per_night: string;
  nights: number;
  lat?: number;
  lng?: number;
}

interface DaySchedule {
  day: number;
  label: string;
  center_lat?: number;
  center_lng?: number;
  slots: TimeSlot[];
}

interface ItineraryViewerProps {
  destination?: string;
  origin?: string;
  flights?: FlightInfo[];
  hotels?: HotelInfo[];
  days?: DaySchedule[];
}

// ─── Design tokens ────────────────────────────────────────────────────────────

const SLOT_CONFIG: Record<SlotType, { dot: string; bg: string; text: string; label: string }> = {
  activity:  { dot: "#4CAF85", bg: "#E8F4F0", text: "#2D6A56", label: "Attraction" },
  meal:      { dot: "#E8895A", bg: "#FDF0E8", text: "#8B4513", label: "Food"       },
  transport: { dot: "#9898BB", bg: "#F0F0F5", text: "#5A5A75", label: "Transport"  },
  rest:      { dot: "#9B87C8", bg: "#F0EDF8", text: "#5A4580", label: "Rest"       },
  checkin:   { dot: "#6B8FD4", bg: "#E8F0FC", text: "#2A4A8A", label: "Check-in"   },
};

const PIN_COLOR: Record<string, string> = {
  activity:  "#4CAF85",
  meal:      "#E8895A",
  transport: "#9898BB",
  rest:      "#9B87C8",
  checkin:   "#6B8FD4",
  hotel:     "#6B8FD4",
};

// Special events get a distinct soft-cream treatment across the schedule + map.
const SPECIAL_COLOR = "#EAE0CF";              // fill / border / dot / pin
const SPECIAL_INK   = "#8A6A00";              // dark companion for foreground marks (pin label, text)
const SPECIAL_CFG = { dot: SPECIAL_COLOR, bg: "#F4EFE3", text: SPECIAL_INK, label: "🎪 Special event" };

// ─── Leaflet helpers ──────────────────────────────────────────────────────────

function ensureLeafletCss() {
  if (document.getElementById("leaflet-css")) return;
  const link = document.createElement("link");
  link.id = "leaflet-css";
  link.rel = "stylesheet";
  link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
  document.head.appendChild(link);
}

function loadLeaflet(): Promise<any> {
  return new Promise((resolve) => {
    if ((window as any).L) return resolve((window as any).L);
    const script = document.createElement("script");
    script.src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
    script.onload = () => resolve((window as any).L);
    document.head.appendChild(script);
  });
}

function makePinIcon(L: any, color: string, label: string, textColor: string = "#fff") {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="30" height="40" viewBox="0 0 30 40">
    <path d="M15 0C7 0 0 7 0 15c0 11 15 25 15 25S30 26 30 15C30 7 23 0 15 0z"
          fill="${color}" stroke="#fff" stroke-width="1.5"/>
    <text x="15" y="20" text-anchor="middle" font-size="11" font-weight="700"
          font-family="system-ui,sans-serif" fill="${textColor}">${label}</text>
  </svg>`;
  return L.divIcon({
    html: svg,
    className: "",
    iconSize: [30, 40],
    iconAnchor: [15, 40],
    popupAnchor: [0, -40],
  });
}

// ─── Sub-components ───────────────────────────────────────────────────────────

const TravelCard: FC<{ flights: FlightInfo[]; hotels: HotelInfo[] }> = ({ flights, hotels }) => (
  <div style={{ ...S.card, padding: 0, overflow: "hidden" }}>
    <div style={S.cardAccentBar}>Flights &amp; Hotels</div>
    <div style={{ padding: "14px 20px 20px" }}>
      <div style={S.travelGrid}>
        {flights.map((f, i) => (
          <div key={i} style={S.travelItem}>
            <div style={S.travelItemInner}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 4 }}>
                  <span style={{ ...S.directionChip, background: "#E8F0FC", color: "#2A4A8A" }}>
                    {f.direction === "outbound" ? "Outbound" : "Return"}
                  </span>
                </div>
                <div style={S.travelTitle}>{f.airline || f.route}</div>
                <div style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 2, flexWrap: "wrap" }}>
                  {f.flight_number && <span style={S.flightNumChip}>{f.flight_number}</span>}
                  {f.airline && <span style={S.travelSub}>{f.route}</span>}
                </div>
                {f.datetime && <div style={S.travelSub}>{f.datetime}</div>}
              </div>
              <span
                className="itv-price"
                style={{
                  ...S.priceBadge,
                  animationName: "pricePop",
                  animationDuration: "0.5s",
                  animationDelay: "0.6s",
                  animationFillMode: "both",
                  animationTimingFunction: "ease",
                }}
              >
                {f.price}
              </span>
            </div>
          </div>
        ))}
        {hotels.map((h, i) => (
          <div key={i} style={S.travelItem}>
            <div style={S.travelItemInner}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 4 }}>
                  <span style={{ ...S.directionChip, background: "#F0EDF8", color: "#5A4580" }}>Hotel</span>
                  {h.stars && (
                    <span style={S.starText}>{"★".repeat(Math.min(h.stars, 5))}</span>
                  )}
                </div>
                <div style={S.travelTitle}>{h.name}</div>
                <div style={S.travelSub}>
                  {h.nights} nights{h.address ? ` · ${h.address}` : ""}
                </div>
              </div>
              <span
                className="itv-price"
                style={{
                  ...S.priceBadge,
                  animationName: "pricePop",
                  animationDuration: "0.5s",
                  animationDelay: "0.6s",
                  animationFillMode: "both",
                  animationTimingFunction: "ease",
                }}
              >
                {h.price_per_night}
                <span style={{ fontSize: 9, fontWeight: 400, opacity: 0.65 }}>/nt</span>
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  </div>
);

const DayNav: FC<{
  days: DaySchedule[];
  current: number;
  onChange: (i: number) => void;
}> = ({ days, current, onChange }) => {
  const primarySlotType = days[current]?.slots[0]?.slot_type ?? "activity";
  const activeColor = SLOT_CONFIG[primarySlotType]?.dot ?? "#4CAF85";
  const progressPct = days.length > 1 ? (current / (days.length - 1)) * 100 : 100;

  return (
    <div style={S.dayNav}>
      <button
        className="itv-nav-btn"
        style={{ ...S.navBtn, ...(current === 0 ? S.navBtnDisabled : {}) }}
        disabled={current === 0}
        onClick={() => onChange(current - 1)}
        aria-label="Previous day"
      >
        ‹
      </button>
      <div style={{ flex: 1 }}>
        <div style={S.dayLabel}>{days[current]?.label}</div>
        <div style={S.dotsRow}>
          {days.map((_, i) => (
            <button
              key={i}
              onClick={() => onChange(i)}
              aria-label={`Day ${i + 1}`}
              style={{
                ...S.dot,
                ...(i === current ? { ...S.dotActive, background: activeColor } : {}),
              }}
            />
          ))}
        </div>
        <div
          style={{
            height: 2,
            borderRadius: 1,
            marginTop: 7,
            background: `linear-gradient(to right, ${activeColor} ${progressPct}%, var(--color-border-tertiary, rgba(0,0,0,0.08)) ${progressPct}%)`,
            transition: "background 0.4s ease",
          }}
        />
      </div>
      <button
        className="itv-nav-btn"
        style={{ ...S.navBtn, ...(current === days.length - 1 ? S.navBtnDisabled : {}) }}
        disabled={current === days.length - 1}
        onClick={() => onChange(current + 1)}
        aria-label="Next day"
      >
        ›
      </button>
    </div>
  );
};

const SlotRow: FC<{ slot: TimeSlot; isLast: boolean; index: number }> = ({ slot, isLast, index }) => {
  const cfg = slot.is_special ? SPECIAL_CFG : (SLOT_CONFIG[slot.slot_type] ?? SLOT_CONFIG.activity);
  const isSecondary = slot.slot_type === "transport" || slot.slot_type === "rest";
  const animName = isSecondary ? "slideUpDim" : "slideUp";

  return (
    <div
      className={slot.is_special ? "itv-slot-special" : undefined}
      style={{
        ...S.slotRow,
        ...(isLast ? { borderBottom: "none" } : {}),
        ...(slot.is_special ? S.slotRowSpecial : {}),
        animationName: animName,
        animationDuration: "0.4s",
        animationTimingFunction: "ease",
        animationFillMode: "both",
        animationDelay: `${index * 0.04}s`,
      }}
    >
      <div style={S.slotTime}>{slot.time}</div>
      <div style={S.slotDotCol}>
        <div style={{ ...S.slotDot, background: cfg.dot }} />
        {!isLast && <div style={S.slotLine} />}
      </div>
      <div style={S.slotBody}>
        <div style={S.slotName}>{slot.name}</div>
        {slot.description && (
          <div style={S.slotDesc as CSSProperties} title={slot.description}>
            {slot.description}
          </div>
        )}
        <div style={S.slotTags}>
          <span
            data-slot={slot.slot_type}
            data-special={slot.is_special ? "1" : undefined}
            style={{ ...S.tag, background: cfg.bg, color: cfg.text }}
          >
            {cfg.label}
          </span>
          {(slot.estimated_cost ?? 0) > 0 && (
            <span style={S.tagNeutral}>${slot.estimated_cost}</span>
          )}
          {(slot.cost_per_person ?? 0) > 0 && (
            <span style={S.tagNeutral}>${slot.cost_per_person} pp</span>
          )}
          {(slot.group_cost ?? 0) > 0 && (
            <span style={S.tagNeutral}>${slot.group_cost} group</span>
          )}
          {slot.transport_mode && (
            <span style={S.tagNeutral}>
              {slot.transport_mode === "walk" ? "Walk" : "Taxi"}
              {slot.distance_km != null ? ` ${slot.distance_km.toFixed(1)} km` : ""}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

// ─── Map component ────────────────────────────────────────────────────────────

const DayMap: FC<{ day: DaySchedule; hotels: HotelInfo[] }> = ({ day, hotels }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<any>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "no-coords">("loading");

  const primaryColor = SLOT_CONFIG[day.slots[0]?.slot_type ?? "activity"]?.dot ?? "#4CAF85";

  useEffect(() => {
    ensureLeafletCss();
    let cancelled = false;

    async function init() {
      const L = await loadLeaflet();
      if (cancelled || !containerRef.current) return;

      if (mapRef.current) { mapRef.current.remove(); mapRef.current = null; }

      const slotsWithCoords: Array<{ slot: TimeSlot; lat: number; lng: number }> = [];
      for (const s of day.slots) {
        if (s.slot_type === "transport" || s.slot_type === "rest") continue;
        if (s.lat != null && s.lng != null) {
          slotsWithCoords.push({ slot: s, lat: s.lat, lng: s.lng });
        }
      }

      const hotelMarkers: Array<{ hotel: HotelInfo; lat: number; lng: number }> = [];
      for (const h of hotels) {
        if (h.lat != null && h.lng != null) {
          hotelMarkers.push({ hotel: h, lat: h.lat, lng: h.lng });
        }
      }

      if (cancelled || !containerRef.current) return;

      const allPoints = [
        ...slotsWithCoords.map((x) => [x.lat, x.lng] as [number, number]),
        ...hotelMarkers.map((x) => [x.lat, x.lng] as [number, number]),
      ];

      const fallbackLat = day.center_lat ?? 48.8566;
      const fallbackLng = day.center_lng ?? 2.3522;

      if (allPoints.length === 0) {
        setStatus("no-coords");
        return;
      }

      const map = L.map(containerRef.current, { zoomControl: true, scrollWheelZoom: false });
      mapRef.current = map;

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "© OpenStreetMap contributors",
        maxZoom: 19,
      }).addTo(map);

      hotelMarkers.forEach(({ hotel, lat, lng }) => {
        L.marker([lat, lng], { icon: makePinIcon(L, PIN_COLOR.hotel, "H") })
          .addTo(map)
          .bindPopup(`<b>${hotel.name}</b>${hotel.address ? `<br><span style="font-size:11px">${hotel.address}</span>` : ""}`);
      });

      let pinNum = 1;
      slotsWithCoords.forEach(({ slot, lat, lng }) => {
        const color = slot.is_special ? SPECIAL_COLOR : (PIN_COLOR[slot.slot_type] ?? PIN_COLOR.activity);
        const label = slot.is_special ? "★" : String(pinNum++);
        const cfgLabel = slot.is_special ? SPECIAL_CFG.label : (SLOT_CONFIG[slot.slot_type] ?? SLOT_CONFIG.activity).label;
        L.marker([lat, lng], { icon: makePinIcon(L, color, label, slot.is_special ? SPECIAL_INK : "#fff") })
          .addTo(map)
          .bindPopup(
            `<b>${slot.name}</b><br>` +
            `<span style="font-size:11px;color:#666">${cfgLabel}${slot.time ? " · " + slot.time : ""}</span>` +
            (slot.description ? `<br><span style="font-size:11px">${slot.description}</span>` : ""),
          );
      });

      if (allPoints.length > 1) {
        map.fitBounds(allPoints, { padding: [36, 36] });
      } else {
        map.setView(allPoints[0] ?? [fallbackLat, fallbackLng], 15);
      }

      setStatus("ready");
    }

    setStatus("loading");
    init();
    return () => {
      cancelled = true;
      if (mapRef.current) { mapRef.current.remove(); mapRef.current = null; }
    };
  }, [day.day, day.slots]);

  return (
    <div
      className="itv-map-container"
      style={{
        ...S.mapContainer,
        animationName: "fadeIn",
        animationDuration: "0.4s",
        animationDelay: "0.3s",
        animationFillMode: "both",
        animationTimingFunction: "ease",
      }}
    >
      <div style={S.mapInner}>
        <div style={{ ...S.mapHeader, borderLeft: `3px solid ${primaryColor}` }}>
          <span style={S.mapTitle}>Day {day.day}</span>
          {status === "loading" && (
            <span style={{ fontSize: 11, color: "var(--color-text-tertiary, #94A3B8)", marginLeft: "auto" }}>
              locating places…
            </span>
          )}
        </div>

        <div style={{ position: "relative" }}>
          <div ref={containerRef} style={S.mapIframe} />
          {status === "loading" && (
            <div style={S.mapOverlay}>
              <div style={S.mapSpinner} />
              <span className="itv-muted" style={{ fontSize: 12, marginTop: 8 }}>Finding locations…</span>
            </div>
          )}
          {status === "no-coords" && (
            <div style={S.mapOverlay}>
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                style={{ opacity: 0.25 }}
              >
                <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z" />
                <line x1="2" y1="2" x2="22" y2="22" />
              </svg>
              <span className="itv-muted" style={{ fontSize: 12, marginTop: 6 }}>No location data available</span>
            </div>
          )}
        </div>

        <div style={S.mapLegend}>
          {[
            { color: "#4CAF85", label: "Attraction" },
            { color: "#E8895A", label: "Restaurant" },
            { color: "#6B8FD4", label: "Hotel"      },
            ...(day.slots.some((s) => s.is_special) ? [{ color: SPECIAL_COLOR, label: "Special event" }] : []),
          ].map((l) => (
            <div key={l.label} style={S.legendItem}>
              <div
                className="itv-legend-dot"
                style={{ ...S.legendDot, background: l.color }}
              />
              <span>{l.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// ─── Main export ──────────────────────────────────────────────────────────────

export default function ItineraryViewer({
  destination = "Destination",
  flights = [],
  hotels = [],
  days = [],
}: ItineraryViewerProps) {
  const [current, setCurrent] = useState(0);
  useEffect(() => { setCurrent(0); }, [destination]);

  if (days.length === 0) {
    return (
      <div style={S.empty} className="itv-root">
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
          <circle cx="16" cy="16" r="13" stroke="currentColor" strokeWidth="1.5" strokeOpacity={0.2} />
          <path d="M16 3 A13 13 0 0 1 29 16" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <animateTransform
              attributeName="transform"
              type="rotate"
              from="0 16 16"
              to="360 16 16"
              dur="1.2s"
              repeatCount="indefinite"
            />
          </path>
        </svg>
        <span>Preparing your itinerary</span>
      </div>
    );
  }

  const day = days[current];

  return (
    <div style={S.root} className="itv-root">
      <div style={S.layout}>
        <div style={S.leftCol}>
          {(flights.length > 0 || hotels.length > 0) && (
            <TravelCard flights={flights} hotels={hotels} />
          )}
          <div style={{ ...S.card, animationDelay: "0.08s" }}>
            <DayNav days={days} current={current} onChange={setCurrent} />
            <div style={{ marginTop: 16 }}>
              {day.slots.map((slot, i) => (
                <SlotRow key={i} slot={slot} isLast={i === day.slots.length - 1} index={i} />
              ))}
            </div>
          </div>
        </div>
        <DayMap day={day} hotels={hotels} />
      </div>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const S: Record<string, any> = {
  root: {
    fontFamily: "Inter, var(--font-sans, system-ui, sans-serif)",
    padding: "0.75rem 0",
  },
  layout: {
    display: "grid",
    gridTemplateColumns: "1fr 310px",
    gap: 14,
    alignItems: "start",
    position: "relative",
  },
  leftCol: { display: "flex", flexDirection: "column", gap: 12, minWidth: 0 },
  card: {
    background: "var(--color-background-primary, #fff)",
    border: "0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.07))",
    borderRadius: 12,
    padding: "20px",
    boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
    animationName: "slideUp",
    animationDuration: "0.4s",
    animationTimingFunction: "ease",
    animationFillMode: "both",
  },
  cardAccentBar: {
    borderLeft: "3px solid #6B8FD4",
    padding: "9px 16px",
    fontSize: 11,
    fontWeight: 500,
    textTransform: "uppercase",
    letterSpacing: "0.07em",
    color: "var(--color-text-tertiary, #94A3B8)",
    background: "var(--color-background-secondary, #F8F9FC)",
    borderBottom: "0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.06))",
  },
  travelGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
    gap: 8,
  },
  travelItem: {
    background: "var(--color-background-secondary, #F8F9FC)",
    borderRadius: 10,
    padding: "11px 13px",
    boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
    minWidth: 0,
  },
  travelItemInner: {
    display: "flex",
    gap: 8,
    alignItems: "flex-start",
    justifyContent: "space-between",
  },
  directionChip: {
    fontSize: 10,
    fontWeight: 500,
    padding: "1px 6px",
    borderRadius: 4,
    letterSpacing: "0.02em",
    flexShrink: 0,
    lineHeight: 1.8,
  },
  flightNumChip: {
    fontSize: 10,
    fontFamily: "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
    color: "var(--color-text-tertiary, #94A3B8)",
    background: "var(--color-border-tertiary, rgba(0,0,0,0.06))",
    padding: "1px 5px",
    borderRadius: 3,
    flexShrink: 0,
    whiteSpace: "nowrap",
  },
  starText: {
    fontSize: 10,
    color: "#E8895A",
    letterSpacing: "-0.02em",
  },
  travelTitle: {
    fontSize: 13,
    fontWeight: 500,
    color: "var(--color-text-primary, #0F172A)",
    overflowWrap: "anywhere",
    lineHeight: 1.35,
    letterSpacing: "-0.01em",
  },
  travelSub: {
    fontSize: 11,
    color: "var(--color-text-secondary, #64748B)",
    marginTop: 2,
    overflowWrap: "anywhere",
    lineHeight: 1.4,
  },
  priceBadge: {
    fontSize: 12,
    fontWeight: 600,
    color: "#2D6A56",
    whiteSpace: "nowrap",
    flexShrink: 0,
    marginTop: 1,
  },
  dayNav: { display: "flex", alignItems: "center", gap: 12 },
  navBtn: {
    background: "transparent",
    border: "none",
    borderRadius: 8,
    width: 32,
    height: 32,
    cursor: "pointer",
    fontSize: 28,
    color: "var(--color-text-secondary, #64748B)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
    lineHeight: 1,
    padding: 0,
    transition: "color 0.2s ease",
  },
  navBtnDisabled: { opacity: 0.3, cursor: "default" },
  dayLabel: {
    fontSize: 17,
    fontWeight: 600,
    letterSpacing: "-0.02em",
    color: "var(--color-text-primary, #0F172A)",
    lineHeight: 1.2,
  },
  dotsRow: { display: "flex", gap: 5, alignItems: "center", marginTop: 7 },
  dot: {
    width: 6,
    height: 6,
    borderRadius: "50%",
    background: "var(--color-border-secondary, rgba(0,0,0,0.2))",
    cursor: "pointer",
    border: "none",
    padding: 0,
    flexShrink: 0,
    transition: "width 0.25s ease, background 0.25s ease",
  },
  dotActive: { width: 18, borderRadius: 3 },
  slotRow: {
    display: "flex",
    gap: 11,
    alignItems: "flex-start",
    paddingBottom: 12,
    borderBottom: "0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.07))",
    marginBottom: 4,
  },
  slotRowSpecial: {
    borderLeft: "3px solid #EAE0CF",
    background: "transparent",
    borderRadius: 8,
    paddingLeft: 10,
    paddingTop: 6,
    marginLeft: -2,
  },
  slotTime: {
    fontSize: 11,
    fontFamily: "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
    color: "var(--color-text-tertiary, #94A3B8)",
    minWidth: 42,
    paddingTop: 3,
    flexShrink: 0,
    lineHeight: 1.4,
  },
  slotDotCol: { display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0 },
  slotDot: { width: 8, height: 8, borderRadius: "50%", marginTop: 5, flexShrink: 0 },
  slotLine: {
    width: 0,
    flex: 1,
    minHeight: 12,
    marginTop: 3,
    borderLeft: "1.5px dashed var(--color-border-tertiary, rgba(0,0,0,0.15))",
  },
  slotBody: { flex: 1, minWidth: 0, paddingBottom: 4 },
  slotName: {
    fontSize: 14,
    fontWeight: 500,
    color: "var(--color-text-primary, #0F172A)",
    overflowWrap: "anywhere",
    letterSpacing: "-0.01em",
    lineHeight: 1.3,
  },
  slotDesc: {
    fontSize: 12,
    color: "var(--color-text-secondary, #64748B)",
    marginTop: 3,
    lineHeight: 1.6,
    overflowWrap: "anywhere",
    display: "-webkit-box",
    WebkitLineClamp: 2,
    WebkitBoxOrient: "vertical",
    overflow: "hidden",
    textOverflow: "ellipsis",
  },
  slotTags: { display: "flex", gap: 5, marginTop: 6, flexWrap: "wrap" },
  tag: { fontSize: 11, padding: "2px 7px", borderRadius: 4, fontWeight: 500, lineHeight: 1.6 },
  tagNeutral: {
    fontSize: 11,
    padding: "2px 7px",
    borderRadius: 4,
    fontWeight: 500,
    lineHeight: 1.6,
    background: "var(--color-background-secondary, #F8F9FC)",
    color: "var(--color-text-secondary, #64748B)",
  },
  mapContainer: {
    border: "0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.07))",
    borderRadius: 12,
    position: "sticky",
    top: 16,
    maxHeight: "calc(100vh - 32px)",
    boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
    backdropFilter: "blur(0)",
  },
  mapInner: {
    borderRadius: 12,
    overflow: "hidden",
    display: "flex",
    flexDirection: "column",
  },
  mapHeader: {
    padding: "9px 14px",
    display: "flex",
    alignItems: "center",
    gap: 7,
    borderBottom: "0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.07))",
    background: "var(--color-background-primary, #fff)",
  },
  mapTitle: {
    fontSize: 13,
    fontWeight: 500,
    letterSpacing: "-0.01em",
    color: "var(--color-text-primary, #0F172A)",
    flex: 1,
  },
  mapIframe: { width: "100%", height: 420, display: "block" },
  mapOverlay: {
    position: "absolute",
    inset: 0,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    background: "var(--color-background-primary, #fff)",
    zIndex: 10,
  },
  mapSpinner: {
    width: 28,
    height: 28,
    borderRadius: "50%",
    border: "2px solid var(--color-border-secondary, rgba(0,0,0,0.12))",
    borderTopColor: "#4CAF85",
    animation: "spin 0.8s linear infinite",
  },
  mapLegend: {
    display: "flex",
    gap: 12,
    flexWrap: "wrap",
    padding: "8px 14px",
    borderTop: "0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.07))",
    background: "var(--color-background-primary, #fff)",
  },
  legendItem: {
    display: "flex",
    alignItems: "center",
    gap: 5,
    fontSize: 11,
    color: "var(--color-text-secondary, #64748B)",
  },
  legendDot: {
    width: 10,
    height: 10,
    borderRadius: "50%",
    flexShrink: 0,
    boxShadow: "0 0 0 2px #fff",
  },
  empty: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    padding: "3rem 0",
    color: "var(--color-text-tertiary, #94A3B8)",
    fontSize: 14,
  },
};

// ─── Inject styles ────────────────────────────────────────────────────────────

if (typeof document !== "undefined") {
  const STYLES = [
    "@keyframes spin { to { transform: rotate(360deg); } }",
    "@keyframes slideUp { from { opacity:0; transform:translateY(12px) } to { opacity:1; transform:translateY(0) } }",
    "@keyframes slideUpDim { from { opacity:0; transform:translateY(12px) } to { opacity:0.75; transform:translateY(0) } }",
    "@keyframes fadeIn { from { opacity:0 } to { opacity:1 } }",
    "@keyframes pricePop { 0%,100%{transform:scale(1)} 50%{transform:scale(1.06)} }",
    ".itv-nav-btn { transition: color 0.2s ease; }",
    ".itv-nav-btn:hover:not(:disabled) { color: var(--color-text-primary, #0F172A) !important; }",
    // Dark mode root variables
    ".dark .itv-root {",
    "  --color-background-primary: #1A1F2E;",
    "  --color-background-secondary: #242938;",
    "  --color-border-secondary: rgba(255,255,255,0.10);",
    "  --color-border-tertiary: rgba(255,255,255,0.07);",
    "  --color-text-primary: #F0F2F8;",
    "  --color-text-secondary: #8B90A0;",
    "  --color-text-tertiary: #555A6E;",
    "}",
    // Dark mode price & muted
    ".dark .itv-root .itv-price { color: #7DD4B0 !important; }",
    ".dark .itv-root .itv-muted { color: #555A6E !important; }",
    // Dark mode slot tags (12% opacity bg)
    ".dark .itv-root [data-slot=activity]  { background: rgba(76,175,133,0.12) !important; color: #7DD4B0 !important; }",
    ".dark .itv-root [data-slot=meal]      { background: rgba(232,137,90,0.12) !important; color: #F0A882 !important; }",
    ".dark .itv-root [data-slot=transport] { background: rgba(152,152,187,0.12) !important; color: #B8B8D4 !important; }",
    ".dark .itv-root [data-slot=rest]      { background: rgba(155,135,200,0.12) !important; color: #C4B4E4 !important; }",
    ".dark .itv-root [data-slot=checkin]   { background: rgba(107,143,212,0.12) !important; color: #9BB8F0 !important; }",
    // Dark mode special event (after data-slot rules so it wins over [data-slot=activity])
    ".dark .itv-root [data-special='1'] { background: rgba(234,224,207,0.16) !important; color: #EAE0CF !important; }",
    ".dark .itv-root .itv-slot-special  { background: transparent !important; }",
    // Dark mode map inner glow
    ".dark .itv-root .itv-map-container { box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06); }",
    // Dark mode legend dot ring (match dark bg)
    ".dark .itv-root .itv-legend-dot { box-shadow: 0 0 0 2px #1A1F2E !important; }",
  ].join("\n");

  const existing = document.getElementById("itinerary-styles");
  if (existing) {
    existing.textContent = STYLES;
  } else {
    const style = document.createElement("style");
    style.id = "itinerary-styles";
    style.textContent = STYLES;
    document.head.appendChild(style);
  }
}