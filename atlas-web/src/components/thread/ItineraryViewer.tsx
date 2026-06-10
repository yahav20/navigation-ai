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
  transport_mode?: "walk" | "taxi";
  distance_km?: number;
  lat?: number;
  lng?: number;
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
  activity:  { dot: "#1D9E75", bg: "#E1F5EE", text: "#0F6E56", label: "Attraction" },
  meal:      { dot: "#D85A30", bg: "#FAECE7", text: "#993C1D", label: "Food"       },
  transport: { dot: "#888780", bg: "#F1EFE8", text: "#5F5E5A", label: "Transport"  },
  rest:      { dot: "#7F77DD", bg: "#EEEDFE", text: "#534AB7", label: "Rest"       },
  checkin:   { dot: "#378ADD", bg: "#E6F1FB", text: "#185FA5", label: "Check-in"   },
};

const PIN_COLOR: Record<string, string> = {
  activity:  "#1D9E75",
  meal:      "#D85A30",
  transport: "#888780",
  rest:      "#7F77DD",
  checkin:   "#378ADD",
  hotel:     "#378ADD",
};

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

function makePinIcon(L: any, color: string, label: string) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="30" height="40" viewBox="0 0 30 40">
    <path d="M15 0C7 0 0 7 0 15c0 11 15 25 15 25S30 26 30 15C30 7 23 0 15 0z"
          fill="${color}" stroke="#fff" stroke-width="1.5"/>
    <text x="15" y="20" text-anchor="middle" font-size="11" font-weight="700"
          font-family="system-ui,sans-serif" fill="#fff">${label}</text>
  </svg>`;
  return L.divIcon({
    html: svg,
    className: "",
    iconSize: [30, 40],
    iconAnchor: [15, 40],
    popupAnchor: [0, -40],
  });
}

// Nominatim geocode cache (persists across re-renders)
const geocodeCache = new Map<string, [number, number] | null>();

async function geocode(name: string, city: string): Promise<[number, number] | null> {
  const query = `${name}, ${city}`;
  if (geocodeCache.has(query)) return geocodeCache.get(query)!;
  try {
    const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query)}&format=json&limit=1`;
    const res = await fetch(url, { headers: { "Accept-Language": "en" } });
    const data = await res.json();
    if (data?.[0]) {
      const coord: [number, number] = [parseFloat(data[0].lat), parseFloat(data[0].lon)];
      geocodeCache.set(query, coord);
      return coord;
    }
  } catch {
    // Geocoding is optional; cache the failed lookup below and render without a pin.
  }
  geocodeCache.set(query, null);
  return null;
}

// ─── Sub-components ───────────────────────────────────────────────────────────

const TravelCard: FC<{ flights: FlightInfo[]; hotels: HotelInfo[] }> = ({ flights, hotels }) => (
  <div style={S.card}>
    <p style={S.sectionLabel}>✈ Flights &amp; Accommodation</p>
    <div style={S.travelGrid}>
      {flights.map((f, i) => (
        <div key={i} style={S.travelItem}>
          <span style={S.travelIcon}>{f.direction === "outbound" ? "🛫" : "🛬"}</span>
          <div style={S.travelMeta}>
            <div style={S.travelTitle}>{f.route}</div>
            <div style={S.travelSub}>
              {f.datetime}
              {f.airline ? ` · ${f.airline}` : ""}
              {f.flight_number ? ` ${f.flight_number}` : ""}
            </div>
          </div>
          <span style={S.priceBadge}>{f.price}</span>
        </div>
      ))}
      {hotels.map((h, i) => (
        <div key={i} style={S.travelItem}>
          <span style={S.travelIcon}>🏨</span>
          <div style={S.travelMeta}>
            <div style={S.travelTitle}>{h.name}</div>
            <div style={S.travelSub}>
              {h.nights} nights
              {h.address ? ` · ${h.address}` : ""}
              {h.stars ? ` · ${"★".repeat(h.stars)}` : ""}
            </div>
          </div>
          <span style={S.priceBadge}>{h.price_per_night}</span>
        </div>
      ))}
    </div>
  </div>
);

const DayNav: FC<{
  days: DaySchedule[];
  current: number;
  onChange: (i: number) => void;
}> = ({ days, current, onChange }) => (
  <div style={S.dayNav}>
    <button
      style={{ ...S.navBtn, ...(current === 0 ? S.navBtnDisabled : {}) }}
      disabled={current === 0}
      onClick={() => onChange(current - 1)}
      aria-label="Previous day"
    >‹</button>
    <div style={{ flex: 1 }}>
      <div style={S.dayLabel}>{days[current]?.label}</div>
      <div style={S.dotsRow}>
        {days.map((_, i) => (
          <button
            key={i}
            onClick={() => onChange(i)}
            aria-label={`Day ${i + 1}`}
            style={{ ...S.dot, ...(i === current ? S.dotActive : {}) }}
          />
        ))}
      </div>
    </div>
    <button
      style={{ ...S.navBtn, ...(current === days.length - 1 ? S.navBtnDisabled : {}) }}
      disabled={current === days.length - 1}
      onClick={() => onChange(current + 1)}
      aria-label="Next day"
    >›</button>
  </div>
);

const SlotRow: FC<{ slot: TimeSlot; isLast: boolean }> = ({ slot, isLast }) => {
  const cfg = SLOT_CONFIG[slot.slot_type] ?? SLOT_CONFIG.activity;
  return (
    <div style={{ ...S.slotRow, ...(isLast ? { borderBottom: "none" } : {}) }}>
      <div style={S.slotTime}>{slot.time}</div>
      <div style={S.slotDotCol}>
        <div style={{ ...S.slotDot, background: cfg.dot }} />
        {!isLast && <div style={S.slotLine} />}
      </div>
      <div style={S.slotBody}>
        <div style={S.slotName}>{slot.name}</div>
        {slot.description && <div style={S.slotDesc}>{slot.description}</div>}
        <div style={S.slotTags}>
          <span style={{ ...S.tag, background: cfg.bg, color: cfg.text }}>{cfg.label}</span>
          {(slot.estimated_cost ?? 0) > 0 && (
            <span style={S.tagNeutral}>${slot.estimated_cost}</span>
          )}
          {slot.transport_mode && (
            <span style={S.tagNeutral}>
              {slot.transport_mode === "walk" ? "🚶" : "🚕"}
              {slot.distance_km != null ? ` ${slot.distance_km.toFixed(1)} km` : ""}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

// ─── Map component ────────────────────────────────────────────────────────────

const DayMap: FC<{ day: DaySchedule; hotels: HotelInfo[]; destination: string }> = ({
  day,
  hotels,
  destination,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<any>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "no-coords">("loading");

  useEffect(() => {
    ensureLeafletCss();
    let cancelled = false;

    async function init() {
      const L = await loadLeaflet();
      if (cancelled || !containerRef.current) return;

      // Destroy old map
      if (mapRef.current) { mapRef.current.remove(); mapRef.current = null; }

      // --- Resolve coordinates ---
      // Slots with backend-provided lat/lng
      const slotsWithCoords: Array<{ slot: TimeSlot; lat: number; lng: number }> = [];
      // Slots needing geocoding
      const slotsNeedingGeo: TimeSlot[] = [];

      for (const s of day.slots) {
        if (s.slot_type === "transport" || s.slot_type === "rest") continue;
        if (s.lat != null && s.lng != null) {
          slotsWithCoords.push({ slot: s, lat: s.lat, lng: s.lng });
        } else {
          slotsNeedingGeo.push(s);
        }
      }

      // Geocode missing slots (throttle: 1 request / 300 ms to respect Nominatim ToS)
      for (const s of slotsNeedingGeo) {
        if (cancelled) return;
        const coord = await geocode(s.name, destination);
        if (coord) slotsWithCoords.push({ slot: s, lat: coord[0], lng: coord[1] });
        await new Promise((r) => setTimeout(r, 300));
      }

      // Hotels
      const hotelMarkers: Array<{ hotel: HotelInfo; lat: number; lng: number }> = [];
      for (const h of hotels) {
        if (h.lat != null && h.lng != null) {
          hotelMarkers.push({ hotel: h, lat: h.lat, lng: h.lng });
        } else {
          const coord = await geocode(h.name, destination);
          if (coord) hotelMarkers.push({ hotel: h, lat: coord[0], lng: coord[1] });
          if (cancelled) return;
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

      // Hotel pins
      hotelMarkers.forEach(({ hotel, lat, lng }) => {
        L.marker([lat, lng], { icon: makePinIcon(L, PIN_COLOR.hotel, "🏨") })
          .addTo(map)
          .bindPopup(`<b>🏨 ${hotel.name}</b>${hotel.address ? `<br><span style="font-size:11px">${hotel.address}</span>` : ""}`);
      });

      // Slot pins (numbered, skip transport/rest)
      let pinNum = 1;
      slotsWithCoords.forEach(({ slot, lat, lng }) => {
        const color = PIN_COLOR[slot.slot_type] ?? PIN_COLOR.activity;
        const cfg = SLOT_CONFIG[slot.slot_type] ?? SLOT_CONFIG.activity;
        L.marker([lat, lng], { icon: makePinIcon(L, color, String(pinNum++)) })
          .addTo(map)
          .bindPopup(
            `<b>${slot.name}</b><br>` +
            `<span style="font-size:11px;color:#666">${cfg.label}${slot.time ? " · " + slot.time : ""}</span>` +
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
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [day.day, destination]);

  return (
    <div style={S.mapContainer}>
      <div style={S.mapHeader}>
        <span>📍</span>
        <span style={S.mapTitle}>Day {day.day} — map</span>
        {status === "loading" && (
          <span style={{ fontSize: 11, color: "var(--color-text-tertiary, #aaa)", marginLeft: "auto" }}>
            locating places…
          </span>
        )}
      </div>

      <div style={{ position: "relative" }}>
        <div ref={containerRef} style={S.mapIframe} />
        {status === "loading" && (
          <div style={S.mapOverlay}>
            <div style={S.mapSpinner} />
            <span style={{ fontSize: 12, color: "#666", marginTop: 8 }}>Finding locations…</span>
          </div>
        )}
        {status === "no-coords" && (
          <div style={S.mapOverlay}>
            <span style={{ fontSize: 24 }}>🗺️</span>
            <span style={{ fontSize: 12, color: "#888", marginTop: 6 }}>No location data available</span>
          </div>
        )}
      </div>

      <div style={S.mapLegend}>
        {[
          { color: "#1D9E75", label: "Attraction" },
          { color: "#D85A30", label: "Restaurant" },
          { color: "#378ADD", label: "Hotel"      },
        ].map((l) => (
          <div key={l.label} style={S.legendItem}>
            <div style={{ ...S.legendDot, background: l.color }} />
            <span>{l.label}</span>
          </div>
        ))}
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
      <div style={S.empty}>
        <span style={{ fontSize: 28 }}>✈️</span>
        <span>Building your itinerary…</span>
      </div>
    );
  }

  const day = days[current];

  return (
    <div style={S.root}>
      <div style={S.layout}>
        <div style={S.leftCol}>
          {(flights.length > 0 || hotels.length > 0) && (
            <TravelCard flights={flights} hotels={hotels} />
          )}
          <div style={S.card}>
            <DayNav days={days} current={current} onChange={setCurrent} />
            <div style={{ marginTop: 14 }}>
              {day.slots.map((slot, i) => (
                <SlotRow key={i} slot={slot} isLast={i === day.slots.length - 1} />
              ))}
            </div>
          </div>
        </div>
        <DayMap day={day} hotels={hotels} destination={destination} />
      </div>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const S: Record<string, CSSProperties> = {
  root: { fontFamily: "var(--font-sans, system-ui, sans-serif)", padding: "0.75rem 0" },
  layout: { display: "grid", gridTemplateColumns: "1fr 310px", gap: 14, alignItems: "start" },
  leftCol: { display: "flex", flexDirection: "column", gap: 12, minWidth: 0 },
  card: {
    background: "var(--color-background-primary, #fff)",
    border: "0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.12))",
    borderRadius: 12,
    padding: "1rem 1.25rem",
  },
  sectionLabel: {
    fontSize: 11, fontWeight: 500, color: "var(--color-text-tertiary, #999)",
    textTransform: "uppercase", letterSpacing: "0.06em", margin: "0 0 10px",
  },
  travelGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 },
  travelItem: {
    background: "var(--color-background-secondary, #f5f5f3)",
    borderRadius: 8, padding: "9px 11px", display: "flex", gap: 9, alignItems: "flex-start",
  },
  travelIcon: { fontSize: 17, flexShrink: 0, marginTop: 1 },
  travelMeta: { flex: 1, minWidth: 0 },
  travelTitle: {
    fontSize: 13, fontWeight: 500, color: "var(--color-text-primary, #111)",
    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
  },
  travelSub: { fontSize: 12, color: "var(--color-text-secondary, #666)", marginTop: 2 },
  priceBadge: { fontSize: 12, fontWeight: 500, color: "#0F6E56", whiteSpace: "nowrap", flexShrink: 0 },
  dayNav: { display: "flex", alignItems: "center", gap: 12 },
  navBtn: {
    background: "var(--color-background-secondary, #f5f5f3)",
    border: "0.5px solid var(--color-border-secondary, rgba(0,0,0,0.2))",
    borderRadius: 8, width: 32, height: 32, cursor: "pointer", fontSize: 22,
    color: "var(--color-text-primary, #111)", display: "flex", alignItems: "center",
    justifyContent: "center", flexShrink: 0, lineHeight: 1, padding: 0,
  },
  navBtnDisabled: { opacity: 0.3, cursor: "default" },
  dayLabel: { fontSize: 15, fontWeight: 500, color: "var(--color-text-primary, #111)" },
  dotsRow: { display: "flex", gap: 5, alignItems: "center", marginTop: 6 },
  dot: {
    width: 6, height: 6, borderRadius: "50%",
    background: "var(--color-border-secondary, rgba(0,0,0,0.25))",
    cursor: "pointer", border: "none", padding: 0, flexShrink: 0,
  },
  dotActive: { width: 18, borderRadius: 3, background: "var(--color-text-primary, #111)" },
  slotRow: {
    display: "flex", gap: 11, alignItems: "flex-start", paddingBottom: 10,
    borderBottom: "0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.08))", marginBottom: 2,
  },
  slotTime: { fontSize: 12, color: "var(--color-text-tertiary, #aaa)", minWidth: 42, paddingTop: 3, flexShrink: 0 },
  slotDotCol: { display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0 },
  slotDot: { width: 8, height: 8, borderRadius: "50%", marginTop: 5, flexShrink: 0 },
  slotLine: { width: 1, flex: 1, minHeight: 12, marginTop: 3, background: "var(--color-border-tertiary, rgba(0,0,0,0.1))" },
  slotBody: { flex: 1, minWidth: 0, paddingBottom: 4 },
  slotName: { fontSize: 13, fontWeight: 500, color: "var(--color-text-primary, #111)" },
  slotDesc: { fontSize: 12, color: "var(--color-text-secondary, #666)", marginTop: 2, lineHeight: 1.4 },
  slotTags: { display: "flex", gap: 5, marginTop: 5, flexWrap: "wrap" },
  tag: { fontSize: 11, padding: "2px 7px", borderRadius: 4, fontWeight: 500, lineHeight: 1.6 },
  tagNeutral: {
    fontSize: 11, padding: "2px 7px", borderRadius: 4, fontWeight: 500, lineHeight: 1.6,
    background: "var(--color-background-secondary, #f5f5f3)", color: "var(--color-text-secondary, #777)",
  },
  mapContainer: {
    borderRadius: 12, overflow: "hidden",
    border: "0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.12))",
    position: "sticky", top: 16,
  },
  mapHeader: {
    padding: "9px 13px", display: "flex", alignItems: "center", gap: 7,
    borderBottom: "0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.1))",
    background: "var(--color-background-primary, #fff)",
  },
  mapTitle: { fontSize: 13, fontWeight: 500, color: "var(--color-text-primary, #111)", flex: 1 },
  mapIframe: { width: "100%", height: 420, display: "block" },
  mapOverlay: {
    position: "absolute", inset: 0, display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center",
    background: "var(--color-background-primary, #fff)",
    zIndex: 10,
  },
  mapSpinner: {
    width: 28, height: 28, borderRadius: "50%",
    border: "3px solid var(--color-border-secondary, rgba(0,0,0,0.15))",
    borderTopColor: "#1D9E75",
    animation: "spin 0.8s linear infinite",
  },
  mapLegend: {
    display: "flex", gap: 12, flexWrap: "wrap", padding: "7px 13px",
    borderTop: "0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.1))",
    background: "var(--color-background-primary, #fff)",
  },
  legendItem: { display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: "var(--color-text-secondary, #777)" },
  legendDot: { width: 8, height: 8, borderRadius: "50%", flexShrink: 0 },
  empty: {
    display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
    gap: 8, padding: "3rem 0", color: "var(--color-text-tertiary, #aaa)", fontSize: 14,
  },
};

// Inject spinner keyframes once
if (typeof document !== "undefined" && !document.getElementById("itinerary-spin")) {
  const style = document.createElement("style");
  style.id = "itinerary-spin";
  style.textContent = "@keyframes spin { to { transform: rotate(360deg); } }";
  document.head.appendChild(style);
}
