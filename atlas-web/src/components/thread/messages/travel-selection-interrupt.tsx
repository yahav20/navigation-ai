/**
 * TravelSelectionInterruptView
 *
 * Rich HITL card selector rendered when plan_check.py emits a
 * { type: "travel_selection", flight_pairings, hotels } interrupt.
 *
 * The user picks one flight pairing and one hotel, then clicks
 * "Confirm selection". They can also click "Choose for me" to
 * auto-select the cheapest options (resume value: "auto").
 *
 * Resume value format: "flight:N,hotel:M" or "auto".
 */

import { useState, type CSSProperties, type FC, type ReactNode } from "react";
import { motion } from "framer-motion";
import { useStreamContext } from "@/providers/Stream";
import { MarkdownText } from "../markdown-text";
import type {
  TravelSelectionInterrupt,
  TravelSelectionFlightLeg,
  TravelSelectionFlightPairing,
  TravelSelectionHotel,
} from "@/lib/travel-selection-interrupt";

// ─── Helpers (mirrors TravelPlanViewer) ──────────────────────────────────────

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

function cheapestPairingIdx(pairings: TravelSelectionFlightPairing[]): number {
  if (!pairings.length) return 0;
  return pairings.reduce(
    (minI, p, i, arr) => (p.total_price < arr[minI].total_price ? i : minI),
    0,
  );
}

function cheapestHotelIdx(hotels: TravelSelectionHotel[]): number {
  if (!hotels.length) return 0;
  return hotels.reduce(
    (minI, h, i, arr) => (h.price_per_night < arr[minI].price_per_night ? i : minI),
    0,
  );
}

// ─── SVG icons ────────────────────────────────────────────────────────────────

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

// ─── Square pixel stars ───────────────────────────────────────────────────────

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

// ─── Section header ───────────────────────────────────────────────────────────

const SectionHead: FC<{ icon: ReactNode; title: string }> = ({ icon, title }) => (
  <div style={S.sectionHead}>
    <span style={S.sectionIcon}>{icon}</span>
    <span style={S.sectionTitle}>{title}</span>
  </div>
);

// ─── Flight leg row ───────────────────────────────────────────────────────────

const FlightLegRow: FC<{ leg: TravelSelectionFlightLeg; label: string }> = ({ leg, label }) => {
  if (!leg || !Object.keys(leg).length) return null;
  const stops     = leg.stops ?? null;
  const isDirect  = stops === 0;
  const isLayover = stops != null && stops > 0;

  return (
    <div style={S.flightRow}>
      <div style={S.flightRowLeft}>
        <span style={S.flightRowDir}>{label}</span>
        <div style={S.flightRowAirline}>
          {leg.airline || "—"}
          {leg.label && (
            <span className="tsi-flight-num" style={S.flightNum}>{leg.label}</span>
          )}
        </div>
        {leg.departure_time && (
          <span style={S.flightDepart}>{fmtIso(leg.departure_time)}</span>
        )}
      </div>
      <div style={S.flightRowMeta}>
        {!!leg.duration_minutes && (
          <span style={S.metaDur}>{fmtDuration(leg.duration_minutes)}</span>
        )}
        {stops != null && (
          <span
            className={isDirect ? "tsi-stops-direct" : isLayover ? "tsi-stops-amber" : undefined}
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
      {!!leg.price && (
        <div style={S.flightRowPrice}>${leg.price.toLocaleString()}</div>
      )}
    </div>
  );
};

// ─── Selectable flight pairing card ──────────────────────────────────────────

const FlightPairingCard: FC<{
  pairing: TravelSelectionFlightPairing;
  index: number;
  isSelected: boolean;
  isCheapest: boolean;
  disabled: boolean;
  onSelect: (i: number) => void;
}> = ({ pairing, index, isSelected, isCheapest, disabled, onSelect }) => (
  <motion.div
    whileTap={disabled ? {} : { scale: 0.995 }}
    onClick={() => !disabled && onSelect(index)}
    style={{
      ...S.pairingCard,
      ...(isSelected ? S.pairingCardSelected : {}),
      cursor: disabled ? "default" : "pointer",
      opacity: disabled ? 0.7 : 1,
    }}
  >
    <div style={S.pairingCardHeader}>
      <span style={S.pairingLabel}>Option {index + 1}</span>
      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        {isCheapest && <span style={S.cheapestBadge}>Cheapest</span>}
        <span style={S.pairingTotal}>${pairing.total_price.toLocaleString()}</span>
      </div>
    </div>
    <div style={S.pairingDivider} />
    <FlightLegRow leg={pairing.outbound}      label="Outbound" />
    <div style={S.flightPanelDivider} />
    <FlightLegRow leg={pairing.return_flight} label="Return" />
    {isSelected && <div style={S.selectedRing} />}
  </motion.div>
);

// ─── Selectable hotel card ────────────────────────────────────────────────────

const HotelSelectionCard: FC<{
  hotel: TravelSelectionHotel;
  index: number;
  isSelected: boolean;
  isCheapest: boolean;
  disabled: boolean;
  onSelect: (i: number) => void;
}> = ({ hotel, index, isSelected, isCheapest, disabled, onSelect }) => (
  <motion.div
    whileTap={disabled ? {} : { scale: 0.995 }}
    onClick={() => !disabled && onSelect(index)}
    style={{
      ...S.hotelCard,
      ...(isSelected ? S.hotelCardSelected : {}),
      cursor: disabled ? "default" : "pointer",
      opacity: disabled ? 0.7 : 1,
    }}
  >
    {isCheapest && <span style={S.cheapestBadge}>Cheapest</span>}
    <div style={S.hotelName}>{hotel.name}</div>
    <SquareStars count={hotel.stars} />
    <div style={S.hotelPriceRow}>
      <span style={S.hotelPrice}>${hotel.price_per_night.toLocaleString()}</span>
      <span style={S.hotelPriceUnit}>/night</span>
    </div>
    {hotel.description && (
      <div style={S.hotelDesc}>{hotel.description}</div>
    )}
  </motion.div>
);

// ─── Main component ───────────────────────────────────────────────────────────

export function TravelSelectionInterruptView({
  interrupt,
}: {
  interrupt: TravelSelectionInterrupt;
}) {
  const thread = useStreamContext();
  const [selectedFlight, setSelectedFlight] = useState<number | null>(null);
  const [selectedHotel,  setSelectedHotel]  = useState<number | null>(null);
  const [submitted, setSubmitted] = useState<string | null>(null);

  const isLoading = thread.isLoading;
  const isDisabled = isLoading || submitted !== null;

  const cfIdx = cheapestPairingIdx(interrupt.flight_pairings);
  const chIdx = cheapestHotelIdx(interrupt.hotels);

  const resume = (value: string) => {
    if (isDisabled) return;
    setSubmitted(value);
    thread.submit(undefined, {
      command: { resume: value },
      config: { recursion_limit: 100 },
      streamMode: ["values"],
      streamSubgraphs: true,
      streamResumable: true,
    });
  };

  const handleConfirm = () => {
    if (selectedFlight === null || selectedHotel === null) return;
    resume(`flight:${selectedFlight},hotel:${selectedHotel}`);
  };

  const handleAuto = () => {
    resume("auto");
  };

  const canConfirm = selectedFlight !== null && selectedHotel !== null;

  return (
    <div className="tsi-root" style={S.root}>
      {/* Header */}
      <div style={S.header}>
        <h3 style={S.headerTitle}>Your input is needed</h3>
      </div>

      <div style={S.body}>
        <div style={S.question}>
          <MarkdownText>{interrupt.question}</MarkdownText>
        </div>

        {/* ── Flights ── */}
        {interrupt.flight_pairings.length > 0 && (
          <div style={S.section}>
            <SectionHead icon={<IconPlane />} title="Select a Flight" />
            <div style={S.pairingList}>
              {interrupt.flight_pairings.map((pairing, i) => (
                <FlightPairingCard
                  key={i}
                  pairing={pairing}
                  index={i}
                  isSelected={selectedFlight === i}
                  isCheapest={i === cfIdx}
                  disabled={isDisabled}
                  onSelect={setSelectedFlight}
                />
              ))}
            </div>
          </div>
        )}

        {/* ── Hotels ── */}
        {interrupt.hotels.length > 0 && (
          <div style={S.section}>
            <SectionHead icon={<IconBed />} title="Select a Hotel" />
            <div className="tsi-hotel-scroll" style={S.hotelScroll}>
              {interrupt.hotels.map((hotel, i) => (
                <HotelSelectionCard
                  key={i}
                  hotel={hotel}
                  index={i}
                  isSelected={selectedHotel === i}
                  isCheapest={i === chIdx}
                  disabled={isDisabled}
                  onSelect={setSelectedHotel}
                />
              ))}
            </div>
          </div>
        )}

        {/* ── Actions ── */}
        <div style={S.actions}>
          <motion.button
            whileTap={isDisabled ? {} : { scale: 0.98 }}
            style={{ ...S.btnSecondary, ...(isDisabled ? { opacity: 0.5 } : {}) }}
            disabled={isDisabled}
            onClick={handleAuto}
          >
            Choose for me (cheapest)
          </motion.button>
          <motion.button
            whileTap={isDisabled || !canConfirm ? {} : { scale: 0.98 }}
            style={{
              ...S.btnPrimary,
              ...(!canConfirm || isDisabled ? S.btnPrimaryDisabled : {}),
            }}
            disabled={isDisabled || !canConfirm}
            onClick={handleConfirm}
          >
            Confirm selection
          </motion.button>
        </div>

        {submitted !== null && (
          <p style={S.sendingNote}>Sending your selection…</p>
        )}
      </div>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const S: Record<string, CSSProperties> = {
  root: {
    fontFamily:   "var(--font-sans, system-ui, -apple-system, sans-serif)",
    borderRadius: 10,
    border:       "1px solid var(--color-border-tertiary, #E2E8F0)",
    overflow:     "hidden",
  },
  header: {
    borderBottom: "1px solid var(--color-border-tertiary, #E2E8F0)",
    background:   "var(--color-background-secondary, #F8FAFC)",
    padding:      "10px 16px",
  },
  headerTitle: {
    margin:     0,
    fontSize:   14,
    fontWeight: 600,
    color:      "var(--color-text-primary, #0F172A)",
  },
  body: {
    display:       "flex",
    flexDirection: "column",
    gap:           16,
    padding:       "16px",
  },
  question: {
    fontSize:   14,
    color:      "var(--color-text-primary, #0F172A)",
    lineHeight: 1.5,
  },
  section: {
    display:       "flex",
    flexDirection: "column",
    gap:           10,
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

  // ── Flight pairing cards ───────────────────────────────────────────────────
  pairingList: {
    display:       "flex",
    flexDirection: "column",
    gap:           8,
  },
  pairingCard: {
    position:      "relative",
    background:    "var(--color-background-primary, #ffffff)",
    borderWidth:   "1.5px",
    borderStyle:   "solid",
    borderColor:   "var(--color-border-tertiary, #E2E8F0)",
    borderRadius:  10,
    padding:       "12px 14px",
    display:       "flex",
    flexDirection: "column",
    gap:           0,
    transition:    "border-color 0.15s",
  },
  pairingCardSelected: {
    borderColor: "var(--tpv-blue, #103076)",
    background:  "var(--tpv-blue-bg, rgba(16,48,118,0.03))",
  },
  selectedRing: {
    position:     "absolute",
    inset:        0,
    borderRadius: 10,
    border:       "2px solid var(--tpv-blue, #103076)",
    pointerEvents: "none" as const,
  },
  pairingCardHeader: {
    display:         "flex",
    justifyContent:  "space-between",
    alignItems:      "center",
    marginBottom:    8,
  },
  pairingLabel: {
    fontSize:   12,
    fontWeight: 600,
    color:      "var(--color-text-secondary, #475569)",
  },
  pairingTotal: {
    fontSize:           18,
    fontWeight:         800,
    color:              "var(--tpv-blue, #103076)",
    letterSpacing:      "-0.02em",
    fontVariantNumeric: "tabular-nums" as any,
  },
  pairingDivider: {
    height:       1,
    background:   "var(--color-border-tertiary, #F1F5F9)",
    marginBottom: 4,
  },
  cheapestBadge: {
    fontSize:     10,
    fontWeight:   700,
    padding:      "2px 7px",
    borderRadius: 20,
    background:   "#F0FDF4",
    color:        "#16A34A",
    letterSpacing: "0.03em",
    textTransform: "uppercase" as const,
  },

  // ── Flight rows (shared with TravelPlanViewer) ────────────────────────────
  flightRow: {
    display:    "flex",
    alignItems: "flex-start",
    gap:        12,
    padding:    "10px 0",
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
    fontSize:      11,
    fontFamily:    "ui-monospace, 'SF Mono', Monaco, monospace",
    color:         "var(--color-text-tertiary, #94A3B8)",
    background:    "var(--color-background-secondary, #F1F5F9)",
    padding:       "1px 6px",
    borderRadius:  3,
    fontWeight:    400,
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
  stopsDirect: { background: "#F0FDF4", color: "#16A34A" },
  stopsAmber:  { background: "#FFFBEB", color: "#D97706" },
  flightRowPrice: {
    fontSize:           15,
    fontWeight:         700,
    color:              "var(--color-text-primary, #0F172A)",
    fontVariantNumeric: "tabular-nums" as any,
    flexShrink:         0,
    minWidth:           48,
    textAlign:          "right" as const,
  },
  flightPanelDivider: {
    height:     1,
    background: "var(--color-border-tertiary, #F1F5F9)",
  },

  // ── Hotel cards ────────────────────────────────────────────────────────────
  hotelScroll: {
    display:        "flex",
    gap:            10,
    overflowX:      "auto",
    paddingBottom:  4,
    scrollbarWidth: "thin" as any,
  },
  hotelCard: {
    position:      "relative",
    flex:          "0 0 200px",
    background:    "var(--color-background-primary, #ffffff)",
    borderWidth:   "1.5px",
    borderStyle:   "solid",
    borderColor:   "var(--color-border-tertiary, #E2E8F0)",
    borderRadius:  10,
    padding:       "12px",
    display:       "flex",
    flexDirection: "column",
    gap:           7,
    transition:    "border-color 0.15s",
  },
  hotelCardSelected: {
    borderColor: "var(--tpv-blue, #103076)",
    background:  "var(--tpv-blue-bg, rgba(16,48,118,0.03))",
    boxShadow:   "0 0 0 1px var(--tpv-blue, #103076)",
  },
  hotelName: {
    fontSize:   13,
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
    fontSize:           20,
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
    fontSize:   11,
    color:      "var(--color-text-secondary, #64748B)",
    lineHeight: 1.4,
  },

  // ── Action buttons ─────────────────────────────────────────────────────────
  actions: {
    display:    "flex",
    gap:        10,
    flexWrap:   "wrap" as any,
    marginTop:  4,
  },
  btnSecondary: {
    flex:          "1 1 auto",
    height:        40,
    fontSize:      13,
    fontWeight:    600,
    color:         "var(--tpv-blue, #103076)",
    background:    "transparent",
    border:        "1.5px solid var(--tpv-blue, #103076)",
    borderRadius:  8,
    cursor:        "pointer",
    letterSpacing: "-0.01em",
    transition:    "background 0.15s",
  },
  btnPrimary: {
    flex:          "2 1 auto",
    height:        40,
    fontSize:      13,
    fontWeight:    600,
    color:         "#ffffff",
    background:    "var(--tpv-blue, #103076)",
    border:        "none",
    borderRadius:  8,
    cursor:        "pointer",
    letterSpacing: "-0.01em",
    transition:    "opacity 0.15s, box-shadow 0.15s",
  },
  btnPrimaryDisabled: {
    opacity: 0.4,
    cursor:  "not-allowed",
  },
  sendingNote: {
    margin:   0,
    fontSize: 12,
    color:    "var(--color-text-tertiary, #94A3B8)",
  },
};

// ─── Injected CSS (dark mode, scrollbar, hover states) ───────────────────────

if (typeof document !== "undefined") {
  const STYLES = [
    ".tsi-root { --tpv-blue: #103076; --tpv-blue-bg: rgba(16,48,118,0.03); }",

    ".dark .tsi-root {",
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

    ".dark .tsi-root .tsi-stops-direct { background: rgba(22,163,74,0.15)  !important; color: #4ADE80 !important; }",
    ".dark .tsi-root .tsi-stops-amber  { background: rgba(217,119,6,0.15)  !important; color: #FCD34D !important; }",
    ".dark .tsi-root .tsi-flight-num   { background: rgba(255,255,255,0.07) !important; color: #4B5563 !important; }",

    ".tsi-hotel-scroll::-webkit-scrollbar              { height: 3px; }",
    ".tsi-hotel-scroll::-webkit-scrollbar-track        { background: transparent; }",
    ".tsi-hotel-scroll::-webkit-scrollbar-thumb        { background: rgba(0,0,0,0.14); border-radius: 2px; }",
    ".dark .tsi-hotel-scroll::-webkit-scrollbar-thumb  { background: rgba(255,255,255,0.12); }",
  ].join("\n");

  const existing = document.getElementById("tsi-styles");
  if (existing) {
    existing.textContent = STYLES;
  } else {
    const style = document.createElement("style");
    style.id          = "tsi-styles";
    style.textContent = STYLES;
    document.head.appendChild(style);
  }
}
