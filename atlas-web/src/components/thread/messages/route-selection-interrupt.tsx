/**
 * RouteSelectionInterruptView
 *
 * Rich HITL card selector rendered when multi_dest.py (RouteSelectNode) emits a
 * { type: "route_selection", routes } interrupt for a long, multi-city trip.
 *
 * The user picks one of the proposed road-trip routes, then clicks
 * "Build this route". They can also click "Choose for me" to accept the first
 * (recommended) route (resume value: "auto").
 *
 * Resume value format: "route:N" or "auto".
 */

import { useState, type CSSProperties, type FC } from "react";
import { motion } from "framer-motion";
import { useStreamContext } from "@/providers/Stream";
import { MarkdownText } from "../markdown-text";
import type {
  RouteSelectionInterrupt,
  RouteSelectionOption,
} from "@/lib/route-selection-interrupt";

// ─── Route card ───────────────────────────────────────────────────────────────

const RouteCard: FC<{
  route: RouteSelectionOption;
  index: number;
  isSelected: boolean;
  disabled: boolean;
  onSelect: (i: number) => void;
}> = ({ route, index, isSelected, disabled, onSelect }) => (
  <motion.div
    whileTap={disabled ? {} : { scale: 0.995 }}
    onClick={() => !disabled && onSelect(index)}
    style={{
      ...S.card,
      ...(isSelected ? S.cardSelected : {}),
      cursor: disabled ? "default" : "pointer",
      opacity: disabled ? 0.7 : 1,
    }}
  >
    <div style={S.cardHeader}>
      <span style={S.cardLabel}>{route.label || `Option ${index + 1}`}</span>
      <span style={S.cardMeta}>
        {route.segments.length} cities · {route.total_days} days
      </span>
    </div>
    <div style={S.cardDivider} />

    <div style={S.routeRow}>
      {route.segments.map((seg, i) => (
        <span key={i} style={S.hop}>
          {i > 0 && <span style={S.hopArrow}>→</span>}
          <span style={S.hopCity}>
            <span style={S.hopIcon}>{i === 0 ? "✈️" : "🚗"}</span>
            {seg.destination}
            <span style={S.hopDays}>{seg.days}d</span>
          </span>
        </span>
      ))}
      <span style={S.hop}>
        <span style={S.hopArrow}>→</span>
        <span style={{ ...S.hopCity, ...S.hopReturn }}>
          <span style={S.hopIcon}>🔙</span>
          {route.segments[0]?.destination}
        </span>
      </span>
    </div>

    {isSelected && <div style={S.selectedRing} />}
  </motion.div>
);

// ─── Main component ───────────────────────────────────────────────────────────

export function RouteSelectionInterruptView({
  interrupt,
}: {
  interrupt: RouteSelectionInterrupt;
}) {
  const thread = useStreamContext();
  const [selected, setSelected] = useState<number | null>(null);
  const [submitted, setSubmitted] = useState<string | null>(null);

  const isLoading = thread.isLoading;
  const isDisabled = isLoading || submitted !== null;

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
    if (selected === null) return;
    resume(`route:${selected}`);
  };

  const canConfirm = selected !== null;

  return (
    <div className="rsi-root" style={S.root}>
      <div style={S.header}>
        <h3 style={S.headerTitle}>Your input is needed</h3>
      </div>

      <div style={S.body}>
        <div style={S.question}>
          <MarkdownText>{interrupt.question}</MarkdownText>
        </div>

        <div style={S.list}>
          {interrupt.routes.map((route, i) => (
            <RouteCard
              key={i}
              route={route}
              index={i}
              isSelected={selected === i}
              disabled={isDisabled}
              onSelect={setSelected}
            />
          ))}
        </div>

        <div style={S.actions}>
          <motion.button
            whileTap={isDisabled ? {} : { scale: 0.98 }}
            style={{ ...S.btnSecondary, ...(isDisabled ? { opacity: 0.5 } : {}) }}
            disabled={isDisabled}
            onClick={() => resume("auto")}
          >
            Choose for me
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
            Build this route
          </motion.button>
        </div>

        {submitted !== null && (
          <p style={S.sendingNote}>Building your itinerary…</p>
        )}
      </div>
    </div>
  );
}

// ─── Styles (mirrors TravelSelectionInterruptView) ────────────────────────────

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
  list: {
    display:       "flex",
    flexDirection: "column",
    gap:           8,
  },
  card: {
    position:      "relative",
    background:    "var(--color-background-primary, #ffffff)",
    borderWidth:   "1.5px",
    borderStyle:   "solid",
    borderColor:   "var(--color-border-tertiary, #E2E8F0)",
    borderRadius:  10,
    padding:       "12px 14px",
    display:       "flex",
    flexDirection: "column",
    transition:    "border-color 0.15s",
  },
  cardSelected: {
    borderColor: "var(--rsi-blue, #103076)",
    background:  "var(--rsi-blue-bg, rgba(16,48,118,0.03))",
  },
  selectedRing: {
    position:      "absolute",
    inset:         0,
    borderRadius:  10,
    border:        "2px solid var(--rsi-blue, #103076)",
    pointerEvents: "none" as const,
  },
  cardHeader: {
    display:        "flex",
    justifyContent: "space-between",
    alignItems:     "center",
    marginBottom:   8,
  },
  cardLabel: {
    fontSize:   13,
    fontWeight: 700,
    color:      "var(--color-text-primary, #0F172A)",
  },
  cardMeta: {
    fontSize:           12,
    fontWeight:         600,
    color:              "var(--rsi-blue, #103076)",
    fontVariantNumeric: "tabular-nums" as any,
  },
  cardDivider: {
    height:       1,
    background:   "var(--color-border-tertiary, #F1F5F9)",
    marginBottom: 10,
  },
  routeRow: {
    display:    "flex",
    flexWrap:   "wrap" as any,
    alignItems: "center",
    gap:        6,
  },
  hop: {
    display:    "inline-flex",
    alignItems: "center",
    gap:        6,
  },
  hopArrow: {
    fontSize: 13,
    color:    "var(--color-text-tertiary, #94A3B8)",
  },
  hopCity: {
    display:      "inline-flex",
    alignItems:   "center",
    gap:          5,
    fontSize:     13,
    fontWeight:   600,
    color:        "var(--color-text-primary, #0F172A)",
    background:   "var(--color-background-secondary, #F1F5F9)",
    borderRadius: 6,
    padding:      "4px 8px",
  },
  hopReturn: {
    fontWeight: 500,
    color:      "var(--color-text-secondary, #475569)",
  },
  hopIcon: {
    fontSize: 12,
  },
  hopDays: {
    fontSize:           11,
    fontWeight:         700,
    color:              "var(--rsi-blue, #103076)",
    fontVariantNumeric: "tabular-nums" as any,
  },
  actions: {
    display:   "flex",
    gap:       10,
    flexWrap:  "wrap" as any,
    marginTop: 4,
  },
  btnSecondary: {
    flex:          "1 1 auto",
    height:        40,
    fontSize:      13,
    fontWeight:    600,
    color:         "var(--rsi-blue, #103076)",
    background:    "transparent",
    border:        "1.5px solid var(--rsi-blue, #103076)",
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
    background:    "var(--rsi-blue, #103076)",
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

// ─── Injected CSS (dark mode) ─────────────────────────────────────────────────

if (typeof document !== "undefined") {
  const STYLES = [
    ".rsi-root { --rsi-blue: #103076; --rsi-blue-bg: rgba(16,48,118,0.03); }",
    ".dark .rsi-root {",
    "  --rsi-blue:        #3B82F6;",
    "  --rsi-blue-bg:     rgba(59,130,246,0.08);",
    "  --color-background-primary:    #111827;",
    "  --color-background-secondary:  #1F2937;",
    "  --color-text-primary:    #F1F5F9;",
    "  --color-text-secondary:  #94A3B8;",
    "  --color-text-tertiary:   #4B5563;",
    "  --color-border-tertiary:  rgba(255,255,255,0.08);",
    "}",
  ].join("\n");

  const existing = document.getElementById("rsi-styles");
  if (existing) {
    existing.textContent = STYLES;
  } else {
    const style = document.createElement("style");
    style.id          = "rsi-styles";
    style.textContent = STYLES;
    document.head.appendChild(style);
  }
}
