/**
 * Type guard + types for the route-selection HITL interrupt emitted by
 * src/agent/itinerary/multi_dest.py (RouteSelectNode) when a long trip is split
 * into several candidate multi-city road-trip routes:
 *
 *   interrupt({
 *     "type": "route_selection",
 *     "question": "...",
 *     "anchor": "Rome",
 *     "total_days": 12,
 *     "routes": [
 *       { "label": "Option 1", "total_days": 12,
 *         "segments": [{ destination, days, drive_from_prev }, ...] },
 *       ...
 *     ],
 *   })
 *
 * Resume value: "route:N" (chosen index) or "auto" (first/recommended).
 */

export interface RouteSelectionSegment {
  destination: string;
  days: number;
  drive_from_prev?: string | null;
}

export interface RouteSelectionOption {
  label?: string;
  total_days: number;
  segments: RouteSelectionSegment[];
}

export interface RouteSelectionInterrupt {
  type: "route_selection";
  question: string;
  anchor: string;
  total_days: number;
  routes: RouteSelectionOption[];
}

function unwrapInterruptValue(value: unknown): unknown {
  const candidate = Array.isArray(value) ? value[0] : value;
  if (candidate && typeof candidate === "object" && "value" in candidate) {
    return (candidate as { value: unknown }).value;
  }
  return candidate;
}

function isSegment(value: unknown): value is RouteSelectionSegment {
  if (!value || typeof value !== "object") return false;
  const s = value as Partial<RouteSelectionSegment>;
  return typeof s.destination === "string" && typeof s.days === "number";
}

function isOption(value: unknown): value is RouteSelectionOption {
  if (!value || typeof value !== "object") return false;
  const o = value as Partial<RouteSelectionOption>;
  return (
    typeof o.total_days === "number" &&
    Array.isArray(o.segments) &&
    o.segments.length > 0 &&
    o.segments.every(isSegment)
  );
}

export function getRouteSelectionInterrupt(
  value: unknown,
): RouteSelectionInterrupt | null {
  const payload = unwrapInterruptValue(value);
  if (!payload || typeof payload !== "object") return null;

  const p = payload as Partial<RouteSelectionInterrupt>;
  if (p.type !== "route_selection") return null;
  if (typeof p.question !== "string") return null;
  if (!Array.isArray(p.routes) || !p.routes.every(isOption)) return null;

  return p as RouteSelectionInterrupt;
}
