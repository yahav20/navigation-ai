/**
 * Type guard + types for the travel-selection HITL interrupt emitted by
 * plan_check.py when a travel plan (flights + hotels) already exists.
 *
 *   interrupt({
 *     "type": "travel_selection",
 *     "question": "...",
 *     "flight_pairings": [...],
 *     "hotels": [...],
 *   })
 *
 * Resume value: "flight:N,hotel:M" or "auto" (cheapest).
 */

export interface TravelSelectionFlightLeg {
  airline?: string;
  label?: string;
  price?: number;
  stops?: number | null;
  duration_minutes?: number | null;
  departure_time?: string;
  destination_airport?: string;
}

export interface TravelSelectionFlightPairing {
  total_price: number;
  description?: string;
  outbound: TravelSelectionFlightLeg;
  return_flight: TravelSelectionFlightLeg;
}

export interface TravelSelectionHotel {
  name: string;
  stars?: number | null;
  price_per_night: number;
  description?: string;
}

export interface TravelSelectionInterrupt {
  type: "travel_selection";
  question: string;
  flight_pairings: TravelSelectionFlightPairing[];
  hotels: TravelSelectionHotel[];
}

function unwrapInterruptValue(value: unknown): unknown {
  const candidate = Array.isArray(value) ? value[0] : value;
  if (candidate && typeof candidate === "object" && "value" in candidate) {
    return (candidate as { value: unknown }).value;
  }
  return candidate;
}

export function getTravelSelectionInterrupt(
  value: unknown,
): TravelSelectionInterrupt | null {
  const payload = unwrapInterruptValue(value);
  if (!payload || typeof payload !== "object") return null;

  const p = payload as Partial<TravelSelectionInterrupt>;
  if (p.type !== "travel_selection") return null;
  if (typeof p.question !== "string") return null;
  if (!Array.isArray(p.flight_pairings) || !Array.isArray(p.hotels)) return null;

  return p as TravelSelectionInterrupt;
}
