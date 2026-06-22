"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  MapPin,
  Globe,
  Calendar,
  DollarSign,
  ChevronDown,
} from "lucide-react";
import { inputCls, labelCls, TravelersRow, FormSubmitButton } from "../shared";

export function TripPlannerForm({ onSubmit }: { onSubmit: (text: string) => void }) {
  const [origin, setOrigin] = useState("");
  const [destination, setDestination] = useState("");
  const [days, setDays] = useState("");
  const [budget, setBudget] = useState("");
  const [month, setMonth] = useState("");
  const [year, setYear] = useState("");
  const [adults, setAdults] = useState("1");
  const [children, setChildren] = useState("0");
  const [rooms, setRooms] = useState("1");
  const [errors, setErrors] = useState<Record<string, string>>({});

  const clearError = (key: string) =>
    setErrors((p) => { const n = { ...p }; delete n[key]; return n; });

  const locationRe = /^[a-zA-ZÀ-ɏ\s-]+$/;

  const handleSubmit = () => {
    const errs: Record<string, string> = {};
    if (!origin.trim()) {
      errs.origin = "Required";
    } else if (!locationRe.test(origin.trim())) {
      errs.origin = "Please enter a valid location (letters only)";
    }
    if (!destination.trim()) {
      errs.destination = "Required";
    } else if (!locationRe.test(destination.trim())) {
      errs.destination = "Please enter a valid location (letters only)";
    }
    const numDays = parseInt(days);
    if (!days || isNaN(numDays) || numDays < 1) errs.days = "Must be at least 1";
    if (budget.trim()) {
      const numBudget = parseFloat(budget.replace(/[^0-9.]/g, ""));
      if (isNaN(numBudget) || numBudget <= 0) errs.budget = "Must be a positive number";
    }
    if (year.trim()) {
      const numYear = parseInt(year);
      if (isNaN(numYear) || numYear < 2025 || numYear > 2030) errs.year = "Please enter a valid year (2025–2030)";
    }
    if (Object.keys(errs).length > 0) { setErrors(errs); return; }
    setErrors({});
    const numAdults   = parseInt(adults)   || 1;
    const numChildren = parseInt(children) || 0;
    const numRooms    = parseInt(rooms)    || 1;
    const travelers =
      numAdults === 1 && numChildren === 0
        ? "1 adult"
        : `${numAdults} adult${numAdults > 1 ? "s" : ""}` +
          (numChildren > 0 ? ` and ${numChildren} child${numChildren > 1 ? "ren" : ""}` : "");
    const effectiveBudget = budget.trim() || "$5,000";
    const effectiveMonth  = month || new Date().toLocaleString("en", { month: "long" });
    const effectiveYear   = year.trim() ? parseInt(year) : new Date().getFullYear();

    const parts = [
      `Build a ${days}-day itinerary from ${origin} to ${destination}`,
      `for ${travelers}`,
      numRooms > 1 ? `in ${numRooms} rooms` : "",
      `with a budget of ${effectiveBudget}`,
      `in ${effectiveMonth} ${effectiveYear}`,
    ].filter(Boolean);
    onSubmit(parts.join(", ") + ".");
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className={labelCls}>
            <Globe className="mb-0.5 mr-1 inline size-3" />
            Origin country<span className="text-[--ring] ml-0.5">*</span>
          </label>
          <input className={cn(inputCls, errors.origin && "border-red-500")} placeholder="Israel" value={origin}
            onChange={(e) => { setOrigin(e.target.value); if (e.target.value.trim() && locationRe.test(e.target.value.trim())) clearError("origin"); }} dir="ltr" />
          {errors.origin && <p className="mt-1 text-xs text-red-500">{errors.origin}</p>}
        </div>
        <div>
          <label className={labelCls}>
            <MapPin className="mb-0.5 mr-1 inline size-3" />
            Destination country<span className="text-[--ring] ml-0.5">*</span>
          </label>
          <input className={cn(inputCls, errors.destination && "border-red-500")} placeholder="Italy" value={destination}
            onChange={(e) => { setDestination(e.target.value); if (e.target.value.trim() && locationRe.test(e.target.value.trim())) clearError("destination"); }} dir="ltr" />
          {errors.destination && <p className="mt-1 text-xs text-red-500">{errors.destination}</p>}
        </div>
        <div>
          <label className={labelCls}>
            <Calendar className="mb-0.5 mr-1 inline size-3" />
            Number of days<span className="text-[--ring] ml-0.5">*</span>
          </label>
          <input className={cn(inputCls, errors.days && "border-red-500")} type="number" placeholder="7" min={1}
            value={days} onChange={(e) => { setDays(e.target.value); if (parseInt(e.target.value) >= 1) clearError("days"); }}
            onWheel={(e) => e.currentTarget.blur()} dir="ltr" />
          {errors.days && <p className="mt-1 text-xs text-red-500">{errors.days}</p>}
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className={labelCls}>
              <Calendar className="mb-0.5 mr-1 inline size-3" />
              Month
            </label>
            <div className="relative">
              <select className={cn(inputCls, "appearance-none pr-8")} value={month} onChange={(e) => setMonth(e.target.value)} dir="ltr">
                <option value="">Select month...</option>
                {["January","February","March","April","May","June",
                  "July","August","September","October","November","December"].map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 size-3.5 -translate-y-1/2 text-[--muted-foreground]" />
            </div>
          </div>
          <div>
            <label className={labelCls}>
              <Calendar className="mb-0.5 mr-1 inline size-3" />
              Year
            </label>
            <input className={cn(inputCls, errors.year && "border-red-500")} type="number" placeholder="2026" min={2025} max={2030}
              value={year} onChange={(e) => { setYear(e.target.value); const y = parseInt(e.target.value); if (y >= 2025 && y <= 2030) clearError("year"); }}
              onWheel={(e) => e.currentTarget.blur()} dir="ltr" />
            {errors.year && <p className="mt-1 text-xs text-red-500">{errors.year}</p>}
          </div>
        </div>
        <div className="col-span-2">
          <label className={labelCls}>
            <DollarSign className="mb-0.5 mr-1 inline size-3" />
            Estimated budget (optional)
          </label>
          <input className={cn(inputCls, errors.budget && "border-red-500")} placeholder="$5,000" value={budget}
            onChange={(e) => { setBudget(e.target.value); if (errors.budget) clearError("budget"); }} dir="ltr" />
          {errors.budget && <p className="mt-1 text-xs text-red-500">{errors.budget}</p>}
        </div>
      </div>

      {/* Travelers */}
      <div>
        <p className="text-xs font-medium text-[--muted-foreground] mb-2">Travelers</p>
        <TravelersRow
          adults={adults} setAdults={setAdults}
          children={children} setChildren={setChildren}
          rooms={rooms} setRooms={setRooms}
        />
      </div>

      <div className="flex justify-end pt-1">
        <FormSubmitButton onClick={handleSubmit} disabled={false}
          label="Plan my trip" icon={<MapPin className="size-4" />} />
      </div>
    </div>
  );
}
