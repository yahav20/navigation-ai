"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  Plane,
  Globe,
  Calendar,
  DollarSign,
  ChevronDown,
} from "lucide-react";
import { inputCls, labelCls, TravelersRow, FormSubmitButton } from "../shared";

const MONTHS = [
  "January","February","March","April","May","June",
  "July","August","September","October","November","December",
];

export function FlightFinderForm({ onSubmit }: { onSubmit: (text: string) => void }) {
  const [origin, setOrigin] = useState("");
  const [destination, setDestination] = useState("");
  const [month, setMonth] = useState("");
  const [year, setYear] = useState("");
  const [budget, setBudget] = useState("");
  const [days, setDays] = useState("");
  const [adults, setAdults] = useState("1");
  const [children, setChildren] = useState("0");
  const [errors, setErrors] = useState<Record<string, string>>({});

  const clearError = (key: string) =>
    setErrors((p) => { const n = { ...p }; delete n[key]; return n; });

  const now = new Date();
  const thisYear = now.getFullYear();
  const thisMonthIdx = now.getMonth(); // 0-indexed

  // Returns true when a month (0-indexed) is past or current given the selected year.
  const isMonthDisabled = (monthIdx: number): boolean => {
    const yr = parseInt(year);
    if (!year.trim() || isNaN(yr)) return monthIdx <= thisMonthIdx;
    if (yr < thisYear) return true;
    if (yr === thisYear) return monthIdx <= thisMonthIdx;
    return false;
  };

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
    if (days.trim()) {
      const numDays = parseInt(days);
      if (isNaN(numDays) || numDays < 1) errs.days = "Must be at least 1";
    }
    if (year.trim()) {
      const numYear = parseInt(year);
      if (isNaN(numYear) || numYear < thisYear || numYear > 2030)
        errs.year = `Please enter a valid year (${thisYear}–2030)`;
    }
    // Validate that the month/year combination is strictly in the future.
    if (!errs.year) {
      const nextMonthDate = new Date(thisYear, thisMonthIdx + 1, 1);
      const resolvedMonthIdx = month ? MONTHS.indexOf(month) : nextMonthDate.getMonth();
      const resolvedYear = year.trim() ? parseInt(year) : nextMonthDate.getFullYear();
      const isPastOrCurrent =
        resolvedYear < thisYear ||
        (resolvedYear === thisYear && resolvedMonthIdx <= thisMonthIdx);
      if (isPastOrCurrent) errs.month = "Please select a future month";
    }
    if (Object.keys(errs).length > 0) { setErrors(errs); return; }
    setErrors({});

    const numAdults   = parseInt(adults)   || 1;
    const numChildren = parseInt(children) || 0;
    const travelers =
      numAdults === 1 && numChildren === 0
        ? "1 adult"
        : `${numAdults} adult${numAdults > 1 ? "s" : ""}` +
          (numChildren > 0 ? ` and ${numChildren} child${numChildren > 1 ? "ren" : ""}` : "");
    const effectiveBudget = budget.trim() || "$5,000";
    // Default to next month so we never send a past/current date.
    const nextMonthDate = new Date(thisYear, thisMonthIdx + 1, 1);
    const effectiveMonth = month || MONTHS[nextMonthDate.getMonth()];
    const effectiveYear  = year.trim() ? parseInt(year) : nextMonthDate.getFullYear();

    const parts = [
      `I'm looking for flights from ${origin} to ${destination}`,
      `in ${effectiveMonth} ${effectiveYear}`,
      days && `for a ${days}-day trip`,
      `for ${travelers}`,
      `with a budget of ${effectiveBudget}`,
    ].filter(Boolean);
    onSubmit(parts.join(", ") + ".");
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className={labelCls}>
            <Globe className="mb-0.5 mr-1 inline size-3" />
            Origin city / airport<span className="text-[--ring] ml-0.5">*</span>
          </label>
          <input className={cn(inputCls, errors.origin && "border-red-500")} placeholder="Tel Aviv (TLV)" value={origin}
            onChange={(e) => { setOrigin(e.target.value); if (e.target.value.trim() && locationRe.test(e.target.value.trim())) clearError("origin"); }} dir="ltr" />
          {errors.origin && <p className="mt-1 text-xs text-red-500">{errors.origin}</p>}
        </div>
        <div>
          <label className={labelCls}>
            <Plane className="mb-0.5 mr-1 inline size-3" />
            Destination<span className="text-[--ring] ml-0.5">*</span>
          </label>
          <input className={cn(inputCls, errors.destination && "border-red-500")} placeholder="Rome (FCO)" value={destination}
            onChange={(e) => { setDestination(e.target.value); if (e.target.value.trim() && locationRe.test(e.target.value.trim())) clearError("destination"); }} dir="ltr" />
          {errors.destination && <p className="mt-1 text-xs text-red-500">{errors.destination}</p>}
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className={labelCls}>
              <Calendar className="mb-0.5 mr-1 inline size-3" />
              Month
            </label>
            <div className="relative">
              <select
                className={cn(inputCls, "appearance-none pr-8", errors.month && "border-red-500")}
                value={month}
                onChange={(e) => { setMonth(e.target.value); clearError("month"); }}
                dir="ltr"
              >
                <option value="">Select month...</option>
                {MONTHS.map((m, idx) => (
                  <option key={m} value={m} disabled={isMonthDisabled(idx)}>{m}</option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 size-3.5 -translate-y-1/2 text-[--muted-foreground]" />
            </div>
            {errors.month && <p className="mt-1 text-xs text-red-500">{errors.month}</p>}
          </div>
          <div>
            <label className={labelCls}>
              <Calendar className="mb-0.5 mr-1 inline size-3" />
              Year
            </label>
            <input
              className={cn(inputCls, errors.year && "border-red-500")}
              type="number"
              placeholder={String(thisYear + 1)}
              min={thisYear}
              max={2030}
              value={year}
              onChange={(e) => {
                const newYear = e.target.value;
                setYear(newYear);
                // Clear a month that becomes invalid when the year changes.
                const yr = parseInt(newYear);
                if (!isNaN(yr) && month) {
                  const monthIdx = MONTHS.indexOf(month);
                  if (yr < thisYear || (yr === thisYear && monthIdx <= thisMonthIdx)) {
                    setMonth("");
                  }
                }
                const y = parseInt(newYear);
                if (y >= thisYear && y <= 2030) clearError("year");
              }}
              onWheel={(e) => e.currentTarget.blur()}
              dir="ltr"
            />
            {errors.year && <p className="mt-1 text-xs text-red-500">{errors.year}</p>}
          </div>
        </div>
        <div>
          <label className={labelCls}>
            <Calendar className="mb-0.5 mr-1 inline size-3" />
            Trip duration (days)
          </label>
          <input className={cn(inputCls, errors.days && "border-red-500")} type="number" min={1} placeholder="7" value={days}
            onChange={(e) => { setDays(e.target.value); if (parseInt(e.target.value) >= 1) clearError("days"); }}
            onWheel={(e) => e.currentTarget.blur()} dir="ltr" />
          {errors.days && <p className="mt-1 text-xs text-red-500">{errors.days}</p>}
        </div>
        <div className="col-span-2">
          <label className={labelCls}>
            <DollarSign className="mb-0.5 mr-1 inline size-3" />
            Flight budget (optional)
          </label>
          <input className={inputCls} placeholder="Up to ₪2,000 one-way" value={budget}
            onChange={(e) => setBudget(e.target.value)} dir="ltr" />
        </div>
      </div>

      {/* Travelers */}
      <div>
        <p className="text-xs font-medium text-[--muted-foreground] mb-2">Travelers</p>
        <TravelersRow
          adults={adults} setAdults={setAdults}
          children={children} setChildren={setChildren}
          rooms="" setRooms={() => {}}
          showRooms={false}
        />
      </div>

      <div className="flex justify-end pt-1">
        <FormSubmitButton onClick={handleSubmit} disabled={false}
          label="Search flights" icon={<Plane className="size-4" />} />
      </div>
    </div>
  );
}
