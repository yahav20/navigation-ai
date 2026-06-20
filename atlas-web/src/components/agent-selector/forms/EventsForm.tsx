"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  Ticket,
  MapPin,
  Calendar,
  ChevronDown,
} from "lucide-react";
import { inputCls, labelCls, FormSubmitButton } from "../shared";

export function EventsForm({ onSubmit }: { onSubmit: (text: string) => void }) {
  const [artist, setArtist] = useState("");
  const [city, setCity] = useState("");
  const [month, setMonth] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});

  const locationRe = /^[a-zA-ZÀ-ɏ\s-]+$/;
  const canSubmit = !!(artist.trim() || city.trim() || month);

  const handleSubmit = () => {
    if (!canSubmit) return;
    const errs: Record<string, string> = {};
    if (city.trim() && !locationRe.test(city.trim())) {
      errs.city = "Please enter a valid location (letters only)";
    }
    if (Object.keys(errs).length > 0) { setErrors(errs); return; }
    setErrors({});
    const base = artist.trim() ? `Find me shows of ${artist.trim()}` : "Find me shows";
    const parts = [
      base,
      city.trim() && `in ${city.trim()}`,
      month && `in ${month}`,
    ].filter(Boolean);
    onSubmit(parts.join(" ") + ".");
  };

  const currentMonth = new Date().toLocaleString("en-US", { month: "long" });

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className={labelCls}>
            <Ticket className="mb-0.5 mr-1 inline size-3" />
            Artist / Show
          </label>
          <input className={inputCls} placeholder="Lady Gaga, Taylor Swift..." value={artist}
            onChange={(e) => setArtist(e.target.value)} dir="ltr" />
        </div>
        <div>
          <label className={labelCls}>
            <MapPin className="mb-0.5 mr-1 inline size-3" />
            City
          </label>
          <input className={cn(inputCls, errors.city && "border-red-500")} placeholder="London, New York..." value={city}
            onChange={(e) => { setCity(e.target.value); if (e.target.value.trim() && locationRe.test(e.target.value.trim())) setErrors({}); }} dir="ltr" />
          {errors.city && <p className="mt-1 text-xs text-red-500">{errors.city}</p>}
        </div>
        <div className="col-span-2">
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
      </div>

      <div className="flex flex-wrap gap-2">
        <button type="button"
          onClick={() => { setArtist("Lady Gaga"); setCity("London"); setErrors({}); }}
          className="rounded-lg bg-[--accent] px-2.5 py-1 text-xs font-medium text-[--foreground] transition-colors duration-150 hover:bg-[--accent]/80">
          🎤 Artist in city
        </button>
        <button type="button"
          onClick={() => setMonth(currentMonth)}
          className="rounded-lg bg-[--accent] px-2.5 py-1 text-xs font-medium text-[--foreground] transition-colors duration-150 hover:bg-[--accent]/80">
          📅 Events this month
        </button>
        <button type="button"
          onClick={() => { setCity("near me"); setErrors({}); }}
          className="rounded-lg bg-[--accent] px-2.5 py-1 text-xs font-medium text-[--foreground] transition-colors duration-150 hover:bg-[--accent]/80">
          🌍 Any shows near me
        </button>
      </div>

      <div className="flex justify-end pt-1">
        <FormSubmitButton onClick={handleSubmit} disabled={!canSubmit}
          label="Find Events" icon={<Ticket className="size-4" />} />
      </div>
    </div>
  );
}
