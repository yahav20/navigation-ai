"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  Ticket,
  MapPin,
  Calendar,
  ChevronDown,
  Music,
} from "lucide-react";
import { inputCls, labelCls, FormSubmitButton } from "../shared";

export function EventsForm({ onSubmit }: { onSubmit: (text: string) => void }) {
  const [artist, setArtist] = useState("");
  const [city,   setCity]   = useState("");
  const [genre,  setGenre]  = useState("");
  const [month,  setMonth]  = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});

  const locationRe = /^[a-zA-ZÀ-ɏ\s-]+$/;

  const a = artist.trim();
  const c = city.trim();
  const g = genre.trim();
  const m = month;

  const canSubmit = !!(a || c || m);

  const handleSubmit = () => {
    if (!canSubmit) return;
    const errs: Record<string, string> = {};
    if (c && !locationRe.test(c)) {
      errs.city = "Please enter a valid location (letters only)";
    }
    if (Object.keys(errs).length > 0) { setErrors(errs); return; }
    setErrors({});

    let prompt: string;

    if (g) {
      // Genre-centric: prepend genre naturally, then weave in the other fields
      const parts = [`Find ${g} concerts`];
      if (a) parts.push(`featuring ${a}`);
      if (c) parts.push(`in ${c}`);
      if (m) parts.push(`in ${m}`);
      prompt = parts.join(" ") + ".";
    } else if (a && c && m) {
      prompt = `Is ${a} performing in ${c} in ${m}? Find all matching shows.`;
    } else if (a && c) {
      prompt = `When does ${a} perform in ${c}? Show all upcoming dates.`;
    } else if (a && m) {
      prompt = `Where is ${a} performing in ${m}? List all cities and dates.`;
    } else if (c && m) {
      prompt = `What concerts and shows are happening in ${c} in ${m}?`;
    } else if (a) {
      prompt = `Find all upcoming tour dates for ${a}.`;
    } else {
      // city-only or month-only fallback
      const parts = ["Find upcoming concerts and shows"];
      if (c) parts.push(`in ${c}`);
      if (m) parts.push(`in ${m}`);
      prompt = parts.join(" ") + ".";
    }

    onSubmit(prompt);
  };

  const now = new Date();
  const thisMonthIdx = now.getMonth(); // 0-indexed
  const nextMonthDate = new Date(now.getFullYear(), thisMonthIdx + 1, 1);
  const nextMonthName = nextMonthDate.toLocaleString("en-US", { month: "long" });

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-3">

        {/* Artist */}
        <div>
          <label className={labelCls}>
            <Ticket className="mb-0.5 mr-1 inline size-3" />
            Artist / Show
          </label>
          <input
            className={inputCls}
            placeholder="Lady Gaga, Taylor Swift..."
            value={artist}
            onChange={(e) => setArtist(e.target.value)}
            dir="ltr"
          />
        </div>

        {/* City */}
        <div>
          <label className={labelCls}>
            <MapPin className="mb-0.5 mr-1 inline size-3" />
            City
          </label>
          <input
            className={cn(inputCls, errors.city && "border-red-500")}
            placeholder="London, New York..."
            value={city}
            onChange={(e) => {
              setCity(e.target.value);
              if (e.target.value.trim() && locationRe.test(e.target.value.trim())) setErrors({});
            }}
            dir="ltr"
          />
          {errors.city && <p className="mt-1 text-xs text-red-500">{errors.city}</p>}
        </div>

        {/* Genre — between City and Month */}
        <div className="col-span-2">
          <label className={labelCls}>
            <Music className="mb-0.5 mr-1 inline size-3" />
            Genre
          </label>
          <input
            className={inputCls}
            placeholder="Rock, Jazz, Electronic..."
            value={genre}
            onChange={(e) => setGenre(e.target.value)}
            dir="ltr"
          />
        </div>

        {/* Month */}
        <div className="col-span-2">
          <label className={labelCls}>
            <Calendar className="mb-0.5 mr-1 inline size-3" />
            Month
          </label>
          <div className="relative">
            <select
              className={cn(inputCls, "appearance-none pr-8")}
              value={month}
              onChange={(e) => setMonth(e.target.value)}
              dir="ltr"
            >
              <option value="">Select month...</option>
              {["January","February","March","April","May","June",
                "July","August","September","October","November","December"].map((mo, idx) => (
                <option key={mo} value={mo} disabled={idx <= thisMonthIdx}>{mo}</option>
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
          onClick={() => setMonth(nextMonthName)}
          className="rounded-lg bg-[--accent] px-2.5 py-1 text-xs font-medium text-[--foreground] transition-colors duration-150 hover:bg-[--accent]/80">
          📅 Events next month
        </button>
        <button type="button"
          onClick={() => { setCity("near me"); setErrors({}); }}
          className="rounded-lg bg-[--accent] px-2.5 py-1 text-xs font-medium text-[--foreground] transition-colors duration-150 hover:bg-[--accent]/80">
          🌍 Any shows near me
        </button>
      </div>

      <div className="flex justify-end pt-1">
        <FormSubmitButton
          onClick={handleSubmit}
          disabled={!canSubmit}
          label="Find Events"
          icon={<Ticket className="size-4" />}
        />
      </div>
    </div>
  );
}
