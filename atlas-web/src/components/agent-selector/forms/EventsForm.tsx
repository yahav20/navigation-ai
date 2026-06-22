"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  Ticket,
  MapPin,
  Calendar,
  ChevronDown,
} from "lucide-react";
import { inputCls, labelCls, FormSubmitButton } from "../shared";

export function EventsForm({ onSubmit }: { onSubmit: (text: string) => void }) {
  const [mode, setMode] = useState<"concerts" | "special">("concerts");

  // Concerts fields
  const [artist, setArtist] = useState("");
  const [concertCity, setConcertCity] = useState("");
  const [concertMonth, setConcertMonth] = useState("");

  // Special events fields
  const [eventCity, setEventCity] = useState("");
  const [eventMonth, setEventMonth] = useState("");

  const [errors, setErrors] = useState<Record<string, string>>({});
  const locationRe = /^[a-zA-ZÀ-ɏ\s-]+$/;

  const canSubmitConcerts = !!(artist.trim() || concertCity.trim() || concertMonth);
  const canSubmitSpecial  = !!(eventCity.trim() || eventMonth);

  const handleSubmit = () => {
    const errs: Record<string, string> = {};

    if (mode === "concerts") {
      if (!canSubmitConcerts) return;
      if (concertCity.trim() && !locationRe.test(concertCity.trim())) {
        errs.concertCity = "Please enter a valid location (letters only)";
      }
      if (Object.keys(errs).length > 0) { setErrors(errs); return; }
      setErrors({});
      const base = artist.trim() ? `Find me shows of ${artist.trim()}` : "Find me shows";
      const parts = [
        base,
        concertCity.trim() && `in ${concertCity.trim()}`,
        concertMonth && `in ${concertMonth}`,
      ].filter(Boolean);
      onSubmit(parts.join(" ") + ".");
    } else {
      if (!canSubmitSpecial) return;
      if (eventCity.trim() && !locationRe.test(eventCity.trim())) {
        errs.eventCity = "Please enter a valid location (letters only)";
      }
      if (Object.keys(errs).length > 0) { setErrors(errs); return; }
      setErrors({});
      const parts = [
        eventCity.trim()
          ? `What special events and festivals are happening in ${eventCity.trim()}`
          : "What special events and festivals are happening",
        eventMonth && `in ${eventMonth}`,
      ].filter(Boolean);
      onSubmit(parts.join(" ") + "?");
    }
  };

  const currentMonth = new Date().toLocaleString("en-US", { month: "long" });

  const toggleCls = (active: boolean) =>
    cn(
      "flex-1 rounded-lg py-1.5 text-xs font-medium transition-all duration-200",
      active
        ? "bg-[--primary] text-[--primary-foreground] shadow-sm"
        : "text-[--muted-foreground] hover:text-[--foreground]",
    );

  return (
    <div className="flex flex-col gap-3">
      {/* Mode toggle */}
      <div className="flex gap-1 rounded-xl bg-[--accent]/60 p-1">
        <button type="button" className={toggleCls(mode === "concerts")}
          onClick={() => { setMode("concerts"); setErrors({}); }}>
          🎤 Concerts &amp; Shows
        </button>
        <button type="button" className={toggleCls(mode === "special")}
          onClick={() => { setMode("special"); setErrors({}); }}>
          🎪 Special Events
        </button>
      </div>

      <AnimatePresence mode="wait">
        {mode === "concerts" ? (
          <motion.div key="concerts" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.18 }}
            className="flex flex-col gap-3">
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
                <input className={cn(inputCls, errors.concertCity && "border-red-500")}
                  placeholder="London, New York..." value={concertCity}
                  onChange={(e) => { setConcertCity(e.target.value); if (e.target.value.trim() && locationRe.test(e.target.value.trim())) setErrors({}); }}
                  dir="ltr" />
                {errors.concertCity && <p className="mt-1 text-xs text-red-500">{errors.concertCity}</p>}
              </div>
              <div className="col-span-2">
                <label className={labelCls}>
                  <Calendar className="mb-0.5 mr-1 inline size-3" />
                  Month
                </label>
                <div className="relative">
                  <select className={cn(inputCls, "appearance-none pr-8")} value={concertMonth}
                    onChange={(e) => setConcertMonth(e.target.value)} dir="ltr">
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
                onClick={() => { setArtist("Lady Gaga"); setConcertCity("London"); setErrors({}); }}
                className="rounded-lg bg-[--accent] px-2.5 py-1 text-xs font-medium text-[--foreground] transition-colors duration-150 hover:bg-[--accent]/80">
                🎤 Artist in city
              </button>
              <button type="button"
                onClick={() => setConcertMonth(currentMonth)}
                className="rounded-lg bg-[--accent] px-2.5 py-1 text-xs font-medium text-[--foreground] transition-colors duration-150 hover:bg-[--accent]/80">
                📅 This month
              </button>
            </div>
          </motion.div>
        ) : (
          <motion.div key="special" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.18 }}
            className="flex flex-col gap-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className={labelCls}>
                  <MapPin className="mb-0.5 mr-1 inline size-3" />
                  City<span className="text-[--ring] ml-0.5">*</span>
                </label>
                <input className={cn(inputCls, errors.eventCity && "border-red-500")}
                  placeholder="Berlin, Prague, Munich..." value={eventCity}
                  onChange={(e) => { setEventCity(e.target.value); if (e.target.value.trim() && locationRe.test(e.target.value.trim())) setErrors({}); }}
                  dir="ltr" />
                {errors.eventCity && <p className="mt-1 text-xs text-red-500">{errors.eventCity}</p>}
              </div>
              <div>
                <label className={labelCls}>
                  <Calendar className="mb-0.5 mr-1 inline size-3" />
                  Month
                </label>
                <div className="relative">
                  <select className={cn(inputCls, "appearance-none pr-8")} value={eventMonth}
                    onChange={(e) => setEventMonth(e.target.value)} dir="ltr">
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
                onClick={() => { setEventCity("Berlin"); setEventMonth("December"); setErrors({}); }}
                className="rounded-lg bg-[--accent] px-2.5 py-1 text-xs font-medium text-[--foreground] transition-colors duration-150 hover:bg-[--accent]/80">
                🎄 Christmas Markets
              </button>
              <button type="button"
                onClick={() => { setEventCity("Munich"); setEventMonth("October"); setErrors({}); }}
                className="rounded-lg bg-[--accent] px-2.5 py-1 text-xs font-medium text-[--foreground] transition-colors duration-150 hover:bg-[--accent]/80">
                🍺 Oktoberfest
              </button>
              <button type="button"
                onClick={() => { setEventCity("Amsterdam"); setEventMonth("April"); setErrors({}); }}
                className="rounded-lg bg-[--accent] px-2.5 py-1 text-xs font-medium text-[--foreground] transition-colors duration-150 hover:bg-[--accent]/80">
                🌷 King's Day
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex justify-end pt-1">
        <FormSubmitButton
          onClick={handleSubmit}
          disabled={mode === "concerts" ? !canSubmitConcerts : !canSubmitSpecial}
          label={mode === "concerts" ? "Find Shows" : "Find Events"}
          icon={<Ticket className="size-4" />}
        />
      </div>
    </div>
  );
}
