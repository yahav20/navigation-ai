"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  MapPin,
  Plane,
  Ticket,
  Star,
  Calendar,
  DollarSign,
  Globe,
  Users,
  BedDouble,
  Baby,
  ChevronRight,
  ChevronDown,
  Check,
  X,
} from "lucide-react";

/* ─────────────────────────────────────────
   Types
───────────────────────────────────────── */
export type AgentType =
  | "trip_planner"
  | "flight_finder"
  | "events"
  | "recommendations"
  | null;

interface AgentCard {
  id: AgentType;
  label: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  hasForm: boolean;
}

const AGENTS: AgentCard[] = [
  {
    id: "trip_planner",
    label: "Full Trip Planning",
    description: "Give me the details and I will build the perfect trip for you",
    icon: <MapPin className="size-5" />,
    color: "oklch(0.55 0.22 240)",
    hasForm: true,
  },
  {
    id: "flight_finder",
    label: "Flight Finder",
    description: "I'll find you the best flights at the right price",
    icon: <Plane className="size-5" />,
    color: "oklch(0.62 0.2 195)",
    hasForm: true,
  },
  {
    id: "recommendations",
    label: "Recommendations",
    description: "Recommendations on destinations, restaurants, attractions, and more",
    icon: <Star className="size-5" />,
    color: "oklch(0.7 0.18 210)",
    hasForm: true,
  },
  {
    id: "events",
    label: "Shows & Events",
    description: "Find concerts, shows, festivals and special events worldwide",
    icon: <Ticket className="size-5" />,
    color: "oklch(0.58 0.22 320)",
    hasForm: true,
  },
];

/* ─────────────────────────────────────────
   Shared input styles
───────────────────────────────────────── */
const inputCls = cn(
  "w-full rounded-xl border border-[--border] bg-[--background]/60 px-3 py-2",
  "text-sm text-[--foreground] placeholder:text-[--muted-foreground]/60",
  "focus:border-[--ring] focus:outline-none focus:ring-0",
  "transition-colors duration-200",
);

const labelCls = "block text-xs font-medium text-[--muted-foreground] mb-1";

/* ─────────────────────────────────────────
   Travelers row (shared between forms)
───────────────────────────────────────── */
function TravelersRow({
  adults, setAdults,
  children, setChildren,
  rooms, setRooms,
  showRooms = true,
}: {
  adults: string; setAdults: (v: string) => void;
  children: string; setChildren: (v: string) => void;
  rooms: string; setRooms: (v: string) => void;
  showRooms?: boolean;
}) {
  return (
    <div className={cn("grid gap-3", showRooms ? "grid-cols-3" : "grid-cols-2")}>
      <div>
        <label className={labelCls}>
          <Users className="inline size-3 mr-1 mb-0.5" />
          Adults
        </label>
        <input
          className={inputCls}
          type="number"
          min={1}
          placeholder="2"
          value={adults}
          onChange={(e) => setAdults(e.target.value)}
          onWheel={(e) => e.currentTarget.blur()}
          dir="ltr"
        />
      </div>
      <div>
        <label className={labelCls}>
          <Baby className="inline size-3 mr-1 mb-0.5" />
          Children
        </label>
        <input
          className={inputCls}
          type="number"
          min={0}
          placeholder="0"
          value={children}
          onChange={(e) => setChildren(e.target.value)}
          onWheel={(e) => e.currentTarget.blur()}
          dir="ltr"
        />
      </div>
      {showRooms && (
        <div>
          <label className={labelCls}>
            <BedDouble className="inline size-3 mr-1 mb-0.5" />
            Rooms
          </label>
          <input
            className={inputCls}
            type="number"
            min={1}
            placeholder="1"
            value={rooms}
            onChange={(e) => setRooms(e.target.value)}
            onWheel={(e) => e.currentTarget.blur()}
            dir="ltr"
          />
        </div>
      )}
    </div>
  );
}

/* ─────────────────────────────────────────
   Trip Planner Form
───────────────────────────────────────── */
function TripPlannerForm({ onSubmit }: { onSubmit: (text: string) => void }) {
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

/* ─────────────────────────────────────────
   Flight Finder Form
───────────────────────────────────────── */
function FlightFinderForm({ onSubmit }: { onSubmit: (text: string) => void }) {
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
      if (isNaN(numYear) || numYear < 2025 || numYear > 2030) errs.year = "Please enter a valid year (2025–2030)";
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
    const effectiveMonth  = month || new Date().toLocaleString("en", { month: "long" });
    const effectiveYear   = year.trim() ? parseInt(year) : new Date().getFullYear();

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

/* ─────────────────────────────────────────
   Recommendations Form
───────────────────────────────────────── */
function RecommendationsForm({ onSubmit }: { onSubmit: (text: string) => void }) {
  const [origin, setOrigin] = useState("");
  const [month, setMonth] = useState("");
  const [topic, setTopic] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});

  const locationRe = /^[a-zA-ZÀ-ɏ\s-]+$/;

  const handleSubmit = () => {
    if (!origin.trim()) { setErrors({ origin: "Required" }); return; }
    if (!locationRe.test(origin.trim())) { setErrors({ origin: "Please enter a valid location (letters only)" }); return; }
    setErrors({});
    const effectiveMonth = month || new Date().toLocaleString("en", { month: "long" });
    const parts = [
      "I'm looking for travel recommendations",
      `from ${origin}`,
      `in ${effectiveMonth}`,
      topic && `about ${topic}`,
    ].filter(Boolean);
    onSubmit(parts.join(" ") + ".");
  };

  return (
    <div className="grid grid-cols-2 gap-3">
      <div>
        <label className={labelCls}>
          <Globe className="mb-0.5 mr-1 inline size-3" />
          Origin country<span className="text-[--ring] ml-0.5">*</span>
        </label>
        <input className={cn(inputCls, errors.origin && "border-red-500")} placeholder="Israel" value={origin}
          onChange={(e) => { setOrigin(e.target.value); if (e.target.value.trim() && locationRe.test(e.target.value.trim())) setErrors({}); }} dir="ltr" />
        {errors.origin && <p className="mt-1 text-xs text-red-500">{errors.origin}</p>}
      </div>
      <div>
        <label className={labelCls}>
          <Calendar className="mb-0.5 mr-1 inline size-3" />
          Estimated date
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
      <div className="col-span-2">
        <label className={labelCls}>
          <Star className="mb-0.5 mr-1 inline size-3" />
          What are you looking for? (Optional)
        </label>
        <input className={inputCls} placeholder="food, beaches, mountains, culture..."
          value={topic} onChange={(e) => setTopic(e.target.value)} dir="ltr" />
      </div>
      <div className="col-span-2 flex justify-end pt-1">
        <FormSubmitButton onClick={handleSubmit} disabled={false}
          label="Get Recommendations" icon={<Star className="size-4" />} />
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────
   Submit button
───────────────────────────────────────── */
function FormSubmitButton({
  onClick, disabled, label, icon,
}: {
  onClick: () => void;
  disabled: boolean;
  label: string;
  icon: React.ReactNode;
}) {
  const [phase, setPhase] = useState<"idle" | "loading" | "done">("idle");

  const handleClick = () => {
    if (phase !== "idle") return;
    onClick();
    setPhase("loading");
    setTimeout(() => {
      setPhase("done");
      setTimeout(() => setPhase("idle"), 600);
    }, 1500);
  };

  return (
    <motion.button
      onClick={handleClick}
      disabled={disabled}
      className={cn(
        "flex min-w-[9rem] items-center justify-center gap-2 rounded-xl px-5 py-2.5",
        "bg-[--primary] text-[--primary-foreground] text-sm font-medium",
        "shadow-[0_0_16px_var(--glow-primary)]",
        "hover:shadow-[0_0_28px_var(--glow-accent)]",
        "disabled:opacity-40 disabled:shadow-none disabled:cursor-not-allowed",
        "transition-all duration-300",
      )}
      whileHover={disabled ? {} : { scale: 1.03 }}
      whileTap={disabled ? {} : { scale: 0.97 }}
    >
      <AnimatePresence mode="wait" initial={false}>
        {phase === "idle" && (
          <motion.span key="idle" className="flex items-center gap-2"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}>
            {icon}{label}<ChevronRight className="size-4" />
          </motion.span>
        )}
        {phase === "loading" && (
          <motion.span key="loading" className="flex items-center gap-2"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}>
            <motion.span
              className="block size-4 rounded-full border-2 border-[--primary-foreground]/30 border-t-[--primary-foreground]"
              animate={{ rotate: 360 }}
              transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
            />
            Processing...
          </motion.span>
        )}
        {phase === "done" && (
          <motion.span key="done" className="flex items-center gap-2"
            initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}>
            <Check className="size-4" />
            Done!
          </motion.span>
        )}
      </AnimatePresence>
    </motion.button>
  );
}

/* ─────────────────────────────────────────
   Events Form  (Concerts + Special Events)
───────────────────────────────────────── */
function EventsForm({ onSubmit }: { onSubmit: (text: string) => void }) {
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

/* ─────────────────────────────────────────
   Agent Selector — main exported component
───────────────────────────────────────── */
export function AgentSelector({ onSubmit }: { onSubmit: (text: string) => void }) {
  const [selected, setSelected] = useState<AgentType>(null);
  const selectedAgent = AGENTS.find((a) => a.id === selected);

  return (
    <div className="w-full max-w-3xl mx-auto flex flex-col gap-4" dir="ltr">
      {/* Agent cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        {AGENTS.map((agent, i) => (
          <motion.button
            key={agent.id}
            onClick={() => setSelected(selected === agent.id ? null : agent.id)}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.07, duration: 0.35, ease: "easeOut" }}
            className={cn(
              "relative flex flex-col items-start gap-2 rounded-2xl border p-3.5 text-right",
              "cursor-pointer transition-all duration-250",
              selected === agent.id
                ? "border-[--ring] bg-[--accent] shadow-[0_0_0_2px_var(--ring),0_4px_24px_var(--glow-accent)]"
                : "border-[--border] bg-[--card]/70 hover:border-[--ring]/50 hover:bg-[--card]",
            )}
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
          >
            <div
              className="flex h-9 w-9 items-center justify-center rounded-xl text-white"
              style={{ background: agent.color, boxShadow: `0 0 12px ${agent.color}55` }}
            >
              {agent.icon}
            </div>
            <div>
              <p className="text-sm font-semibold leading-tight text-[--foreground]">
                {agent.label}
              </p>
              <p className="mt-0.5 text-[11px] leading-snug text-[--muted-foreground]">
                {agent.description}
              </p>
            </div>
            {selected === agent.id && (
              <motion.div
                layoutId="agent-selected-dot"
                className="absolute top-2.5 left-2.5 h-2 w-2 rounded-full bg-[--ring]"
                style={{ boxShadow: "0 0 8px var(--glow-accent)" }}
              />
            )}
          </motion.button>
        ))}
      </div>

      {/* Form panel */}
      <AnimatePresence mode="wait">
        {selected && selectedAgent?.hasForm && (
          <motion.div
            key={selected}
            initial={{ opacity: 0, height: 0, y: -8 }}
            animate={{ opacity: 1, height: "auto", y: 0 }}
            exit={{ opacity: 0, height: 0, y: -8 }}
            transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="overflow-hidden"
          >
            <div className={cn(
              "relative rounded-2xl border border-[--ring]/40 bg-[--card]/80 p-4 backdrop-blur-md",
              "shadow-[0_0_0_1px_var(--ring)/20,0_8px_32px_var(--glow-accent)]",
            )}>
              <div
                className="absolute inset-x-6 top-0 h-px rounded-full"
                style={{ background: "linear-gradient(to right, transparent, var(--ring), transparent)", opacity: 0.7 }}
              />
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div
                    className="flex h-7 w-7 items-center justify-center rounded-lg text-white text-xs"
                    style={{ background: selectedAgent.color }}
                  >
                    {selectedAgent.icon}
                  </div>
                  <span className="text-sm font-semibold text-[--foreground]">
                    {selectedAgent.label}
                  </span>
                </div>
                <button
                  onClick={() => setSelected(null)}
                  className="flex h-6 w-6 items-center justify-center rounded-full text-[--muted-foreground] hover:text-[--foreground] transition-colors"
                >
                  <X className="size-3.5" />
                </button>
              </div>

              {selected === "trip_planner" && (
                <TripPlannerForm onSubmit={(t) => { onSubmit(t); setSelected(null); }} />
              )}
              {selected === "flight_finder" && (
                <FlightFinderForm onSubmit={(t) => { onSubmit(t); setSelected(null); }} />
              )}
              {selected === "recommendations" && (
                <RecommendationsForm onSubmit={(t) => { onSubmit(t); setSelected(null); }} />
              )}
              {selected === "events" && (
                <EventsForm onSubmit={(t) => { onSubmit(t); setSelected(null); }} />
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}