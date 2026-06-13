"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  MapPin,
  Plane,
  MessageCircle,
  Star,
  Calendar,
  DollarSign,
  Globe,
  Users,
  BedDouble,
  Baby,
  ChevronRight,
  X,
} from "lucide-react";

/* ─────────────────────────────────────────
   Types
───────────────────────────────────────── */
export type AgentType =
  | "trip_planner"
  | "flight_finder"
  | "general"
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
    id: "general",
    label: "General Questions",
    description: "Any question you have about travel and tourism",
    icon: <MessageCircle className="size-5" />,
    color: "oklch(0.5 0.15 260)",
    hasForm: false,
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
  const [adults, setAdults] = useState("");
  const [children, setChildren] = useState("");
  const [rooms, setRooms] = useState("");

  const handleSubmit = () => {
    if (!origin || !destination || !days) return;
    const numAdults   = parseInt(adults)   || 1;
    const numChildren = parseInt(children) || 0;
    const numRooms    = parseInt(rooms)    || 1;
    const travelers =
      numAdults === 1 && numChildren === 0
        ? "1 adult"
        : `${numAdults} adult${numAdults > 1 ? "s" : ""}` +
          (numChildren > 0 ? ` and ${numChildren} child${numChildren > 1 ? "ren" : ""}` : "");

    const parts = [
      `Build a 3-day itinerary from ${origin} to ${destination}`,
      `for ${days} days`,
      `for ${travelers}`,
      numRooms > 1 ? `in ${numRooms} rooms` : "",
      budget && `with a budget of ${budget}`,
      month && `in ${month}`,
    ].filter(Boolean);
    onSubmit(parts.join(", ") + ".");
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className={labelCls}>
            <Globe className="mb-0.5 mr-1 inline size-3" />
            Origin country
          </label>
          <input className={inputCls} placeholder="Israel" value={origin}
            onChange={(e) => setOrigin(e.target.value)} dir="ltr" />
        </div>
        <div>
          <label className={labelCls}>
            <MapPin className="mb-0.5 mr-1 inline size-3" />
            Destination country
          </label>
          <input className={inputCls} placeholder="Italy" value={destination}
            onChange={(e) => setDestination(e.target.value)} dir="ltr" />
        </div>
        <div>
          <label className={labelCls}>
            <Calendar className="mb-0.5 mr-1 inline size-3" />
            Number of days
          </label>
          <input className={inputCls} type="number" placeholder="7" min={1}
            value={days} onChange={(e) => setDays(e.target.value)} dir="ltr" />
        </div>
        <div>
          <label className={labelCls}>
            <Calendar className="mb-0.5 mr-1 inline size-3" />
            Estimated month
          </label>
          <select className={inputCls} value={month} onChange={(e) => setMonth(e.target.value)} dir="ltr">
            <option value="">Select month...</option>
            {["January","February","March","April","May","June",
              "July","August","September","October","November","December"].map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
        <div className="col-span-2">
          <label className={labelCls}>
            <DollarSign className="mb-0.5 mr-1 inline size-3" />
            Estimated budget (optional)
          </label>
          <input className={inputCls} placeholder="$5,000 per person" value={budget}
            onChange={(e) => setBudget(e.target.value)} dir="ltr" />
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
        <FormSubmitButton onClick={handleSubmit} disabled={!origin || !destination || !days}
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
  const [budget, setBudget] = useState("");
  const [days, setDays] = useState("");
  const [adults, setAdults] = useState("");
  const [children, setChildren] = useState("");

  const handleSubmit = () => {
    if (!origin || !destination) return;
    const numAdults   = parseInt(adults)   || 1;
    const numChildren = parseInt(children) || 0;
    const travelers =
      numAdults === 1 && numChildren === 0
        ? "1 adult"
        : `${numAdults} adult${numAdults > 1 ? "s" : ""}` +
          (numChildren > 0 ? ` and ${numChildren} child${numChildren > 1 ? "ren" : ""}` : "");

    const parts = [
      `I'm looking for flights from ${origin} to ${destination}`,
      month && `in ${month}`,
      days && `for a ${days}-day trip`,
      `for ${travelers}`,
      budget && `with a budget of ${budget}`,
    ].filter(Boolean);
    onSubmit(parts.join(", ") + ".");
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className={labelCls}>
            <Globe className="mb-0.5 mr-1 inline size-3" />
            Origin city / airport
          </label>
          <input className={inputCls} placeholder="Tel Aviv (TLV)" value={origin}
            onChange={(e) => setOrigin(e.target.value)} dir="ltr" />
        </div>
        <div>
          <label className={labelCls}>
            <Plane className="mb-0.5 mr-1 inline size-3" />
            Destination
          </label>
          <input className={inputCls} placeholder="Rome (FCO)" value={destination}
            onChange={(e) => setDestination(e.target.value)} dir="ltr" />
        </div>
        <div>
          <label className={labelCls}>
            <Calendar className="mb-0.5 mr-1 inline size-3" />
            Estimated month
          </label>
          <select className={inputCls} value={month} onChange={(e) => setMonth(e.target.value)} dir="ltr">
            <option value="">Select month...</option>
            {["January","February","March","April","May","June",
              "July","August","September","October","November","December"].map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
        <div>
          <label className={labelCls}>
            <Calendar className="mb-0.5 mr-1 inline size-3" />
            Trip duration (days)
          </label>
          <input className={inputCls} type="number" min={1} placeholder="7" value={days}
            onChange={(e) => setDays(e.target.value)} dir="ltr" />
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
        <FormSubmitButton onClick={handleSubmit} disabled={!origin || !destination}
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

  const handleSubmit = () => {
    if (!origin) return;
    const parts = [
      "I'm looking for travel recommendations",
      origin && `from ${origin}`,
      month && `in ${month}`,
      topic && `about ${topic}`,
    ].filter(Boolean);
    onSubmit(parts.join(" ") + ".");
  };

  return (
    <div className="grid grid-cols-2 gap-3">
      <div>
        <label className={labelCls}>
          <Globe className="mb-0.5 mr-1 inline size-3" />
          Origin country
        </label>
        <input className={inputCls} placeholder="Israel" value={origin}
          onChange={(e) => setOrigin(e.target.value)} dir="ltr" />
      </div>
      <div>
        <label className={labelCls}>
          <Calendar className="mb-0.5 mr-1 inline size-3" />
          Estimated date
        </label>
        <select className={inputCls} value={month} onChange={(e) => setMonth(e.target.value)} dir="ltr">
          <option value="">Select month...</option>
          {["January","February","March","April","May","June",
            "July","August","September","October","November","December"].map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
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
        <FormSubmitButton onClick={handleSubmit} disabled={!origin}
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
  return (
    <motion.button
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "flex items-center gap-2 rounded-xl px-5 py-2.5",
        "bg-[--primary] text-[--primary-foreground] text-sm font-medium",
        "shadow-[0_0_16px_var(--glow-primary)]",
        "hover:shadow-[0_0_28px_var(--glow-accent)]",
        "disabled:opacity-40 disabled:shadow-none disabled:cursor-not-allowed",
        "transition-all duration-300",
      )}
      whileHover={disabled ? {} : { scale: 1.03 }}
      whileTap={disabled ? {} : { scale: 0.97 }}
    >
      {icon}
      {label}
      <ChevronRight className="size-4" />
    </motion.button>
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
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}