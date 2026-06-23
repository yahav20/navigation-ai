"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  MapPin,
  Plane,
  Ticket,
  Star,
  X,
} from "lucide-react";
import { TripPlannerForm } from "./forms/TripPlannerForm";
import { FlightFinderForm } from "./forms/FlightFinderForm";
import { RecommendationsForm } from "./forms/RecommendationsForm";
import { EventsForm } from "./forms/EventsForm";

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
