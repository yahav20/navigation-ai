"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

// Matches the shape in agent-selector/forms/ExploreForm.tsx
export interface ExploreCard {
  country: string;
  city: string;
  season: string;
  weather: string;
  events: string[];
  flight_cost: string;
  top_attractions: string[];
  top_restaurants: string[];
}

interface ExploreViewProps {
  cards: ExploreCard[];
}

/* ─────────────────────────────────────────
   Per-destination placeholder gradients
   (used when the static image fails to load)
───────────────────────────────────────── */
const PLACEHOLDER_GRADIENTS = [
  "linear-gradient(135deg, oklch(0.38 0.18 265), oklch(0.28 0.14 280))", // France — violet-blue
  "linear-gradient(135deg, oklch(0.5 0.2 45),  oklch(0.38 0.18 30))",   // Italy  — amber-terracotta
  "linear-gradient(135deg, oklch(0.44 0.2 205), oklch(0.34 0.16 220))", // Greece — Mediterranean teal
  "linear-gradient(135deg, oklch(0.38 0.14 250), oklch(0.28 0.1 260))", // Germany — steel-blue
  "linear-gradient(135deg, oklch(0.4 0.16 160), oklch(0.3 0.14 150))",  // England — forest-green
];

/* ─────────────────────────────────────────
   Image with graceful fallback
───────────────────────────────────────── */
function CardImage({
  country,
  city,
  gradient,
}: {
  country: string;
  city: string;
  gradient: string;
}) {
  const [error, setError] = useState(false);

  if (error) {
    return (
      <div
        className="w-full h-full flex items-center justify-center"
        style={{ background: gradient }}
      >
        <span
          className="text-[5rem] font-black leading-none select-none"
          style={{ color: "oklch(1 0 0 / 0.15)" }}
        >
          {city[0]}
        </span>
      </div>
    );
  }

  return (
    <motion.div
      className="w-full h-full"
      variants={{
        hover: {
          scale: 1.05,
          transition: { duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] },
        },
      }}
    >
      <img
        src={`/images/explore/${country.toLowerCase()}.jpg`}
        alt={`${city}, ${country}`}
        className="w-full h-full object-cover"
        onError={() => setError(true)}
      />
    </motion.div>
  );
}

/* ─────────────────────────────────────────
   Info section — fades in after card
───────────────────────────────────────── */
function InfoSection({
  delay,
  className,
  children,
}: {
  delay: number;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay, duration: 0.35, ease: "easeOut" }}
    >
      {children}
    </motion.div>
  );
}

/* ─────────────────────────────────────────
   Single destination card
───────────────────────────────────────── */
function DestinationCard({
  card,
  index,
}: {
  card: ExploreCard;
  index: number;
}) {
  const entranceDelay = index * 0.09;
  const gradient = PLACEHOLDER_GRADIENTS[index % PLACEHOLDER_GRADIENTS.length];

  return (
    <motion.article
      className="relative rounded-2xl border border-[--border] overflow-hidden flex flex-col"
      initial={{ opacity: 0, y: 28 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: entranceDelay, duration: 0.45, ease: "easeOut" }}
      whileHover="hover"
      variants={{
        hover: {
          y: -6,
          boxShadow:
            "0 24px 64px oklch(0.4 0.18 230 / 0.28), 0 0 0 1px oklch(0.55 0.18 230 / 0.35)",
          transition: { duration: 0.25, ease: "easeOut" },
        },
      }}
      style={{
        boxShadow: "0 2px 16px oklch(0.5 0.12 240 / 0.1)",
      }}
    >
      {/* ── Top: image with gradient overlay ── */}
      <div className="relative h-52 overflow-hidden shrink-0">
        <CardImage country={card.country} city={card.city} gradient={gradient} />

        {/* Dark gradient fade from transparent → dark at bottom */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              "linear-gradient(to bottom, transparent 35%, oklch(0.1 0.05 240 / 0.85) 100%)",
          }}
        />

        {/* City name floated over the image bottom */}
        <div className="absolute bottom-3 left-4 right-4">
          <p className="text-white font-bold text-lg leading-tight drop-shadow-sm">
            {card.city}
          </p>
          <p className="text-white/70 text-xs mt-0.5">{card.country}</p>
        </div>
      </div>

      {/* ── Bottom: frosted glass info panel ── */}
      <div className="flex-1 flex flex-col gap-3 p-4 backdrop-blur-md bg-[--card]/80 border-t border-[--border]">

        {/* Season badge */}
        <InfoSection delay={entranceDelay + 0.18}>
          <span
            className="inline-flex items-center rounded-full px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider"
            style={{
              background: "oklch(0.55 0.18 230 / 0.12)",
              color: "oklch(0.55 0.18 230)",
              border: "1px solid oklch(0.55 0.18 230 / 0.25)",
            }}
          >
            {card.season}
          </span>
        </InfoSection>

        {/* Weather + Flight */}
        <InfoSection delay={entranceDelay + 0.25} className="flex flex-col gap-1">
          <p className="text-xs text-[--muted-foreground]">🌤 {card.weather}</p>
          <p className="text-xs text-[--muted-foreground]">✈️ {card.flight_cost}</p>
        </InfoSection>

        {/* Events */}
        {card.events.length > 0 && (
          <InfoSection delay={entranceDelay + 0.32} className="flex flex-col gap-1">
            <p className="text-[11px] font-semibold text-[--foreground]">🎟 Events</p>
            {card.events.slice(0, 2).map((ev, i) => (
              <p
                key={i}
                className={cn(
                  "text-xs text-[--muted-foreground] leading-snug",
                  i === 1 && card.events.length > 2 && "opacity-60",
                )}
              >
                {ev}
              </p>
            ))}
          </InfoSection>
        )}

        {/* Attractions */}
        {card.top_attractions.length > 0 && (
          <InfoSection delay={entranceDelay + 0.39} className="flex flex-col gap-1">
            <p className="text-[11px] font-semibold text-[--foreground]">🏛 Attractions</p>
            <ul className="flex flex-col gap-0.5">
              {card.top_attractions.slice(0, 5).map((name, i) => (
                <li
                  key={i}
                  className="flex items-center gap-1.5 text-xs text-[--muted-foreground]"
                >
                  <span
                    className="size-1 rounded-full shrink-0"
                    style={{ background: "oklch(0.55 0.18 230 / 0.5)" }}
                  />
                  {name}
                </li>
              ))}
            </ul>
          </InfoSection>
        )}

        {/* Restaurants */}
        {card.top_restaurants.length > 0 && (
          <InfoSection delay={entranceDelay + 0.46} className="flex flex-col gap-1">
            <p className="text-[11px] font-semibold text-[--foreground]">🍽 Restaurants</p>
            <ul className="flex flex-col gap-0.5">
              {card.top_restaurants.slice(0, 5).map((name, i) => (
                <li
                  key={i}
                  className="flex items-center gap-1.5 text-xs text-[--muted-foreground]"
                >
                  <span
                    className="size-1 rounded-full shrink-0"
                    style={{ background: "oklch(0.55 0.18 230 / 0.5)" }}
                  />
                  {name}
                </li>
              ))}
            </ul>
          </InfoSection>
        )}
      </div>
    </motion.article>
  );
}

/* ─────────────────────────────────────────
   ExploreView — 5-card grid
───────────────────────────────────────── */
export default function ExploreView({ cards }: ExploreViewProps) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 xl:gap-5 w-full">
      {cards.map((card, i) => (
        <DestinationCard key={card.country} card={card} index={i} />
      ))}
    </div>
  );
}
