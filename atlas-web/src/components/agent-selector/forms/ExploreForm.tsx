"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Compass } from "lucide-react";
import { useQueryState } from "nuqs";
import { createClient } from "@/providers/client";
import { getApiKey } from "@/lib/api-key";
import { FormSubmitButton } from "../shared";

interface ExploreCard {
  country: string;
  city: string;
  season: string;
  weather: string;
  events: string[];
  flight_cost: string;
  top_attractions: string[];
  top_restaurants: string[];
}

/* ─────────────────────────────────────────
   Explore — single destination card
───────────────────────────────────────── */
function ExploreCardItem({ card }: { card: ExploreCard }) {
  return (
    <div className="rounded-xl border border-[--border] bg-[--background]/60 p-3.5 flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <span className="font-semibold text-sm text-[--foreground]">
          {card.country} — {card.city}
        </span>
        <span className="text-xs text-[--muted-foreground]">{card.season}</span>
      </div>
      <p className="text-xs text-[--muted-foreground]">🌤 {card.weather}</p>
      <p className="text-xs text-[--muted-foreground]">✈️ {card.flight_cost}</p>
      {card.events.length > 0 && (
        <p className="text-xs text-[--muted-foreground]">🎟 {card.events[0]}</p>
      )}
      {card.top_attractions.length > 0 && (
        <p className="text-xs text-[--muted-foreground]">🏛 {card.top_attractions.join(" · ")}</p>
      )}
      {card.top_restaurants.length > 0 && (
        <p className="text-xs text-[--muted-foreground]">🍽 {card.top_restaurants.join(" · ")}</p>
      )}
    </div>
  );
}

/* ─────────────────────────────────────────
   Explore Form — no user inputs, just fetch
───────────────────────────────────────── */
export function ExploreForm() {
  const [apiUrl] = useQueryState("apiUrl", {
    defaultValue: process.env.NEXT_PUBLIC_API_URL ?? "",
  });
  const [authScheme] = useQueryState("authScheme", {
    defaultValue: process.env.NEXT_PUBLIC_AUTH_SCHEME ?? "",
  });
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [cards, setCards] = useState<ExploreCard[]>([]);

  const handleFetch = async () => {
    setStatus("loading");
    try {
      const client = createClient(
        apiUrl || "http://localhost:2024",
        getApiKey() ?? undefined,
        authScheme || undefined,
      );
      const result = await client.runs.wait(null, "explore", { input: {} }) as { cards?: ExploreCard[] };
      setCards(result.cards ?? []);
      setStatus("done");
    } catch {
      setStatus("error");
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {status === "idle" && (
        <div className="flex flex-col items-center gap-3 py-2 text-center">
          <p className="text-sm text-[--muted-foreground]">
            Browse highlights for 5 top European destinations — weather, events, flights, attractions, and restaurants.
          </p>
          <FormSubmitButton
            onClick={handleFetch}
            disabled={false}
            label="Explore Now"
            icon={<Compass className="size-4" />}
          />
        </div>
      )}

      {status === "loading" && (
        <div className="flex flex-col items-center gap-3 py-4">
          <motion.div
            className="size-8 rounded-full border-2 border-[--primary]/30 border-t-[--primary]"
            animate={{ rotate: 360 }}
            transition={{ duration: 0.9, repeat: Infinity, ease: "linear" }}
          />
          <p className="text-sm text-[--muted-foreground]">Fetching 5 destinations in parallel…</p>
        </div>
      )}

      {status === "error" && (
        <p className="py-2 text-center text-sm text-red-500">
          Could not load explore data. Please try again.
        </p>
      )}

      {status === "done" && cards.length > 0 && (
        <div className="flex flex-col gap-3">
          {cards.map((card) => (
            <ExploreCardItem key={card.country} card={card} />
          ))}
        </div>
      )}
    </div>
  );
}
