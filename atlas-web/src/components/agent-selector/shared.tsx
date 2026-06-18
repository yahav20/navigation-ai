"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  Users,
  Baby,
  BedDouble,
  ChevronRight,
  Check,
} from "lucide-react";

/* ─────────────────────────────────────────
   Shared input styles
───────────────────────────────────────── */
export const inputCls = cn(
  "w-full rounded-xl border border-[--border] bg-[--background]/60 px-3 py-2",
  "text-sm text-[--foreground] placeholder:text-[--muted-foreground]/60",
  "focus:border-[--ring] focus:outline-none focus:ring-0",
  "transition-colors duration-200",
);

export const labelCls = "block text-xs font-medium text-[--muted-foreground] mb-1";

/* ─────────────────────────────────────────
   Travelers row (shared between forms)
───────────────────────────────────────── */
export function TravelersRow({
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
   Submit button
───────────────────────────────────────── */
export function FormSubmitButton({
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
