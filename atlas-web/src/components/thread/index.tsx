import { v4 as uuidv4 } from "uuid";
import { ReactNode, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { DEFAULT_API_URL } from "@/lib/deployment";
import { useStreamContext } from "@/providers/Stream";
import { useState, FormEvent } from "react";
import { Button } from "../ui/button";
import { Checkpoint, Message } from "@langchain/langgraph-sdk";
import { AssistantMessage, AssistantMessageLoading } from "./messages/ai";
import { HumanMessage } from "./messages/human";
import { useTheme } from "next-themes";
import { Moon, Sun, Compass } from "lucide-react";
import {
  DO_NOT_RENDER_ID_PREFIX,
  ensureToolCallsHaveResponses,
} from "@/lib/ensure-tool-responses";
import { LangGraphLogoSVG } from "../icons/langgraph";
import { TooltipIconButton } from "./tooltip-icon-button";
import {
  ArrowDown,
  LoaderCircle,
  PanelRightOpen,
  PanelRightClose,
  SquarePen,
  XIcon,
  Plus,
  Send,
  Shuffle,
  Dices,
  Users,
  Baby,
  DollarSign,
  Globe,
  Minus,
} from "lucide-react";
import { useQueryState, parseAsBoolean } from "nuqs";
import {
  StickToBottom,
  useStickToBottomContext,
  type StickToBottomContext,
} from "use-stick-to-bottom";
import ThreadHistory from "./history";
import { toast } from "sonner";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { Label } from "../ui/label";
import { GitHubSVG } from "../icons/github";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import { useFileUpload } from "@/hooks/use-file-upload";
import { ContentBlocksPreview } from "./ContentBlocksPreview";
import {
  useArtifactOpen,
  ArtifactContent,
  ArtifactTitle,
  useArtifactContext,
} from "./artifact";
import { AgentSelector } from "./agent-selector";
import { createClient } from "@/providers/client";
import { getApiKey } from "@/lib/api-key";
import ExploreView, { type ExploreCard } from "./ExploreView";
import ExploreLoadingScreen from "./ExploreLoadingScreen";
import { inputCls, labelCls } from "../agent-selector/shared";

/* ─────────────────────────────────────────
   Ambient background: animated ocean orbs
───────────────────────────────────────── */
function OceanBackground() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      {/* Large top-left orb */}
      <motion.div
        className="absolute -top-60 -left-60 h-[800px] w-[800px] rounded-full"
        style={{
          background:
            "radial-gradient(ellipse, oklch(0.52 0.2 245 / 0.14) 0%, oklch(0.45 0.15 260 / 0.05) 50%, transparent 75%)",
          filter: "blur(80px)",
        }}
        animate={{ x: [0, 50, 0], y: [0, 35, 0], scale: [1, 1.1, 1] }}
        transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
      />
      {/* Bottom-right teal orb */}
      <motion.div
        className="absolute -bottom-48 -right-48 h-[700px] w-[700px] rounded-full"
        style={{
          background:
            "radial-gradient(ellipse, oklch(0.62 0.18 195 / 0.11) 0%, oklch(0.55 0.14 210 / 0.04) 55%, transparent 78%)",
          filter: "blur(70px)",
        }}
        animate={{ x: [0, -40, 0], y: [0, -28, 0], scale: [1, 1.14, 1] }}
        transition={{ duration: 24, repeat: Infinity, ease: "easeInOut", delay: 4 }}
      />
      {/* Centre-right accent orb */}
      <motion.div
        className="absolute top-1/3 right-1/4 h-[380px] w-[380px] rounded-full"
        style={{
          background:
            "radial-gradient(ellipse, oklch(0.68 0.16 215 / 0.07) 0%, transparent 70%)",
          filter: "blur(55px)",
        }}
        animate={{ x: [0, 28, -14, 0], y: [0, -32, 12, 0] }}
        transition={{ duration: 16, repeat: Infinity, ease: "easeInOut", delay: 8 }}
      />
      {/* Subtle centre glow — anchored, no motion */}
      <div
        className="absolute left-1/2 top-1/2 h-[500px] w-[900px] -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{
          background:
            "radial-gradient(ellipse, oklch(0.58 0.12 235 / 0.06) 0%, transparent 70%)",
          filter: "blur(90px)",
        }}
      />
    </div>
  );
}

// ─── Plane Hero ───────────────────────────────────────────────────────────────

function PlaneHero() {
  const [phase, setPhase] = useState<"flying" | "logo">(() =>
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ? "logo"
      : "flying",
  );
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    if (phase === "logo") return;

    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const W = window.innerWidth;
    const H = window.innerHeight;
    canvas.width = W;
    canvas.height = H;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const isDark = document.documentElement.classList.contains("dark");
    const planeColor = isDark ? "#eef0ff" : "#0d0d18";
    const trailRGB = isDark ? "215,220,255" : "13,13,24";

    // Landing target: center of container
    const rect = container.getBoundingClientRect();
    const endX = rect.left + rect.width / 2;
    const endY = rect.top + rect.height / 2 - 20;

    // Cubic bezier: off-screen bottom-left → arc top-right → center
    const P0 = { x: -50, y: H * 1.08 };
    const P1 = { x: W * 0.17, y: H * 0.1 };
    const P2 = { x: W * 0.74, y: H * 0.07 };
    const P3 = { x: endX, y: endY };

    const bez = (t: number) => ({
      x: (1-t)**3*P0.x + 3*(1-t)**2*t*P1.x + 3*(1-t)*t**2*P2.x + t**3*P3.x,
      y: (1-t)**3*P0.y + 3*(1-t)**2*t*P1.y + 3*(1-t)*t**2*P2.y + t**3*P3.y,
    });
    const bezD = (t: number) => ({
      x: 3*(1-t)**2*(P1.x-P0.x) + 6*(1-t)*t*(P2.x-P1.x) + 3*t**2*(P3.x-P2.x),
      y: 3*(1-t)**2*(P1.y-P0.y) + 6*(1-t)*t*(P2.y-P1.y) + 3*t**2*(P3.y-P2.y),
    });

    const N = 500;
    const pts = Array.from({ length: N + 1 }, (_, i) => bez(i / N));
    const FLIGHT_MS = 2200;
    const TRAIL_WIN = 0.2;
    let t0: number | null = null;

    const drawPlane = (x: number, y: number, angle: number, alpha: number, scale: number) => {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(angle);
      ctx.scale(scale, scale);
      ctx.globalAlpha = Math.max(0, Math.min(1, alpha));
      ctx.fillStyle = planeColor;
      // Fuselage
      ctx.beginPath(); ctx.moveTo(17,0); ctx.lineTo(-13,-3); ctx.lineTo(-13,3); ctx.closePath(); ctx.fill();
      // Upper wing
      ctx.beginPath(); ctx.moveTo(4,-1.5); ctx.lineTo(-2,-14); ctx.lineTo(-8,-14); ctx.lineTo(-6,-1.5); ctx.closePath(); ctx.fill();
      // Lower wing
      ctx.beginPath(); ctx.moveTo(4,1.5); ctx.lineTo(-2,14); ctx.lineTo(-8,14); ctx.lineTo(-6,1.5); ctx.closePath(); ctx.fill();
      // Tail fin
      ctx.beginPath(); ctx.moveTo(-9,-1.5); ctx.lineTo(-13,-7); ctx.lineTo(-13,-3); ctx.closePath(); ctx.fill();
      ctx.restore();
    };

    const tick = (ts: number) => {
      if (!t0) t0 = ts;
      const t = Math.min((ts - t0) / FLIGHT_MS, 1);
      const si = Math.min(Math.floor(t * N), N - 1);
      ctx.clearRect(0, 0, W, H);

      // Dashed fading trail
      const trailT0 = Math.max(0, t - TRAIL_WIN);
      const trailI0 = Math.floor(trailT0 * N);
      for (let i = trailI0; i < si; i++) {
        if (Math.floor((i - trailI0) / 5) % 2 === 1) continue;
        const ratio = (i / N - trailT0) / Math.max(t - trailT0, 1e-6);
        ctx.beginPath();
        ctx.strokeStyle = `rgba(${trailRGB},${Math.max(0, ratio) * 0.45})`;
        ctx.lineWidth = 1.5;
        ctx.lineCap = "round";
        ctx.moveTo(pts[i].x, pts[i].y);
        ctx.lineTo(pts[Math.min(i+1, N)].x, pts[Math.min(i+1, N)].y);
        ctx.stroke();
      }

      if (t < 1) {
        const pos = bez(t);
        const d = bezD(t);
        drawPlane(pos.x, pos.y, Math.atan2(d.y, d.x), 1, 2);
        rafRef.current = requestAnimationFrame(tick);
      } else {
        // Landing: shrink + dissolve
        const fa = Math.atan2(bezD(0.999).y, bezD(0.999).x);
        let lp = 0;
        const land = () => {
          lp = Math.min(lp + 0.055, 1);
          ctx.clearRect(0, 0, W, H);
          drawPlane(P3.x, P3.y, fa, 1 - lp * 2, 2 - lp * 1.7);
          if (lp < 1) { rafRef.current = requestAnimationFrame(land); }
          else { setPhase("logo"); }
        };
        rafRef.current = requestAnimationFrame(land);
      }
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [phase]);

  return (
    <div ref={containerRef} className="relative flex flex-col items-center gap-4 text-center">
      {/* Full-viewport canvas — renders the plane flight */}
      {phase === "flying" && (
        <canvas
          ref={canvasRef}
          className="pointer-events-none"
          style={{ position: "fixed", inset: 0, zIndex: 9999 }}
        />
      )}

      {/* Logo + text — invisible while flying, fades in on landing */}
      <motion.div
        className="relative flex flex-col items-center gap-4"
        initial={{ opacity: 0, scale: 0.92 }}
        animate={{ opacity: phase === "logo" ? 1 : 0, scale: phase === "logo" ? 1 : 0.92 }}
        transition={{ duration: 0.7, ease: [0.23, 1, 0.32, 1] }}
      >
        {/* Soft glow — dark mode only, appears after landing */}
        <div
          className="pointer-events-none absolute left-1/2 top-0 -translate-x-1/2 -translate-y-1/2 rounded-full opacity-0 dark:opacity-100"
          style={{
            width: 300, height: 300,
            background: "radial-gradient(ellipse, oklch(0.45 0.18 278 / 0.24) 0%, transparent 68%)",
            filter: "blur(40px)",
          }}
        />

        {/* Bare SVG logo — no container, no border, no shadow */}
        <LangGraphLogoSVG width={160} height={160} className="relative z-10 flex-shrink-0" />

        <h1
          className="text-[2.75rem] text-foreground dark:text-white"
          style={{ fontFamily: "var(--font-marcellus), sans-serif", letterSpacing: "0.01em", lineHeight: 1.1 }}
        >
          Navigation AI
        </h1>

        <p className="text-sm text-[--muted-foreground]" style={{ fontFamily: "var(--font-marcellus), sans-serif" }}>
          Choose an agent or type a freeform question below
        </p>
      </motion.div>
    </div>
  );
}

function StickyToBottomContent(props: {
  content: ReactNode;
  footer?: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  const context = useStickToBottomContext();
  return (
    <div
      ref={context.scrollRef}
      style={{ width: "100%", height: "100%" }}
      className={props.className}
    >
      <div ref={context.contentRef} className={props.contentClassName}>
        {props.content}
      </div>
      {props.footer}
    </div>
  );
}

function ScrollToBottom(props: { className?: string }) {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();
  if (isAtBottom) return null;
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 8 }}
    >
      <Button
        variant="outline"
        className={cn(
          "gap-2 rounded-full border-[--border] bg-[--card]/80 shadow-lg backdrop-blur-md",
          "hover:border-[--ring] hover:shadow-[0_0_16px_var(--glow-accent)]",
          "transition-all duration-300",
          props.className,
        )}
        onClick={() => scrollToBottom()}
      >
        <ArrowDown className="h-4 w-4" />
        <span className="text-sm">Scroll to bottom</span>
      </Button>
    </motion.div>
  );
}

function OpenGitHubRepo() {
  const { theme, setTheme } = useTheme();
  return (
    <div className="flex items-center gap-1">
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <motion.a
              href="https://github.com/yahav20/navigation-ai.git"
              target="_blank"
              className="flex h-9 w-9 items-center justify-center rounded-full text-[--muted-foreground] transition-colors hover:text-[--foreground]"
              whileHover={{ scale: 1.12 }}
              whileTap={{ scale: 0.95 }}
            >
              <GitHubSVG width="20" height="20" />
            </motion.a>
          </TooltipTrigger>
          <TooltipContent side="left">
            <p>Open GitHub repo</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      <motion.button
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        className={cn(
          "relative flex h-9 w-9 items-center justify-center rounded-full",
          "border border-[--border] bg-[--card]/60 backdrop-blur-sm",
          "text-[--muted-foreground] transition-all duration-300",
          "hover:border-[--ring] hover:text-[--foreground]",
          "hover:shadow-[0_0_12px_var(--glow-accent)]",
        )}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.92, rotate: 15 }}
        aria-label="Toggle theme"
      >
        <Sun className="h-[1.1rem] w-[1.1rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
        <Moon className="absolute h-[1.1rem] w-[1.1rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
      </motion.button>
    </div>
  );
}

function BrandLogo({ onClick }: { onClick?: () => void }) {
  return (
    <motion.button
      onClick={onClick}
      className="group flex cursor-pointer items-center gap-2.5"
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="relative">
        <div
          className="absolute inset-0 rounded-full opacity-0 blur-lg transition-opacity duration-300 group-hover:opacity-100"
          style={{ background: "var(--glow-accent)" }}
        />
        <LangGraphLogoSVG width={60} height={60} className="relative z-10" />
      </div>
      <span className="text-xl font-semibold tracking-tight" style={{ letterSpacing: "-0.02em" }}>
        Agent Chat
      </span>
    </motion.button>
  );
}

function PanelToggleButton({ isOpen, onToggle }: { isOpen: boolean; onToggle: () => void }) {
  return (
    <motion.button
      onClick={onToggle}
      className={cn(
        "flex h-9 w-9 items-center justify-center rounded-full",
        "border border-[--border] bg-[--card]/50 backdrop-blur-sm",
        "text-[--muted-foreground] transition-all duration-200",
        "hover:border-[--ring] hover:text-[--foreground]",
        "hover:shadow-[0_0_10px_var(--glow-primary)]",
      )}
      whileHover={{ scale: 1.08 }}
      whileTap={{ scale: 0.92 }}
    >
      {isOpen ? <PanelRightOpen className="size-4" /> : <PanelRightClose className="size-4" />}
    </motion.button>
  );
}

const DESTINATIONS = ["Rome", "Bali", "Lisbon", "Bangkok", "Amsterdam", "Santorini", "Kyoto", "Cape Town", "Prague", "Vienna"];
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const YEARS = [2026, 2027];

function SurpriseMeModal({
  open,
  onClose,
  onSubmit,
}: {
  open: boolean;
  onClose: () => void;
  onSubmit: (text: string) => void;
}) {
  const [origin, setOrigin] = useState("");
  const [budget, setBudget] = useState("");
  const [adults, setAdults] = useState(1);
  const [children, setChildren] = useState(0);
  const [originError, setOriginError] = useState("");

  useEffect(() => {
    if (open) {
      setOrigin("");
      setBudget("");
      setAdults(1);
      setChildren(0);
      setOriginError("");
    }
  }, [open]);

  const handleSubmit = () => {
    if (!origin.trim()) {
      setOriginError("Origin city is required");
      return;
    }
    setOriginError("");
    const pick = <T,>(arr: T[]) => arr[Math.floor(Math.random() * arr.length)];
    let destination = pick(DESTINATIONS);
    while (destination.toLowerCase() === origin.trim().toLowerCase()) destination = pick(DESTINATIONS);
    const month = pick(MONTHS);
    const year = pick(YEARS);
    const days = Math.floor(Math.random() * 10) + 5;
    const travelers =
      adults === 1 && children === 0
        ? "1 adult"
        : `${adults} adult${adults > 1 ? "s" : ""}` +
          (children > 0 ? ` and ${children} child${children > 1 ? "ren" : ""}` : "");
    const text =
      [
        `I'm looking for flights from ${origin.trim()} to ${destination}`,
        `in ${month} ${year}`,
        `for a ${days}-day trip`,
        `for ${travelers}`,
        budget.trim() ? `with a budget of $${Number(budget).toLocaleString()}` : null,
      ].filter(Boolean).join(", ") + ".";
    onSubmit(text);
    onClose();
  };

  const stepperCls = cn(
    "flex h-8 w-8 items-center justify-center rounded-lg",
    "border border-[--border] bg-[--background]/60",
    "text-[--muted-foreground] transition-colors duration-200",
    "hover:border-[--ring] hover:text-[--foreground]",
  );

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[60] flex items-center justify-center bg-[--background]/80 backdrop-blur-md"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
        >
          <motion.div
            className={cn(
              "relative mx-4 w-full max-w-md rounded-2xl p-6",
              "border border-[--ring]/40 bg-[--card]/90",
              "shadow-[0_0_48px_var(--glow-accent)]",
            )}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
          >
            {/* Header */}
            <div className="mb-5 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Dices className="size-5 text-[--ring]" />
                <h2 className="text-base font-semibold text-[--foreground]">Surprise Me!</h2>
              </div>
              <motion.button
                onClick={onClose}
                className={cn(
                  "flex h-7 w-7 items-center justify-center rounded-full",
                  "text-[--muted-foreground] transition-all duration-200",
                  "hover:bg-[--muted] hover:text-[--foreground]",
                )}
                whileHover={{ scale: 1.12, rotate: 90 }}
                whileTap={{ scale: 0.9 }}
              >
                <XIcon className="size-4" />
              </motion.button>
            </div>

            {/* Fields */}
            <div className="flex flex-col gap-4">
              {/* Origin */}
              <div>
                <label className={labelCls}>
                  <Globe className="mb-0.5 mr-1 inline size-3" />
                  Where are you flying from?<span className="ml-0.5 text-[--ring]">*</span>
                </label>
                <input
                  className={cn(inputCls, originError && "border-red-500")}
                  placeholder="e.g. Tel Aviv, London, New York…"
                  value={origin}
                  onChange={(e) => { setOrigin(e.target.value); if (e.target.value.trim()) setOriginError(""); }}
                  dir="ltr"
                  autoFocus
                />
                {originError && <p className="mt-1 text-xs text-red-500">{originError}</p>}
              </div>

              {/* Budget */}
              <div>
                <label className={labelCls}>
                  <DollarSign className="mb-0.5 mr-1 inline size-3" />
                  Budget ($)
                </label>
                <input
                  className={inputCls}
                  type="number"
                  min={0}
                  placeholder="e.g. 3000"
                  value={budget}
                  onChange={(e) => setBudget(e.target.value)}
                  onWheel={(e) => e.currentTarget.blur()}
                  dir="ltr"
                />
              </div>

              {/* Travelers */}
              <div>
                <p className={labelCls}>Travelers</p>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className={labelCls}>
                      <Users className="mb-0.5 mr-1 inline size-3" />
                      Adults
                    </label>
                    <div className="flex items-center gap-2">
                      <motion.button
                        type="button"
                        onClick={() => setAdults((a) => Math.max(1, a - 1))}
                        className={cn(stepperCls, adults <= 1 && "cursor-not-allowed opacity-40")}
                        whileTap={adults > 1 ? { scale: 0.92 } : {}}
                      >
                        <Minus className="size-3.5" />
                      </motion.button>
                      <span className="w-6 text-center text-sm font-medium text-[--foreground]">{adults}</span>
                      <motion.button
                        type="button"
                        onClick={() => setAdults((a) => Math.min(4, a + 1))}
                        className={cn(stepperCls, adults >= 4 && "cursor-not-allowed opacity-40")}
                        whileTap={adults < 4 ? { scale: 0.92 } : {}}
                      >
                        <Plus className="size-3.5" />
                      </motion.button>
                    </div>
                  </div>
                  <div>
                    <label className={labelCls}>
                      <Baby className="mb-0.5 mr-1 inline size-3" />
                      Children
                    </label>
                    <div className="flex items-center gap-2">
                      <motion.button
                        type="button"
                        onClick={() => setChildren((c) => Math.max(0, c - 1))}
                        className={cn(stepperCls, children <= 0 && "cursor-not-allowed opacity-40")}
                        whileTap={children > 0 ? { scale: 0.92 } : {}}
                      >
                        <Minus className="size-3.5" />
                      </motion.button>
                      <span className="w-6 text-center text-sm font-medium text-[--foreground]">{children}</span>
                      <motion.button
                        type="button"
                        onClick={() => setChildren((c) => Math.min(3, c + 1))}
                        className={cn(stepperCls, children >= 3 && "cursor-not-allowed opacity-40")}
                        whileTap={children < 3 ? { scale: 0.92 } : {}}
                      >
                        <Plus className="size-3.5" />
                      </motion.button>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="mt-6 flex items-center justify-end gap-3">
              <motion.button
                onClick={onClose}
                className={cn(
                  "rounded-xl border border-[--border] bg-transparent px-4 py-2",
                  "text-sm text-[--muted-foreground]",
                  "hover:border-[--ring]/60 hover:text-[--foreground]",
                  "transition-all duration-200",
                )}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
              >
                Cancel
              </motion.button>
              <motion.button
                onClick={handleSubmit}
                className={cn(
                  "flex items-center gap-2 rounded-xl px-5 py-2.5",
                  "bg-[--primary] text-[--primary-foreground] text-sm font-medium",
                  "shadow-[0_0_16px_var(--glow-primary)]",
                  "hover:shadow-[0_0_28px_var(--glow-accent)]",
                  "transition-all duration-300",
                )}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
              >
                <Dices className="size-4" />
                Surprise Me!
              </motion.button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function SurpriseMeButton({ onClick, disabled }: { onClick: () => void; disabled: boolean }) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <motion.button
            onClick={onClick}
            disabled={disabled}
            className={cn(
              "relative flex h-9 w-9 items-center justify-center rounded-full",
              "border bg-[--card]/60 backdrop-blur-sm",
              "transition-all duration-300",
              disabled
                ? "cursor-not-allowed border-[--border] opacity-40"
                : "border-[--border] text-[--muted-foreground] hover:border-[--ring] hover:text-[--foreground] hover:shadow-[0_0_12px_var(--glow-accent)]",
            )}
            whileHover={disabled ? {} : { scale: 1.1 }}
            whileTap={disabled ? {} : { scale: 0.92 }}
            aria-label="Surprise me"
          >
            <Shuffle className="size-[1.1rem]" />
          </motion.button>
        </TooltipTrigger>
        <TooltipContent side="left">
          <p>Surprise me!</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

function ExploreButton({ onClick, active }: { onClick: () => void; active: boolean }) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <motion.button
            onClick={onClick}
            className={cn(
              "relative flex h-9 w-9 items-center justify-center rounded-full",
              "border bg-[--card]/60 backdrop-blur-sm",
              "transition-all duration-300",
              active
                ? "border-[--ring] text-[--ring] shadow-[0_0_14px_var(--glow-accent)]"
                : "border-[--border] text-[--muted-foreground] hover:border-[--ring] hover:text-[--foreground] hover:shadow-[0_0_12px_var(--glow-accent)]",
            )}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.92 }}
            aria-label="Explore destinations"
          >
            <Compass className="size-[1.1rem]" />
          </motion.button>
        </TooltipTrigger>
        <TooltipContent side="left">
          <p>Explore destinations</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

/* ─────────────────────────────────────────
   Main Thread component
───────────────────────────────────────── */
export function Thread() {
  const [artifactContext, setArtifactContext] = useArtifactContext();
  const [artifactOpen, closeArtifact] = useArtifactOpen();

  const [threadId, _setThreadId] = useQueryState("threadId");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );
  const [input, setInput] = useState("");
  const {
    contentBlocks,
    setContentBlocks,
    dropRef,
    removeBlock,
    resetBlocks: _resetBlocks,
    dragOver,
    handlePaste,
  } = useFileUpload();
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);
  const [inputFocused, setInputFocused] = useState(false);
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");

  const [exploreOpen, setExploreOpen] = useState(false);
  const [surpriseMeOpen, setSurpriseMeOpen] = useState(false);
  const [cards, setCards] = useState<ExploreCard[]>([]);
  const [fetchStatus, setFetchStatus] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [apiUrl] = useQueryState("apiUrl", { defaultValue: DEFAULT_API_URL });
  const [authScheme] = useQueryState("authScheme", { defaultValue: process.env.NEXT_PUBLIC_AUTH_SCHEME ?? "" });

  const stream = useStreamContext();
  const messages = stream.messages;
  const isLoading = stream.isLoading;
  const isHITLActive = !!stream.interrupt;

  const lastError = useRef<string | undefined>(undefined);
  // Captures the stick-to-bottom context so submit handlers (which live outside
  // the StickToBottom provider) can force a scroll-to-bottom on send.
  const stickToBottomRef = useRef<StickToBottomContext | null>(null);
  const { theme, setTheme } = useTheme();

  const setThreadId = (id: string | null) => {
    _setThreadId(id);
    closeArtifact();
    setArtifactContext({});
  };

  const fetchExplore = async () => {
    setFetchStatus("loading");
    try {
      const client = createClient(
        apiUrl || "http://localhost:2024",
        getApiKey() ?? undefined,
        authScheme || undefined,
      );
      const result = await client.runs.wait(null, "explore", { input: {} }) as { cards?: ExploreCard[] };
      setCards(result.cards ?? []);
      setFetchStatus("done");
    } catch {
      setFetchStatus("error");
    }
  };

  const handleExploreToggle = () => {
    const next = !exploreOpen;
    setExploreOpen(next);
    if (next && fetchStatus === "idle") {
      void fetchExplore();
    }
  };

  const handleSurpriseMe = () => setSurpriseMeOpen(true);

  useEffect(() => {
    if (!stream.error) {
      lastError.current = undefined;
      return;
    }
    try {
      const message = (stream.error as any).message;
      if (!message || lastError.current === message) return;
      lastError.current = message;
      toast.error("An error occurred. Please try again.", {
        description: (
          <p>
            <strong>Error:</strong> <code>{message}</code>
          </p>
        ),
        richColors: true,
        closeButton: true,
      });
    } catch {
      // no-op
    }
  }, [stream.error]);

  const prevMessageLength = useRef(0);
  useEffect(() => {
    if (
      messages.length !== prevMessageLength.current &&
      messages?.length &&
      messages[messages.length - 1].type === "ai"
    ) {
      setFirstTokenReceived(true);
    }
    prevMessageLength.current = messages.length;
  }, [messages]);

  /* ── Submit from text input ── */
  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (isHITLActive || (input.trim().length === 0 && contentBlocks.length === 0) || isLoading) return;
    submitMessage(input);
  };

  /* ── Submit from agent selector form ── */
  const handleAgentSubmit = (text: string) => {
    if (!text.trim() || isLoading) return;
    submitMessage(text);
  };

  /* ── Shared submit logic ── */
  const submitMessage = (text: string) => {
    setFirstTokenReceived(false);

    const newHumanMessage: Message = {
      id: uuidv4(),
      type: "human",
      content: [
        ...(text.trim().length > 0 ? [{ type: "text", text }] : []),
        ...contentBlocks,
      ] as Message["content"],
    };

    const toolMessages = ensureToolCallsHaveResponses(stream.messages);
    const context =
      Object.keys(artifactContext).length > 0 ? artifactContext : undefined;

    stream.submit(
      { messages: [...toolMessages, newHumanMessage], context },
      {
        config: { recursion_limit: 100 },
        streamMode: ["values"],
        streamSubgraphs: true,
        streamResumable: true,
        optimisticValues: (prev) => ({
          ...prev,
          context,
          messages: [
            ...(prev.messages ?? []),
            ...toolMessages,
            newHumanMessage,
          ],
        }),
      },
    );

    setInput("");
    setContentBlocks([]);

    // Snap to the bottom on send (even if the user had scrolled up) and re-engage
    // the auto-follow lock so the streaming reply keeps scrolling into view. rAF
    // waits for the optimistic human message to render first.
    requestAnimationFrame(() => stickToBottomRef.current?.scrollToBottom());
  };

  const handleRegenerate = (parentCheckpoint: Checkpoint | null | undefined) => {
    prevMessageLength.current = prevMessageLength.current - 1;
    setFirstTokenReceived(false);
    stream.submit(undefined, {
      checkpoint: parentCheckpoint,
      config: { recursion_limit: 100 },
      streamMode: ["values"],
      streamSubgraphs: true,
      streamResumable: true,
    });
  };

  const chatStarted = !!threadId || !!messages.length;
  const hasNoAIOrToolMessages = !messages.find(
    (m) => m.type === "ai" || m.type === "tool",
  );

  return (
    <div className="relative flex h-screen w-full overflow-hidden bg-[--background]">
      <OceanBackground />

      {/* ── Sidebar ── */}
      <div className="relative hidden lg:flex">
        <motion.div
          className="absolute z-20 h-full overflow-hidden border-r border-[--border] bg-[--sidebar]/90 backdrop-blur-xl"
          style={{ width: 300 }}
          animate={{ x: chatHistoryOpen ? 0 : -300 }}
          initial={{ x: -300 }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          <div
            className="pointer-events-none absolute inset-y-0 right-0 w-px"
            style={{
              background:
                "linear-gradient(to bottom, transparent, oklch(0.6 0.2 210 / 0.4), transparent)",
            }}
          />
          <div className="relative h-full" style={{ width: 300 }}>
            <ThreadHistory />
          </div>
        </motion.div>
      </div>

      {/* ── Main content ── */}
      <div
        className={cn(
          "grid w-full grid-cols-[1fr_0fr] transition-all duration-500",
          artifactOpen && "grid-cols-[3fr_2fr]",
        )}
      >
        <motion.div
          className={cn(
            "relative flex min-w-0 flex-1 flex-col overflow-hidden",
            !chatStarted && "grid-rows-[1fr]",
          )}
          layout={isLargeScreen}
          animate={{
            marginLeft: chatHistoryOpen ? (isLargeScreen ? 300 : 0) : 0,
            width: chatHistoryOpen
              ? isLargeScreen
                ? "calc(100% - 300px)"
                : "100%"
              : "100%",
          }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          {/* Header — pre-chat */}
          {!chatStarted && (
            <motion.div
              className="absolute top-0 left-0 z-10 flex w-full items-center justify-between gap-3 p-3 pl-4"
              initial={{ opacity: 0, y: -12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
            >
              <div>
                {(!chatHistoryOpen || !isLargeScreen) && (
                  <PanelToggleButton
                    isOpen={chatHistoryOpen ?? false}
                    onToggle={() => setChatHistoryOpen((p) => !p)}
                  />
                )}
              </div>
              <div className="absolute top-3 right-4 flex items-center gap-1">
                <SurpriseMeButton onClick={handleSurpriseMe} disabled={isLoading} />
                <ExploreButton onClick={handleExploreToggle} active={exploreOpen} />
                <OpenGitHubRepo />
              </div>
            </motion.div>
          )}

          {/* Header — in-chat */}
          {chatStarted && (
            <motion.div
              className="relative z-10 flex items-center justify-between gap-3 border-b border-[--border]/60 bg-[--background]/70 px-3 py-2 backdrop-blur-md"
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35 }}
            >
              <div className="relative flex items-center justify-start gap-2">
                <div className="absolute left-0 z-10">
                  {(!chatHistoryOpen || !isLargeScreen) && (
                    <PanelToggleButton
                      isOpen={chatHistoryOpen ?? false}
                      onToggle={() => setChatHistoryOpen((p) => !p)}
                    />
                  )}
                </div>
                <motion.div
                  animate={{ marginLeft: !chatHistoryOpen ? 44 : 0 }}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                >
                  <BrandLogo onClick={() => setThreadId(null)} />
                </motion.div>
              </div>

              <div className="flex items-center gap-2">
                <SurpriseMeButton onClick={handleSurpriseMe} disabled={isLoading} />
                <ExploreButton onClick={handleExploreToggle} active={exploreOpen} />
                <OpenGitHubRepo />
                <TooltipIconButton
                  size="lg"
                  className={cn(
                    "h-9 w-9 rounded-full border border-[--border] bg-[--card]/50",
                    "text-[--muted-foreground] backdrop-blur-sm",
                    "hover:border-[--ring] hover:text-[--foreground]",
                    "hover:shadow-[0_0_10px_var(--glow-primary)] transition-all duration-200",
                  )}
                  tooltip="New thread"
                  variant="ghost"
                  onClick={() => setThreadId(null)}
                >
                  <SquarePen className="size-4" />
                </TooltipIconButton>
              </div>

              <div className="from-background/80 to-background/0 absolute inset-x-0 top-full h-6 bg-gradient-to-b" />
            </motion.div>
          )}

          {/* Messages */}
          <StickToBottom
            className="relative flex-1 overflow-hidden"
            contextRef={stickToBottomRef}
          >
            <StickyToBottomContent
              className={cn(
                "absolute inset-0 overflow-y-scroll px-4",
                "[&::-webkit-scrollbar]:w-1",
                "[&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-[--border]",
                "[&::-webkit-scrollbar-track]:bg-transparent",
                !chatStarted && "flex flex-col items-stretch pt-6 pb-[28vh]",
                chatStarted && "grid grid-rows-[1fr_auto]",
              )}
              contentClassName="pt-8 pb-16 max-w-3xl mx-auto flex flex-col gap-4 w-full"
              content={
                <AnimatePresence initial={false}>
                  {messages
                    .filter((m) => !m.id?.startsWith(DO_NOT_RENDER_ID_PREFIX))
                    .map((message, index) => (
                      <motion.div
                        key={message.id || `${message.type}-${index}`}
                        initial={{ opacity: 0, y: 16 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
                      >
                        {message.type === "human" ? (
                          <HumanMessage message={message} isLoading={isLoading} />
                        ) : (
                          <AssistantMessage
                            message={message}
                            isLoading={isLoading}
                            handleRegenerate={handleRegenerate}
                          />
                        )}
                      </motion.div>
                    ))}
                  {/* Render a standalone interrupt bubble when no AssistantMessage
                      will host it: either there are no AI/tool messages yet, or the
                      graph interrupted right after a human turn (last message is human,
                      e.g. plan_check's HITL after the user supplies trip details). */}
                  {(hasNoAIOrToolMessages ||
                    messages[messages.length - 1]?.type === "human") &&
                    !!stream.interrupt && (
                    <motion.div
                      key="interrupt-msg"
                      initial={{ opacity: 0, y: 16 }}
                      animate={{ opacity: 1, y: 0 }}
                    >
                      <AssistantMessage
                        message={undefined}
                        isLoading={isLoading}
                        handleRegenerate={handleRegenerate}
                      />
                    </motion.div>
                  )}
                  {isLoading && !firstTokenReceived && (
                    <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                      <AssistantMessageLoading />
                    </motion.div>
                  )}
                </AnimatePresence>
              }
              footer={
                <div
                  className={cn(
                    "flex flex-col items-center gap-4 pt-4 pb-4",
                    !chatStarted && "my-auto",
                    chatStarted &&
                      "sticky bottom-0 bg-gradient-to-t from-[--background] via-[--background]/95 to-transparent",
                  )}
                >

                  {/* ── Welcome hero ── */}
                  {!chatStarted && <PlaneHero />}

                  {/* ── Agent Selector (pre-chat only) ── */}
                  <AnimatePresence>
                    {!chatStarted && (
                      <motion.div
                        className="w-full"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -6 }}
                        transition={{ duration: 0.35, delay: 0.1 }}
                      >
                        <AgentSelector onSubmit={handleAgentSubmit} />
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Divider between agent selector and text input */}
                  {!chatStarted && (
                    <div className="flex w-full max-w-3xl items-center gap-3">
                      <div className="h-px flex-1 bg-[--border]" />
                      <span className="text-xs text-[--muted-foreground]">Or type freely</span>
                      <div className="h-px flex-1 bg-[--border]" />
                    </div>
                  )}

                  <ScrollToBottom className="animate-in fade-in-0 zoom-in-95 absolute bottom-full left-1/2 mb-4 -translate-x-1/2" />

                  {/* ── Input box ── */}
                  <div
                    ref={dropRef}
                    onFocusCapture={() => setInputFocused(true)}
                    onBlurCapture={() => setInputFocused(false)}
                    className={cn(
                      "relative z-10 mx-auto mb-4 w-full max-w-3xl rounded-2xl",
                      "border bg-[--card]/80 backdrop-blur-md",
                      "transition-all duration-300",
                      isHITLActive
                        ? "cursor-not-allowed border-[--border] opacity-60"
                        : dragOver
                          ? "scale-[1.01] border-[--ring] shadow-[0_0_0_2px_var(--ring),0_4px_32px_var(--glow-accent)]"
                          : inputFocused
                            ? "border-[--ring] shadow-[0_0_0_2px_var(--ring),0_8px_40px_var(--glow-accent)]"
                            : "border-[--border] shadow-[0_4px_24px_oklch(0.3_0.1_255_/_0.1)] hover:border-[--ring]/50 hover:shadow-[0_4px_32px_oklch(0.3_0.12_250_/_0.12)]",
                    )}
                  >
                    <div
                      className="absolute inset-x-4 top-0 h-px rounded-full opacity-60"
                      style={{
                        background:
                          "linear-gradient(to right, transparent, var(--ring), transparent)",
                      }}
                    />

                    <form
                      onSubmit={handleSubmit}
                      className="mx-auto grid max-w-3xl grid-rows-[1fr_auto] gap-2"
                    >
                      <ContentBlocksPreview blocks={contentBlocks} onRemove={removeBlock} />
                      <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onPaste={handlePaste}
                        disabled={isHITLActive}
                        onKeyDown={(e) => {
                          if (isHITLActive) { e.preventDefault(); return; }
                          if (
                            e.key === "Enter" &&
                            !e.shiftKey &&
                            !e.metaKey &&
                            !e.nativeEvent.isComposing
                          ) {
                            e.preventDefault();
                            const el = e.target as HTMLElement | undefined;
                            const form = el?.closest("form");
                            form?.requestSubmit();
                          }
                        }}
                        placeholder={isHITLActive ? "Please respond using the options above" : "Type a question..."}
                        dir="auto"
                        className={cn(
                          "field-sizing-content resize-none border-none bg-transparent",
                          "p-4 pb-0 text-[--foreground] shadow-none ring-0 outline-none",
                          "placeholder:text-[--muted-foreground]/60",
                          "focus:ring-0 focus:outline-none",
                          isHITLActive && "cursor-not-allowed opacity-50",
                        )}
                      />

                      <div className="flex items-center gap-4 p-3 pt-2">
                        {stream.isLoading ? (
                          <motion.div className="ml-auto" whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}>
                            <Button
                              key="stop"
                              onClick={() => stream.stop()}
                              className={cn(
                                "gap-2 rounded-xl border border-[--border]",
                                "bg-[--destructive]/10 text-[--destructive-foreground]",
                                "hover:bg-[--destructive]/20",
                              )}
                              variant="ghost"
                            >
                              <LoaderCircle className="h-4 w-4 animate-spin" />
                              Cancel
                            </Button>
                          </motion.div>
                        ) : (
                          <motion.div className="ml-auto" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                            <Button
                              type="submit"
                              className={cn(
                                "gap-2 rounded-xl px-5",
                                "bg-[--primary] text-[--primary-foreground]",
                                "shadow-[0_0_16px_var(--glow-primary)]",
                                "hover:shadow-[0_0_28px_var(--glow-accent)]",
                                "disabled:opacity-40 disabled:shadow-none",
                                "transition-all duration-300",
                              )}
                              disabled={
                                isHITLActive ||
                                isLoading ||
                                (!input.trim() && contentBlocks.length === 0)
                              }
                            >
                              <Send className="h-4 w-4" />
                              Send
                            </Button>
                          </motion.div>
                        )}
                      </div>
                    </form>
                  </div>
                </div>
              }
            />
          </StickToBottom>
        </motion.div>

        {/* ── Artifact panel ── */}
        <div className="relative flex flex-col border-l border-[--border]/60">
          <div className="absolute inset-0 flex min-w-[30vw] flex-col bg-[--card]/60 backdrop-blur-sm">
            <div className="grid grid-cols-[1fr_auto] items-center border-b border-[--border]/60 bg-[--card]/80 px-4 py-3 backdrop-blur-md">
              <ArtifactTitle className="truncate overflow-hidden text-sm font-medium text-[--foreground]" />
              <motion.button
                onClick={closeArtifact}
                className={cn(
                  "flex h-7 w-7 items-center justify-center rounded-full",
                  "text-[--muted-foreground] transition-all duration-200",
                  "hover:bg-[--muted] hover:text-[--foreground]",
                )}
                whileHover={{ scale: 1.12, rotate: 90 }}
                whileTap={{ scale: 0.9 }}
              >
                <XIcon className="size-4" />
              </motion.button>
            </div>
            <ArtifactContent className="relative flex-grow" />
          </div>
        </div>
      </div>

      {/* ── Surprise Me modal ── */}
      <SurpriseMeModal
        open={surpriseMeOpen}
        onClose={() => setSurpriseMeOpen(false)}
        onSubmit={handleAgentSubmit}
      />

      {/* ── Explore overlay ── */}
      <AnimatePresence>
        {exploreOpen && (
          <motion.div
            className="fixed inset-0 z-50 overflow-auto bg-[--background]/95 backdrop-blur-md"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            <div className="min-h-full p-6">
              {/* Header */}
              <div className="mb-6 flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <Compass className="size-5 text-[--ring]" />
                  <h2 className="text-lg font-semibold text-[--foreground]">Explore Destinations</h2>
                </div>
                <motion.button
                  onClick={() => setExploreOpen(false)}
                  className={cn(
                    "flex h-9 w-9 items-center justify-center rounded-full",
                    "border border-[--border] bg-[--card]/60 backdrop-blur-sm",
                    "text-[--muted-foreground] transition-all duration-200",
                    "hover:border-[--ring] hover:text-[--foreground]",
                    "hover:shadow-[0_0_12px_var(--glow-accent)]",
                  )}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.92 }}
                >
                  <XIcon className="size-4" />
                </motion.button>
              </div>

              {/* Loading */}
              {fetchStatus === "loading" && <ExploreLoadingScreen />}

              {/* Error */}
              {fetchStatus === "error" && (
                <div className="flex flex-col items-center gap-4 py-24">
                  <p className="text-sm text-red-500">Could not load explore data.</p>
                  <motion.button
                    onClick={fetchExplore}
                    className={cn(
                      "rounded-xl border border-[--border] bg-[--card]/70 px-5 py-2.5",
                      "text-sm text-[--foreground] transition-all duration-200",
                      "hover:border-[--ring] hover:shadow-[0_0_12px_var(--glow-accent)]",
                    )}
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                  >
                    Retry
                  </motion.button>
                </div>
              )}

              {/* Cards */}
              {fetchStatus === "done" && cards.length > 0 && (
                <ExploreView cards={cards} />
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}