import { v4 as uuidv4 } from "uuid";
import { ReactNode, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStreamContext } from "@/providers/Stream";
import { useState, FormEvent } from "react";
import { Button } from "../ui/button";
import { Checkpoint, Message } from "@langchain/langgraph-sdk";
import { AssistantMessage, AssistantMessageLoading } from "./messages/ai";
import { HumanMessage } from "./messages/human";
import { useTheme } from "next-themes";
import { Moon, Sun } from "lucide-react";
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
import { Switch } from "../ui/switch";
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

/* ─────────────────────────────────────────
   Ambient background: animated ocean orbs
───────────────────────────────────────── */
function OceanBackground() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <motion.div
        className="absolute -top-40 -left-40 h-[600px] w-[600px] rounded-full"
        style={{
          background:
            "radial-gradient(ellipse, oklch(0.55 0.22 240 / 0.18) 0%, transparent 70%)",
          filter: "blur(60px)",
        }}
        animate={{ x: [0, 40, 0], y: [0, 30, 0], scale: [1, 1.08, 1] }}
        transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute -bottom-32 -right-32 h-[500px] w-[500px] rounded-full"
        style={{
          background:
            "radial-gradient(ellipse, oklch(0.65 0.2 195 / 0.14) 0%, transparent 70%)",
          filter: "blur(50px)",
        }}
        animate={{ x: [0, -30, 0], y: [0, -20, 0], scale: [1, 1.12, 1] }}
        transition={{ duration: 22, repeat: Infinity, ease: "easeInOut", delay: 3 }}
      />
      <motion.div
        className="absolute top-20 right-1/3 h-[260px] w-[260px] rounded-full"
        style={{
          background:
            "radial-gradient(ellipse, oklch(0.7 0.18 210 / 0.1) 0%, transparent 70%)",
          filter: "blur(40px)",
        }}
        animate={{ x: [0, 20, -10, 0], y: [0, -25, 10, 0] }}
        transition={{ duration: 14, repeat: Infinity, ease: "easeInOut", delay: 6 }}
      />
    </div>
  );
}

function GridOverlay() {
  return (
    <div
      className="pointer-events-none fixed inset-0 -z-10 opacity-[0.025] dark:opacity-[0.04]"
      style={{
        backgroundImage:
          "linear-gradient(oklch(0.4 0.1 240) 1px, transparent 1px), linear-gradient(90deg, oklch(0.4 0.1 240) 1px, transparent 1px)",
        backgroundSize: "48px 48px",
      }}
    />
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
  const [hideToolCalls, setHideToolCalls] = useQueryState(
    "hideToolCalls",
    parseAsBoolean.withDefault(false),
  );
  const [input, setInput] = useState("");
  const {
    contentBlocks,
    setContentBlocks,
    handleFileUpload,
    dropRef,
    removeBlock,
    resetBlocks: _resetBlocks,
    dragOver,
    handlePaste,
  } = useFileUpload();
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);
  const [inputFocused, setInputFocused] = useState(false);
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");

  const stream = useStreamContext();
  const messages = stream.messages;
  const isLoading = stream.isLoading;

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
    if ((input.trim().length === 0 && contentBlocks.length === 0) || isLoading) return;
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
      <GridOverlay />

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
              <div className="absolute top-3 right-4 flex items-center">
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
                  {hasNoAIOrToolMessages && !!stream.interrupt && (
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
                  {!chatStarted && (
                    <motion.div
                      className="flex flex-col items-center gap-2 text-center"
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.5, ease: "easeOut" }}
                    >
                      <div className="relative mb-1">
                        <div
                          className="absolute inset-0 scale-150 rounded-full blur-2xl"
                          style={{ background: "var(--glow-accent)" }}
                        />
                        <LangGraphLogoSVG width={140} height={140} className="relative z-10 flex-shrink-0" />
                      </div>
                      <h1 className="text-3xl font-semibold tracking-tight" style={{ letterSpacing: "-0.03em" }}>
                        Navigation AI
                      </h1>
                      <p className="text-sm text-[--muted-foreground]">
                        Choose an agent or type a freeform question below
                      </p>
                    </motion.div>
                  )}

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
                      dragOver
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
                        onKeyDown={(e) => {
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
                        placeholder="Type a question..."
                        dir="auto"
                        className={cn(
                          "field-sizing-content resize-none border-none bg-transparent",
                          "p-4 pb-0 text-[--foreground] shadow-none ring-0 outline-none",
                          "placeholder:text-[--muted-foreground]/60",
                          "focus:ring-0 focus:outline-none",
                        )}
                      />

                      <div className="flex items-center gap-4 p-3 pt-2">
                        <div className="flex items-center space-x-2">
                          <Switch
                            id="render-tool-calls"
                            checked={hideToolCalls ?? false}
                            onCheckedChange={setHideToolCalls}
                          />
                          <Label
                            htmlFor="render-tool-calls"
                            className="cursor-pointer select-none text-xs text-[--muted-foreground]"
                          >
                            Hide Tool Calls
                          </Label>
                        </div>

                        <Label
                          htmlFor="file-input"
                          className="flex cursor-pointer items-center gap-1.5 text-xs text-[--muted-foreground] transition-colors hover:text-[--foreground]"
                        >
                          <Plus className="size-4" />
                          <span>Attach</span>
                        </Label>
                        <input
                          id="file-input"
                          type="file"
                          onChange={handleFileUpload}
                          multiple
                          accept="image/jpeg,image/png,image/gif,image/webp,application/pdf"
                          className="hidden"
                        />

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
    </div>
  );
}