import { parsePartialJson } from "@langchain/core/output_parsers";
import { useStreamContext } from "@/providers/Stream";
import { AIMessage, Checkpoint, Message } from "@langchain/langgraph-sdk";
import { useStream } from "@langchain/langgraph-sdk/react";
import { getContentString } from "../utils";
import { BranchSwitcher, CommandBar } from "./shared";
import { MarkdownText } from "../markdown-text";
import { LoadExternalComponent } from "@langchain/langgraph-sdk/react-ui";
import ItineraryViewer from "../ItineraryViewer";
import { cn } from "@/lib/utils";
import { ToolCalls, ToolResult } from "./tool-calls";
import { MessageContentComplex } from "@langchain/core/messages";
import { isAgentInboxInterruptSchema } from "@/lib/agent-inbox-interrupt";
import { ThreadView } from "../agent-inbox";
import { useQueryState, parseAsBoolean } from "nuqs";
import { GenericInterruptView } from "./generic-interrupt";
import { getQuestionOptionsInterrupt } from "@/lib/question-options-interrupt";
import { QuestionOptionsInterruptView } from "./question-options-interrupt";
import { useArtifact } from "../artifact";
import { Fragment } from "react";

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * For a given AI message, return the UI components that belong to it.
 *
 * Priority:
 *  1. Any UI item whose metadata.message_id matches the message exactly.
 *  2. Fallback (only for the LAST AI message): the single most-recent
 *     ItineraryViewer in the whole ui list (handles Python AIMessage ID
 *     mismatch). We intentionally pick only ONE so that old viewers from
 *     earlier turns don't resurface.
 *
 * FIX #1 & #2: previously the fallback returned ALL ItineraryViewer items,
 * causing (a) every edit to stack a new viewer on the last message and
 * (b) old viewers persisting throughout the conversation.
 */
function resolveCustomComponents(
  allUi: ReturnType<typeof useStreamContext>["values"]["ui"],
  messageId: string | undefined,
  isLastAiMessage: boolean,
) {
  const ui = allUi ?? [];

  // Exact match by message_id — always preferred, works for all component types
  const direct = messageId
    ? ui.filter((item) => item.metadata?.message_id === messageId)
    : [];
  if (direct.length > 0) return direct;

  // Fallback: only the LATEST ItineraryViewer, only on the last AI message
  if (isLastAiMessage) {
    const latestItinerary = [...ui]
      .reverse()
      .find((item) => item.name === "ItineraryViewer");
    return latestItinerary ? [latestItinerary] : [];
  }

  return [];
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function CustomComponent({
  message,
  thread,
}: {
  message: Message;
  thread: ReturnType<typeof useStreamContext>;
}) {
  const artifact = useArtifact();
  const { values } = useStreamContext();
  const allUi = values.ui ?? [];

  const lastAiMessageId = thread.messages
    .filter((m) => m.type === "ai")
    .at(-1)?.id;
  const isLastAiMessage = message.id === lastAiMessageId;

  const customComponents = resolveCustomComponents(allUi, message.id, isLastAiMessage);

  if (!customComponents.length) return null;

  return (
    <Fragment key={message.id}>
      {customComponents.map((customComponent, index) => (
        <LoadExternalComponent
          key={customComponent.id ?? `custom-ui-${index}`}
          stream={thread as unknown as ReturnType<typeof useStream>}
          message={customComponent}
          meta={{ ui: customComponent, artifact }}
          components={{ ItineraryViewer }}
        />
      ))}
    </Fragment>
  );
}

/**
 * Returns true when this specific message has an ItineraryViewer attached to it.
 * Used to suppress the plain-text content when the rich viewer is shown.
 *
 * FIX #3: previously only checked metadata.message_id, so the text was never
 * suppressed for the fallback case (last AI message without an exact id match).
 */
function useHasItineraryViewer(
  message: Message | undefined,
  isLastAiMessage: boolean,
): boolean {
  const { values } = useStreamContext();
  const allUi = values.ui ?? [];
  if (!message) return false;

  const components = resolveCustomComponents(allUi, message.id, isLastAiMessage);
  return components.some((c) => c.name === "ItineraryViewer");
}

/** Reads the latest node activity from progress_log in stream values */
function useAgentStatus(): { node: string; detail: string } | null {
  const { values, isLoading } = useStreamContext();
  if (!isLoading) return null;
  const log: string[] = (values as any).progress_log ?? [];
  if (!log.length) return null;

  const last = log[log.length - 1].trim();
  const nodeMatch = last.match(/\*\*([a-z_]+)\*\*/i);
  const stepMatch = last.match(/`([a-z_]+)`/i);

  const node = nodeMatch?.[1] ?? "";
  const detail = stepMatch?.[1] ?? last.replace(/\*\*/g, "").replace(/`/g, "").slice(0, 60);

  return { node, detail };
}

function parseAnthropicStreamedToolCalls(
  content: MessageContentComplex[],
): AIMessage["tool_calls"] {
  const toolCallContents = content.filter((c) => c.type === "tool_use" && c.id);

  return toolCallContents.map((tc) => {
    const toolCall = tc as Record<string, any>;
    let json: Record<string, any> = {};
    if (toolCall?.input) {
      try {
        json = parsePartialJson(toolCall.input) ?? {};
      } catch {
        // Pass
      }
    }
    return {
      name: toolCall.name ?? "",
      id: toolCall.id ?? "",
      args: json,
      type: "tool_call",
    };
  });
}

interface InterruptProps {
  interrupt?: unknown;
  isLastMessage: boolean;
  hasNoAIOrToolMessages: boolean;
}

function Interrupt({
  interrupt,
  isLastMessage,
  hasNoAIOrToolMessages,
}: InterruptProps) {
  const fallbackValue = Array.isArray(interrupt)
    ? (interrupt as Record<string, any>[])
    : (((interrupt as { value?: unknown } | undefined)?.value ??
        interrupt) as Record<string, any>);

  const showInterrupt = isLastMessage || hasNoAIOrToolMessages;
  const isAgentInbox = isAgentInboxInterruptSchema(interrupt);
  const questionOptions = isAgentInbox
    ? null
    : getQuestionOptionsInterrupt(interrupt);

  return (
    <>
      {isAgentInbox && showInterrupt && <ThreadView interrupt={interrupt} />}
      {questionOptions && showInterrupt && (
        <QuestionOptionsInterruptView interrupt={questionOptions} />
      )}
      {interrupt &&
      !isAgentInbox &&
      !questionOptions &&
      showInterrupt ? (
        <GenericInterruptView interrupt={fallbackValue} />
      ) : null}
    </>
  );
}

// ─── Main exports ─────────────────────────────────────────────────────────────

export function AssistantMessage({
  message,
  isLoading,
  handleRegenerate,
}: {
  message: Message | undefined;
  isLoading: boolean;
  handleRegenerate: (parentCheckpoint: Checkpoint | null | undefined) => void;
}) {
  const content = message?.content ?? [];
  const contentString = getContentString(content);
  const [hideToolCalls] = useQueryState(
    "hideToolCalls",
    parseAsBoolean.withDefault(false),
  );

  const thread = useStreamContext();

  const lastMessage = thread.messages.length > 0
    ? thread.messages[thread.messages.length - 1]
    : undefined;
  const isLastMessage = message === undefined || lastMessage?.id === message?.id;

  // Compute isLastAiMessage here so we can pass it to both hooks consistently
  const lastAiMessageId = thread.messages.filter((m) => m.type === "ai").at(-1)?.id;
  const isLastAiMessage = !!message && message.id === lastAiMessageId;

  const hasItineraryViewer = useHasItineraryViewer(message, isLastAiMessage);

  const hasNoAIOrToolMessages = !thread.messages.find(
    (m) => m.type === "ai" || m.type === "tool",
  );
  const meta = message ? thread.getMessagesMetadata(message) : undefined;
  const threadInterrupt = thread.interrupt;

  const parentCheckpoint = meta?.firstSeenState?.parent_checkpoint;
  const anthropicStreamedToolCalls = Array.isArray(content)
    ? parseAnthropicStreamedToolCalls(content)
    : undefined;

  const hasToolCalls =
    message &&
    "tool_calls" in message &&
    message.tool_calls &&
    message.tool_calls.length > 0;
  const toolCallsHaveContents =
    hasToolCalls &&
    message.tool_calls?.some(
      (tc) => tc.args && Object.keys(tc.args).length > 0,
    );
  const hasAnthropicToolCalls = !!anthropicStreamedToolCalls?.length;
  const isToolResult = message?.type === "tool";

  if (isToolResult && hideToolCalls) {
    return null;
  }

  return (
    <div className="group mr-auto flex w-full items-start gap-2">
      <div className="flex w-full flex-col gap-2">
        {isToolResult ? (
          <>
            <ToolResult message={message} />
            <Interrupt
              interrupt={threadInterrupt}
              isLastMessage={isLastMessage}
              hasNoAIOrToolMessages={hasNoAIOrToolMessages}
            />
          </>
        ) : (
          <>
            {contentString.length > 0 && !hasItineraryViewer && (
              <div className="py-1">
                <MarkdownText>{contentString}</MarkdownText>
              </div>
            )}

            {!hideToolCalls && (
              <>
                {(hasToolCalls && toolCallsHaveContents && (
                  <ToolCalls toolCalls={message.tool_calls} />
                )) ||
                  (hasAnthropicToolCalls && (
                    <ToolCalls toolCalls={anthropicStreamedToolCalls} />
                  )) ||
                  (hasToolCalls && (
                    <ToolCalls toolCalls={message.tool_calls} />
                  ))}
              </>
            )}

            {message && (
              <CustomComponent
                message={message}
                thread={thread}
              />
            )}
            <Interrupt
              interrupt={threadInterrupt}
              isLastMessage={isLastMessage}
              hasNoAIOrToolMessages={hasNoAIOrToolMessages}
            />
            <div
              className={cn(
                "mr-auto flex items-center gap-2 transition-opacity",
                "opacity-0 group-focus-within:opacity-100 group-hover:opacity-100",
              )}
            >
              <BranchSwitcher
                branch={meta?.branch}
                branchOptions={meta?.branchOptions}
                onSelect={(branch) => thread.setBranch(branch)}
                isLoading={isLoading}
              />
              <CommandBar
                content={contentString}
                isLoading={isLoading}
                isAiMessage={true}
                handleRegenerate={() => handleRegenerate(parentCheckpoint)}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export function AssistantMessageLoading() {
  const { values } = useStreamContext();
  const log: string[] = (values as any).progress_log ?? [];

  let nodeName = "";
  let stepDetail = "";

  for (let i = log.length - 1; i >= 0; i--) {
    const entry = log[i].trim();
    const nodeMatch = entry.match(/\*\*([a-zA-Z_][a-zA-Z0-9_]*)\*\*/);
    const stepMatch = entry.match(/`([a-z_]+)`/i);
    if (!nodeName && nodeMatch) nodeName = nodeMatch[1];
    if (!stepDetail && stepMatch) stepDetail = stepMatch[1];
    if (nodeName && stepDetail) break;
  }

  const NODE_LABELS: Record<string, string> = {
    itinerary_executor:   "🗓️ Building itinerary",
    itinerary_planner:    "🧠 Planning",
    itinerary_replanner:  "🔄 Replanning",
    itinerary_critic:     "🔍 Reviewing budget",
    itinerary_formatter:  "✍️ Writing summary",
    advisor_executor:     "🔎 Searching",
    advisor_planner:      "🧠 Planning search",
    advisor_formatter:    "✍️ Preparing answer",
    travel_agent:         "✈️ Fetching travel data",
    summary_node:         "📝 Summarising",
    enrichment_node:      "💬 Gathering details",
    intent_classifier:    "🤔 Understanding request",
  };

  const STEP_LABELS: Record<string, string> = {
    fetch_weather:          "weather data",
    fetch_activities:       "attractions & restaurants",
    fetch_avg_prices:       "average prices",
    fetch_min_prices:       "best prices",
    build_day_schedule:     "day schedule",
    switch_travel_options:  "travel options",
    verify_budget:          "budget check",
  };

  const friendlyNode = nodeName ? (NODE_LABELS[nodeName] ?? nodeName.replace(/_/g, " ")) : "";
  const friendlyStep = stepDetail ? (STEP_LABELS[stepDetail] ?? stepDetail.replace(/_/g, " ")) : "";

  const hasStatus = !!(friendlyNode || friendlyStep);

  return (
    <div className="mr-auto flex flex-col gap-2">
      <div className="bg-muted flex h-8 items-center gap-1 rounded-2xl px-4 py-2">
        <div className="bg-foreground/50 h-1.5 w-1.5 animate-[pulse_1.5s_ease-in-out_infinite] rounded-full" />
        <div className="bg-foreground/50 h-1.5 w-1.5 animate-[pulse_1.5s_ease-in-out_0.5s_infinite] rounded-full" />
        <div className="bg-foreground/50 h-1.5 w-1.5 animate-[pulse_1.5s_ease-in-out_1s_infinite] rounded-full" />
        {hasStatus && (
          <span className="ml-2 text-xs text-muted-foreground animate-pulse">
            {friendlyNode}
            {friendlyNode && friendlyStep && (
              <span className="mx-1 opacity-40">·</span>
            )}
            {friendlyStep && (
              <span className="opacity-70">{friendlyStep}</span>
            )}
          </span>
        )}
      </div>
    </div>
  );
}