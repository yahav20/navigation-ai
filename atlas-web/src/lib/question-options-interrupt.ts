/**
 * Type guard + types for the itinerary HITL interrupt shape emitted by the
 * Python agent (see src/agent/itinerary/plan_check.py and critic.py):
 *
 *   interrupt({ "question": "...", "options": [("yes", "label"), ...] })
 *
 * `options` may be empty, in which case the user is expected to provide a
 * free-text answer (e.g. the critic's "adjust preferences" branch).
 */

/** A single option tuple `[value, label]` as serialized from the Python side. */
export type QuestionOption = [string, string];

export interface QuestionOptionsInterrupt {
  question: string;
  options: QuestionOption[];
}

function isQuestionOption(value: unknown): value is QuestionOption {
  return (
    Array.isArray(value) &&
    value.length === 2 &&
    typeof value[0] === "string" &&
    typeof value[1] === "string"
  );
}

/**
 * Unwrap a raw interrupt into its `value` payload. The LangGraph SDK delivers
 * interrupts as `{ value, ... }` (or an array of them); the agent payload may
 * also arrive already unwrapped.
 */
function unwrapInterruptValue(value: unknown): unknown {
  const candidate = Array.isArray(value) ? value[0] : value;
  if (candidate && typeof candidate === "object" && "value" in candidate) {
    return (candidate as { value: unknown }).value;
  }
  return candidate;
}

export function getQuestionOptionsInterrupt(
  value: unknown,
): QuestionOptionsInterrupt | null {
  const payload = unwrapInterruptValue(value);
  if (!payload || typeof payload !== "object") {
    return null;
  }

  const { question, options } = payload as Partial<QuestionOptionsInterrupt>;
  if (typeof question !== "string") {
    return null;
  }
  if (!Array.isArray(options) || !options.every(isQuestionOption)) {
    return null;
  }

  return { question, options };
}

export function isQuestionOptionsInterrupt(
  value: unknown,
): value is QuestionOptionsInterrupt {
  return getQuestionOptionsInterrupt(value) !== null;
}
