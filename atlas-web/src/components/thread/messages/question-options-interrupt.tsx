import { useState } from "react";
import { motion } from "framer-motion";
import { useStreamContext } from "@/providers/Stream";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { MarkdownText } from "../markdown-text";
import { QuestionOptionsInterrupt } from "@/lib/question-options-interrupt";

/**
 * Interactive renderer for the itinerary HITL interrupt shape
 * (`{ question, options }`). Lets the user pick an option (or type a
 * free-text answer when `options` is empty) and resumes the paused run via
 * `stream.submit(undefined, { command: { resume: <choice> } })`.
 */
export function QuestionOptionsInterruptView({
  interrupt,
}: {
  interrupt: QuestionOptionsInterrupt;
}) {
  const thread = useStreamContext();
  const [submitted, setSubmitted] = useState<string | null>(null);
  const [freeText, setFreeText] = useState("");

  const isLoading = thread.isLoading;
  const isFreeText = interrupt.options.length === 0;

  const resume = (choice: string) => {
    if (!choice.trim() || isLoading || submitted !== null) return;
    setSubmitted(choice);
    thread.submit(undefined, {
      command: { resume: choice },
      config: { recursion_limit: 100 },
      streamMode: ["values"],
      streamSubgraphs: true,
      streamResumable: true,
    });
  };

  return (
    <div className="overflow-hidden rounded-lg border border-gray-200">
      <div className="border-b border-gray-200 bg-gray-50 px-4 py-2">
        <h3 className="font-medium text-gray-900">Your input is needed</h3>
      </div>

      <div className="flex flex-col gap-3 p-4">
        <div className="text-sm text-gray-800">
          <MarkdownText>{interrupt.question}</MarkdownText>
        </div>

        {isFreeText ? (
          <form
            className="flex flex-col gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              resume(freeText);
            }}
          >
            <Textarea
              value={freeText}
              onChange={(e) => setFreeText(e.target.value)}
              disabled={isLoading || submitted !== null}
              placeholder="Describe the changes you'd like to make…"
              rows={3}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault();
                  resume(freeText);
                }
              }}
            />
            <Button
              type="submit"
              className="self-start"
              disabled={
                isLoading || submitted !== null || freeText.trim().length === 0
              }
            >
              Submit
            </Button>
          </form>
        ) : (
          <div className="flex flex-col gap-2">
            {interrupt.options.map(([value, label]) => (
              <motion.div key={value} whileTap={{ scale: 0.99 }}>
                <Button
                  variant={submitted === value ? "default" : "outline"}
                  className="w-full justify-start text-left whitespace-normal"
                  disabled={isLoading || submitted !== null}
                  onClick={() => resume(value)}
                >
                  {label}
                </Button>
              </motion.div>
            ))}
          </div>
        )}

        {submitted !== null && (
          <p className="text-xs text-gray-500">Sending your response…</p>
        )}
      </div>
    </div>
  );
}
