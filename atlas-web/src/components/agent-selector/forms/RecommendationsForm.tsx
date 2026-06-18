"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  Globe,
  Calendar,
  Star,
  ChevronDown,
} from "lucide-react";
import { inputCls, labelCls, FormSubmitButton } from "../shared";

export function RecommendationsForm({ onSubmit }: { onSubmit: (text: string) => void }) {
  const [origin, setOrigin] = useState("");
  const [month, setMonth] = useState("");
  const [topic, setTopic] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});

  const locationRe = /^[a-zA-ZÀ-ɏ\s-]+$/;

  const handleSubmit = () => {
    if (!origin.trim()) { setErrors({ origin: "Required" }); return; }
    if (!locationRe.test(origin.trim())) { setErrors({ origin: "Please enter a valid location (letters only)" }); return; }
    setErrors({});
    const effectiveMonth = month || new Date().toLocaleString("en", { month: "long" });
    const parts = [
      "I'm looking for travel recommendations",
      `from ${origin}`,
      `in ${effectiveMonth}`,
      topic && `about ${topic}`,
    ].filter(Boolean);
    onSubmit(parts.join(" ") + ".");
  };

  return (
    <div className="grid grid-cols-2 gap-3">
      <div>
        <label className={labelCls}>
          <Globe className="mb-0.5 mr-1 inline size-3" />
          Origin country<span className="text-[--ring] ml-0.5">*</span>
        </label>
        <input className={cn(inputCls, errors.origin && "border-red-500")} placeholder="Israel" value={origin}
          onChange={(e) => { setOrigin(e.target.value); if (e.target.value.trim() && locationRe.test(e.target.value.trim())) setErrors({}); }} dir="ltr" />
        {errors.origin && <p className="mt-1 text-xs text-red-500">{errors.origin}</p>}
      </div>
      <div>
        <label className={labelCls}>
          <Calendar className="mb-0.5 mr-1 inline size-3" />
          Estimated date
        </label>
        <div className="relative">
          <select className={cn(inputCls, "appearance-none pr-8")} value={month} onChange={(e) => setMonth(e.target.value)} dir="ltr">
            <option value="">Select month...</option>
            {["January","February","March","April","May","June",
              "July","August","September","October","November","December"].map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
          <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 size-3.5 -translate-y-1/2 text-[--muted-foreground]" />
        </div>
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
        <FormSubmitButton onClick={handleSubmit} disabled={false}
          label="Get Recommendations" icon={<Star className="size-4" />} />
      </div>
    </div>
  );
}
