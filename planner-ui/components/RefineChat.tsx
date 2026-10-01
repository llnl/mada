"use client";

import { useState } from "react";

interface RefineChatProps {
  onRefine: (feedback: string) => Promise<void>;
  disabled?: boolean;
}

/**
 * A small chat-style input for iteratively refining the current plan via
 * natural language. Sits alongside the plan graph so users can say things
 * like "make agents higher-level" or "add a summarizer" instead of clicking
 * through each agent's detail panel.
 */
export default function RefineChat({ onRefine, disabled = false }: RefineChatProps) {
  const [feedback, setFeedback] = useState("");
  const [pending, setPending] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = feedback.trim();
    if (!text || pending || disabled) return;
    setPending(true);
    try {
      await onRefine(text);
      setFeedback("");
    } finally {
      setPending(false);
    }
  };

  return (
    <div>
      <div className="text-xs font-medium text-gray-500 mb-2">
        Refine plan with feedback
      </div>
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          placeholder={pending ? "Refining plan…" : ""}
          disabled={pending || disabled}
          className="input-field flex-1"
        />
        <button
          type="submit"
          disabled={pending || disabled || !feedback.trim()}
          className="rounded-lg bg-mada-600 px-4 py-2 text-sm font-medium text-white
                     hover:bg-mada-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {pending ? "Refining…" : "Refine"}
        </button>
      </form>
    </div>
  );
}
