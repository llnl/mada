"use client";

import { useState } from "react";
import { AVAILABLE_MODELS, DEFAULT_MODEL } from "@/lib/constants";

interface TaskInputProps {
  onGenerate: (taskDescription: string, model: string) => void;
  disabled: boolean;
  defaultValue?: string;
}

export default function TaskInput({
  onGenerate,
  disabled,
  defaultValue = "",
}: TaskInputProps) {
  const [text, setText] = useState(defaultValue);
  const [model, setModel] = useState<string>(DEFAULT_MODEL);

  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium text-gray-700">
        Describe a task and let AI design a multi-agent workflow for you
      </label>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder=""
        rows={5}
        className="w-full rounded-lg border border-gray-300 px-4 py-3 text-sm
                   focus:border-mada-500 focus:ring-2 focus:ring-mada-100
                   disabled:bg-gray-50 disabled:text-gray-500
                   placeholder:text-gray-400 resize-y"
        disabled={disabled}
      />
      <div className="flex flex-wrap items-end gap-3">
        <button
          onClick={() => text.trim() && onGenerate(text.trim(), model)}
          disabled={disabled || !text.trim()}
          className="rounded-lg bg-mada-600 px-6 py-2.5 text-sm font-medium text-white
                     hover:bg-mada-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                     transition-colors"
        >
          {disabled ? "Generating..." : "Generate Plan"}
        </button>
        <div className="flex flex-col">
          <label className="text-xs font-medium text-gray-600">
            Planner model
          </label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={disabled}
            className="input-field mt-0.5"
            title="Which LLM designs the workflow. Agents at runtime use whichever model you pick after the plan is generated."
          >
            {AVAILABLE_MODELS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}
