"use client";

import type { AppState } from "@/lib/types";

interface ActionBarProps {
  appState: AppState;
  hasExecuted: boolean;
  onRegenerate: () => void;
  onApprove: () => void;
  onExecute: () => void;
  onAddAgent: () => void;
}

export default function ActionBar({
  appState,
  hasExecuted,
  onRegenerate,
  onApprove,
  onExecute,
  onAddAgent,
}: ActionBarProps) {
  return (
    <div className="flex items-center gap-3">
      {(appState === "review" || appState === "generated") && (
        <>
          <button onClick={onAddAgent} className="btn-secondary">
            + Add Agent
          </button>
          <button onClick={onRegenerate} className="btn-secondary">
            Regenerate Plan
          </button>
        </>
      )}
      {appState === "review" && (
        <button onClick={onApprove} className="btn-primary">
          Approve &amp; Generate
        </button>
      )}
      {appState === "generated" && !hasExecuted && (
        <button onClick={onExecute} className="btn-primary">
          Execute Now
        </button>
      )}
      {appState === "generated" && hasExecuted && (
        <>
          <button
            disabled
            className="rounded-lg bg-gray-200 px-4 py-2 text-sm font-medium
                       text-gray-500 cursor-not-allowed"
            title="Already executed. Use the chat below for follow-ups, or Regenerate Plan for a fresh run."
          >
            ✓ Executed
          </button>
          <button
            onClick={onExecute}
            className="btn-secondary"
            title="Start a fresh execution with this plan (opens a new session)"
          >
            Re-execute
          </button>
        </>
      )}
      {appState === "generating" && (
        <span className="text-sm text-gray-500">Generating code...</span>
      )}
    </div>
  );
}
