"use client";

import type { PlannedWorkflow } from "@/lib/types";
import { AVAILABLE_MODELS } from "@/lib/constants";

interface OrchestrationSettingsProps {
  workflow: PlannedWorkflow;
  onUpdateWorkflow: (workflow: PlannedWorkflow) => void;
}

export default function OrchestrationSettings({
  workflow,
  onUpdateWorkflow,
}: OrchestrationSettingsProps) {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-gray-900">
        Orchestration Settings
      </h3>
      <div className="grid grid-cols-3 gap-3">
        <div>
          <label className="mb-1 block text-xs font-medium text-gray-600">
            Strategy
          </label>
          <select
            value={workflow.orchestration_type}
            onChange={(e) =>
              onUpdateWorkflow({
                ...workflow,
                orchestration_type: e.target.value as
                  | "agent-as-tool"
                  | "magentic",
              })
            }
            className="input-field"
          >
            <option value="agent-as-tool">agent-as-tool</option>
            <option value="magentic">magentic</option>
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-gray-600">
            Max Turns
          </label>
          <input
            type="number"
            value={workflow.max_turns}
            onChange={(e) =>
              onUpdateWorkflow({
                ...workflow,
                max_turns: parseInt(e.target.value) || 20,
              })
            }
            min={1}
            max={100}
            className="input-field"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-gray-600">
            Model
          </label>
          <select
            value={workflow.model}
            onChange={(e) =>
              onUpdateWorkflow({ ...workflow, model: e.target.value })
            }
            className="input-field"
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
