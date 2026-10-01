"use client";

import type { ExecutionStep } from "@/lib/types";

interface ExecutionStepsProps {
  steps: ExecutionStep[];
}

export default function ExecutionSteps({ steps }: ExecutionStepsProps) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-gray-900">Execution Steps</h3>
      <ol className="space-y-2">
        {steps.map((step) => (
          <li
            key={step.step_number}
            className="flex items-start gap-3 rounded-lg border border-gray-100
                       bg-white px-3 py-2 text-sm"
          >
            <span className="flex h-6 w-6 shrink-0 items-center justify-center
                             rounded-full bg-mada-50 text-xs font-semibold text-mada-700">
              {step.step_number}
            </span>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-medium text-gray-900">
                  {step.agents.join(", ")}
                </span>
                {step.parallel && (
                  <span className="rounded-full bg-blue-50 px-2 py-0.5 text-xs
                                   text-blue-700 border border-blue-200">
                    parallel
                  </span>
                )}
              </div>
              <p className="mt-0.5 text-gray-500">{step.description}</p>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
