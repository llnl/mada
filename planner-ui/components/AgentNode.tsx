"use client";

import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";

interface AgentNodeData {
  label: string;
  domain: string;
  mcpServers: string[];
  [key: string]: unknown;
}

export default function AgentNode({ data, selected }: NodeProps) {
  const { label, domain, mcpServers } = data as unknown as AgentNodeData;

  return (
    <div
      className={`rounded-lg border bg-white px-4 py-3 shadow-sm min-w-[200px] max-w-[260px]
                  transition-all ${
                    selected
                      ? "border-mada-600 shadow-md ring-2 ring-mada-100"
                      : "border-gray-200 hover:border-mada-400"
                  }`}
    >
      <Handle type="target" position={Position.Top} className="!bg-mada-500" />
      <Handle id="loop-in" type="target" position={Position.Left} className="!bg-orange-500" />
      <Handle id="loop-out" type="source" position={Position.Left} className="!bg-orange-500" />

      {/* Primary: agent name */}
      <div className="text-sm font-semibold text-gray-900 leading-tight truncate">
        {label as string}
      </div>

      <div className="mt-0.5 text-xs text-gray-500 leading-tight">
        {domain as string}
      </div>


      {(mcpServers as string[]).length > 0 && (
        <div className="mt-2 flex flex-wrap gap-x-2 gap-y-1">
          {(mcpServers as string[]).map((server) => (
            <span
              key={server}
              className="inline-flex items-center gap-1 text-xs text-amber-700"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
              {server}
            </span>
          ))}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} className="!bg-mada-500" />
    </div>
  );
}
