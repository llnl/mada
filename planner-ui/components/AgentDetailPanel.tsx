"use client";

import type { PlannedAgent, PlannedWorkflow } from "@/lib/types";

interface AgentDetailPanelProps {
  workflow: PlannedWorkflow;
  agentName: string;
  mcpServerOptions: string[];
  onUpdateWorkflow: (workflow: PlannedWorkflow) => void;
  onRemoveAgent: (agentName: string) => void;
}

export default function AgentDetailPanel({
  workflow,
  agentName,
  mcpServerOptions,
  onUpdateWorkflow,
  onRemoveAgent,
}: AgentDetailPanelProps) {
  const agent = workflow.agents.find((a) => a.agent_name === agentName);
  if (!agent) return null;

  const updateAgent = (updates: Partial<PlannedAgent>) => {
    const newAgents = workflow.agents.map((a) =>
      a.agent_name === agentName ? { ...a, ...updates } : a
    );
    onUpdateWorkflow({ ...workflow, agents: newAgents });
  };

  const otherAgents = workflow.agents
    .filter((a) => a.agent_name !== agentName)
    .map((a) => a.agent_name);

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-gray-900">
        Agent Details: {agentName}
      </h3>

      <Field label="Name">
        <input
          type="text"
          value={agent.agent_name}
          onChange={(e) => {
            const newName = e.target.value;
            // Also update references in other agents' depends_on
            const newAgents = workflow.agents.map((a) => {
              if (a.agent_name === agentName) {
                return { ...a, agent_name: newName };
              }
              return {
                ...a,
                depends_on: a.depends_on.map((d) =>
                  d === agentName ? newName : d
                ),
              };
            });
            onUpdateWorkflow({ ...workflow, agents: newAgents });
          }}
          className="input-field"
        />
      </Field>

      <Field label="Domain">
        <input
          type="text"
          value={agent.domain}
          onChange={(e) => updateAgent({ domain: e.target.value })}
          className="input-field"
        />
      </Field>

      <Field label="Description">
        <textarea
          value={agent.description}
          onChange={(e) => updateAgent({ description: e.target.value })}
          rows={2}
          className="input-field resize-y"
        />
      </Field>

      <Field label="MCP Servers">
        <div className="space-y-1 max-h-32 overflow-y-auto">
          {mcpServerOptions.map((server) => (
            <label key={server} className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={agent.mcp_servers.includes(server)}
                onChange={(e) => {
                  const servers = e.target.checked
                    ? [...agent.mcp_servers, server]
                    : agent.mcp_servers.filter((s) => s !== server);
                  updateAgent({ mcp_servers: servers });
                }}
                className="rounded border-gray-300 text-mada-600"
              />
              {server}
            </label>
          ))}
        </div>
      </Field>

      <Field label="Instructions">
        <textarea
          value={agent.instructions}
          onChange={(e) => updateAgent({ instructions: e.target.value })}
          rows={6}
          className="input-field resize-y text-xs"
        />
      </Field>

      <Field label="Depends On">
        <div className="space-y-1 max-h-32 overflow-y-auto">
          {otherAgents.map((name) => (
            <label key={name} className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={agent.depends_on.includes(name)}
                onChange={(e) => {
                  const deps = e.target.checked
                    ? [...agent.depends_on, name]
                    : agent.depends_on.filter((d) => d !== name);
                  updateAgent({ depends_on: deps });
                }}
                className="rounded border-gray-300 text-mada-600"
              />
              {name}
            </label>
          ))}
        </div>
      </Field>

      <button
        onClick={() => onRemoveAgent(agentName)}
        className="w-full rounded-lg border border-red-200 bg-red-50 px-4 py-2
                   text-sm text-red-700 hover:bg-red-100 transition-colors"
      >
        Remove Agent
      </button>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="mb-1 block text-xs font-medium text-gray-600">
        {label}
      </label>
      {children}
    </div>
  );
}
