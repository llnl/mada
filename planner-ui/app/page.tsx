"use client";

import { useCallback, useEffect, useState } from "react";

import type {
  AppState,
  ChatMessage,
  GeneratedOutput,
  PlannedAgent,
  PlannedWorkflow,
} from "@/lib/types";
import {
  createExecutionSocket,
  fetchMCPServers,
  generateOutput,
  generatePlan,
  refinePlan,
  sendFollowup,
} from "@/lib/api";

import TaskInput from "@/components/TaskInput";
import WorkflowGraph from "@/components/WorkflowGraph";
import AgentDetailPanel from "@/components/AgentDetailPanel";
import ExecutionSteps from "@/components/ExecutionSteps";
import OrchestrationSettings from "@/components/OrchestrationSettings";
import CodeOutput from "@/components/CodeOutput";
import ActionBar from "@/components/ActionBar";
import ChatPanel from "@/components/ChatPanel";
import RefineChat from "@/components/RefineChat";

export default function PlannerPage() {
  const [appState, setAppState] = useState<AppState>("input");
  const [workflow, setWorkflow] = useState<PlannedWorkflow | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [output, setOutput] = useState<GeneratedOutput | null>(null);
  const [mcpServers, setMcpServers] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [taskText, setTaskText] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [executionSocket, setExecutionSocket] = useState<WebSocket | null>(null);
  const [followupInFlight, setFollowupInFlight] = useState(false);
  const [hasExecuted, setHasExecuted] = useState(false);
  // Which tab is currently visible. "run" is the chat/execution view.
  const [activeTab, setActiveTab] = useState<"plan" | "run" | "code">("plan");

  useEffect(() => {
    fetchMCPServers()
      .then((servers) => setMcpServers(Object.keys(servers)))
      .catch((err) => console.warn("Failed to load MCP servers:", err));
  }, []);

  const [plannerModel, setPlannerModel] = useState<string>("");

  const handleGenerate = useCallback(
    async (taskDescription: string, model: string) => {
      setError(null);
      setAppState("planning");
      setTaskText(taskDescription);
      setPlannerModel(model);
      setSelectedAgent(null);
      setOutput(null);
      setHasExecuted(false);
      setChatMessages([]);
      setActiveTab("plan");
      try {
        const plan = await generatePlan(taskDescription, model);
        setWorkflow(plan);
        setAppState("review");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Plan generation failed");
        setAppState("input");
      }
    },
    []
  );

  const handleRegenerate = useCallback(() => {
    if (taskText && plannerModel) {
      handleGenerate(taskText, plannerModel);
    }
  }, [taskText, plannerModel, handleGenerate]);

  const handleRefine = useCallback(
    async (feedback: string) => {
      if (!workflow) return;
      setError(null);
      try {
        const refined = await refinePlan(workflow, feedback, plannerModel || undefined);
        setWorkflow(refined);
        setSelectedAgent(null);
        // Clear any stale generated output; the plan changed
        setOutput(null);
        setHasExecuted(false);
        setChatMessages([]);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Plan refinement failed");
      }
    },
    [workflow, plannerModel]
  );

  const handleApprove = useCallback(async () => {
    if (!workflow) return;
    setError(null);
    setAppState("generating");
    try {
      const result = await generateOutput(workflow);
      setOutput(result);
      setAppState("generated");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Code generation failed");
      setAppState("review");
    }
  }, [workflow]);

  const handleExecute = useCallback(() => {
    if (!workflow) return;
    setAppState("executing");
    setHasExecuted(true);
    setActiveTab("run");
    // Close any prior WebSocket so a re-execute opens a fresh session
    if (executionSocket && executionSocket.readyState === WebSocket.OPEN) {
      executionSocket.close();
    }
    // Seed the unified chat with the initial task as a user message and an
    // empty assistant message that will fill with agent activity + response.
    setChatMessages([
      { role: "user", content: workflow.task_description },
      { role: "assistant", content: "", agents: [], streaming: true },
    ]);

    // Helper: patch the last assistant message in the chat.
    const patchLastAssistant = (fn: (msg: ChatMessage) => ChatMessage) =>
      setChatMessages((prev) => {
        for (let i = prev.length - 1; i >= 0; i--) {
          if (prev[i].role === "assistant") {
            const next = [...prev];
            next[i] = fn(next[i]);
            return next;
          }
        }
        return prev;
      });

    const ws = createExecutionSocket(workflow, {
      onChunk: (content) =>
        patchLastAssistant((msg) => ({ ...msg, content: msg.content + content })),
      onStatus: (message, tools) =>
        patchLastAssistant((msg) => ({
          ...msg,
          status: `${message} (${tools} tools)`,
        })),
      onAgentStart: (agent) =>
        patchLastAssistant((msg) => ({
          ...msg,
          agents: [
            ...(msg.agents || []),
            { agent, status: "running", result: "" },
          ],
        })),
      onAgentChunk: (agent, content) =>
        patchLastAssistant((msg) => {
          const agents = msg.agents || [];
          // Append text to the latest running instance of this agent.
          for (let i = agents.length - 1; i >= 0; i--) {
            if (agents[i].agent === agent && agents[i].status === "running") {
              const next = [...agents];
              next[i] = { ...next[i], result: next[i].result + content };
              return { ...msg, agents: next };
            }
          }
          return msg;
        }),
      onAgentResult: (agent, result) =>
        patchLastAssistant((msg) => {
          const agents = msg.agents || [];
          const next = agents.map((a) =>
            a.agent === agent && a.status === "running"
              ? { ...a, status: "done" as const, result: a.result || result }
              : a
          );
          return { ...msg, agents: next };
        }),
      onDone: () => {
        patchLastAssistant((msg) => ({ ...msg, streaming: false }));
        setAppState("generated");
      },
      onError: (message) => {
        patchLastAssistant((msg) => ({
          ...msg,
          content: msg.content + `\n\n[Error] ${message}`,
          streaming: false,
        }));
        setAppState("generated");
        setFollowupInFlight(false);
      },
      onFollowupStart: (_message) => {
        // The user bubble was appended optimistically in handleSendFollowup.
        // Add a new empty assistant bubble for the follow-up's response +
        // any agent activity it triggers.
        setFollowupInFlight(true);
        setChatMessages((prev) => [
          ...prev,
          { role: "assistant", content: "", agents: [], streaming: true },
        ]);
      },
      onFollowupChunk: (content) =>
        patchLastAssistant((msg) => ({ ...msg, content: msg.content + content })),
      onFollowupDone: () => {
        setFollowupInFlight(false);
        patchLastAssistant((msg) => ({ ...msg, streaming: false }));
      },
    });
    setExecutionSocket(ws);
  }, [workflow]);

  const handleSendFollowup = useCallback(
    (message: string) => {
      if (!executionSocket || executionSocket.readyState !== WebSocket.OPEN) return;
      setChatMessages((prev) => [...prev, { role: "user", content: message }]);
      sendFollowup(executionSocket, message);
    },
    [executionSocket]
  );

  // Clean up the WebSocket on unmount
  useEffect(() => {
    return () => {
      if (executionSocket && executionSocket.readyState === WebSocket.OPEN) {
        executionSocket.close();
      }
    };
  }, [executionSocket]);

  const handleAddAgent = useCallback(() => {
    if (!workflow) return;
    const newAgent: PlannedAgent = {
      agent_name: `NewAgent${workflow.agents.length + 1}`,
      description: "Describe this agent's role",
      domain: "general",
      mcp_servers: [],
      instructions: "You are a helpful agent. Define your role and behavior here.",
      depends_on: [],
    };
    setWorkflow({
      ...workflow,
      agents: [...workflow.agents, newAgent],
    });
    setSelectedAgent(newAgent.agent_name);
  }, [workflow]);

  const handleRemoveAgent = useCallback(
    (agentName: string) => {
      if (!workflow) return;
      const newAgents = workflow.agents
        .filter((a) => a.agent_name !== agentName)
        .map((a) => ({
          ...a,
          depends_on: a.depends_on.filter((d) => d !== agentName),
        }));
      setWorkflow({ ...workflow, agents: newAgents });
      setSelectedAgent(null);
    },
    [workflow]
  );

  return (
    <div className="mx-auto max-w-none px-6 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">MADA</h1>
      </div>

      {/* Task Input */}
      <div className="mb-6">
        <TaskInput
          onGenerate={handleGenerate}
          disabled={appState === "planning"}
          defaultValue={taskText}
        />
      </div>

      {/* Error */}
      {error && (
        <div className="mb-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Planning spinner */}
      {appState === "planning" && (
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full
                            border-4 border-mada-200 border-t-mada-600" />
            <p className="text-sm text-gray-500">
              Designing your workflow...
            </p>
          </div>
        </div>
      )}

      {workflow && appState !== "input" && appState !== "planning" && (
        <>
          <div className="mb-4 flex items-center gap-1 border-b-2 border-gray-300">

            {chatMessages.length > 0 && (
              <TabButton
                label="Run"
                hint={`${chatMessages.length} messages`}
                active={activeTab === "run"}
                onClick={() => setActiveTab("run")}
              />
            )}
            <TabButton
              label="Plan"
              hint={`${workflow.agents.length} agents`}
              active={activeTab === "plan"}
              onClick={() => setActiveTab("plan")}
            />
            {output && (
              <TabButton
                label="Code"
                hint="JSON & Python"
                active={activeTab === "code"}
                onClick={() => setActiveTab("code")}
              />
            )}
          </div>

          {activeTab === "plan" && (
            <PlanContent
              workflow={workflow}
              selectedAgent={selectedAgent}
              setSelectedAgent={setSelectedAgent}
              setWorkflow={setWorkflow}
              handleRefine={handleRefine}
              handleRemoveAgent={handleRemoveAgent}
              mcpServers={mcpServers}
              appState={appState}
              hasExecuted={hasExecuted}
              handleRegenerate={handleRegenerate}
              handleApprove={handleApprove}
              handleExecute={handleExecute}
              handleAddAgent={handleAddAgent}
            />
          )}

          {activeTab === "run" && chatMessages.length > 0 && (
            <ChatPanel
              messages={chatMessages}
              onSend={handleSendFollowup}
              disabled={
                appState === "executing" ||
                followupInFlight ||
                !executionSocket ||
                executionSocket.readyState !== WebSocket.OPEN
              }
              placeholder={
                appState === "executing"
                  ? "Workflow running…"
                  : !executionSocket ||
                    executionSocket.readyState !== WebSocket.OPEN
                  ? "Connection closed"
                  : followupInFlight
                  ? "Waiting…"
                  : ""
              }
            />
          )}

          {activeTab === "code" && output && <CodeOutput output={output} />}
        </>
      )}
    </div>
  );
}

interface TabButtonProps {
  label: string;
  hint?: string;
  active: boolean;
  onClick: () => void;
}

function TabButton({ label, hint, active, onClick }: TabButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`px-4 py-2 text-sm font-medium -mb-[2px] border-b-2 transition-colors ${
        active
          ? "border-mada-600 text-mada-700"
          : "border-transparent text-gray-500 hover:text-gray-800 hover:border-gray-300"
      }`}
    >
      {label}
      {hint && (
        <span className={`ml-2 text-xs ${
          active ? "text-mada-500" : "text-gray-400"
        }`}>
          {hint}
        </span>
      )}
    </button>
  );
}

/**
 * Shared render of the plan tab: graph, right settings panel, reasoning,
 * refine chat, and the action bar. Always uses the two-column layout since
 * the tab takes the full page width.
 */
interface PlanContentProps {
  workflow: PlannedWorkflow;
  selectedAgent: string | null;
  setSelectedAgent: (name: string | null) => void;
  setWorkflow: (w: PlannedWorkflow | null) => void;
  handleRefine: (feedback: string) => Promise<void>;
  handleRemoveAgent: (name: string) => void;
  mcpServers: string[];
  appState: AppState;
  hasExecuted: boolean;
  handleRegenerate: () => void;
  handleApprove: () => void;
  handleExecute: () => void;
  handleAddAgent: () => void;
}

function PlanContent({
  workflow,
  selectedAgent,
  setSelectedAgent,
  setWorkflow,
  handleRefine,
  handleRemoveAgent,
  mcpServers,
  appState,
  hasExecuted,
  handleRegenerate,
  handleApprove,
  handleExecute,
  handleAddAgent,
}: PlanContentProps) {
  return (
    <div>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <WorkflowGraph
            workflow={workflow}
            selectedAgent={selectedAgent}
            onSelectAgent={setSelectedAgent}
          />
          <div className="mt-3">
            <div className="text-xs font-medium text-gray-500 mb-1">
              Reasoning
            </div>
            <p className="text-sm text-gray-700 leading-relaxed">
              {workflow.reasoning}
            </p>
          </div>
          <div className="mt-4">
            <RefineChat
              onRefine={handleRefine}
              disabled={appState === "executing"}
            />
          </div>
        </div>

        <div className="max-h-[600px] overflow-y-auto">
          {selectedAgent ? (
            <AgentDetailPanel
              workflow={workflow}
              agentName={selectedAgent}
              mcpServerOptions={mcpServers}
              onUpdateWorkflow={setWorkflow}
              onRemoveAgent={handleRemoveAgent}
            />
          ) : (
            <div className="space-y-4">
              <OrchestrationSettings
                workflow={workflow}
                onUpdateWorkflow={setWorkflow}
              />
              <ExecutionSteps steps={workflow.execution_steps} />
            </div>
          )}
        </div>
      </div>

      <div className="mt-6">
        <ActionBar
          appState={appState}
          hasExecuted={hasExecuted}
          onRegenerate={handleRegenerate}
          onApprove={handleApprove}
          onExecute={handleExecute}
          onAddAgent={handleAddAgent}
        />
      </div>
    </div>
  );
}
