import type { GeneratedOutput, MCPServer, PlannedWorkflow } from "./types";
import { DEFAULT_MODEL } from "./constants";

// Prefer direct backend URL when NEXT_PUBLIC_API_URL is set (bypasses Next.js
// proxy which has ECONNRESET issues under Next 14 dev). Falls back to relative
// URLs (Next.js rewrites) otherwise.
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export async function generatePlan(
  taskDescription: string,
  model: string = DEFAULT_MODEL
): Promise<PlannedWorkflow> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/plan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_description: taskDescription, model }),
    });
  } catch (fetchError) {
    throw new Error(
      `Network error connecting to ${API_BASE}/api/plan - is the backend running?`
    );
  }
  if (!res.ok) {
    let detail = await res.text();
    try {
      const json = JSON.parse(detail);
      detail = json.detail || detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

export async function refinePlan(
  workflow: PlannedWorkflow,
  feedback: string,
  model: string = DEFAULT_MODEL
): Promise<PlannedWorkflow> {
  const res = await fetch(`${API_BASE}/api/refine-plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ workflow, feedback, model }),
  });
  if (!res.ok) {
    let detail = await res.text();
    try {
      const json = JSON.parse(detail);
      detail = json.detail || detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

export async function generateOutput(
  workflow: PlannedWorkflow
): Promise<GeneratedOutput> {
  const res = await fetch(`${API_BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(workflow),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Code generation failed: ${err}`);
  }
  return res.json();
}

export async function fetchMCPServers(): Promise<Record<string, MCPServer>> {
  const res = await fetch(`${API_BASE}/api/mcp-servers`);
  if (!res.ok) return {};
  return res.json();
}

export interface ExecutionCallbacks {
  onChunk: (content: string) => void;
  onStatus: (message: string, tools: number) => void;
  onAgentStart: (agent: string) => void;
  onAgentChunk: (agent: string, content: string) => void;
  onAgentResult: (agent: string, result: string) => void;
  onDone: () => void;
  onError: (message: string) => void;
  // Post-execution chat: arrive only after the initial workflow's `done`
  onFollowupStart: (message: string) => void;
  onFollowupChunk: (content: string) => void;
  onFollowupDone: () => void;
}

/**
 * Send a follow-up user message on an already-open execution WebSocket.
 * Backend routes it to the same orchestrator instance.
 */
export function sendFollowup(ws: WebSocket, message: string): void {
  ws.send(JSON.stringify({ message }));
}

export function createExecutionSocket(
  workflow: PlannedWorkflow,
  callbacks: ExecutionCallbacks
): WebSocket {
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://127.0.0.1:8000";
  // Wormhole community subtoken: browsers can't set custom WS headers, so
  // send it as a query param. Backend accepts both header and query fallback.
  const token =
    (typeof window !== "undefined" && window.localStorage.getItem("wormhole_subtoken")) ||
    process.env.NEXT_PUBLIC_WORMHOLE_SUBTOKEN ||
    "";
  const url = token
    ? `${wsUrl}/ws/execute?token=${encodeURIComponent(token)}`
    : `${wsUrl}/ws/execute`;
  const ws = new WebSocket(url);

  ws.onopen = () => {
    ws.send(JSON.stringify(workflow));
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    switch (data.type) {
      case "chunk":
        callbacks.onChunk(data.content);
        break;
      case "status":
        callbacks.onStatus(data.message, data.tools);
        break;
      case "agent_start":
        callbacks.onAgentStart(data.agent);
        break;
      case "agent_chunk":
        callbacks.onAgentChunk(data.agent, data.content);
        break;
      case "agent_result":
        callbacks.onAgentResult(data.agent, data.result);
        break;
      case "done":
        callbacks.onDone();
        // Note: don't close the socket. The backend keeps it open for
        // post-execution follow-up messages.
        break;
      case "followup_start":
        callbacks.onFollowupStart(data.message);
        break;
      case "followup_chunk":
        callbacks.onFollowupChunk(data.content);
        break;
      case "followup_done":
        callbacks.onFollowupDone();
        break;
      case "error":
        callbacks.onError(data.message);
        ws.close();
        break;
    }
  };

  ws.onerror = () => {
    callbacks.onError("WebSocket connection failed");
  };

  return ws;
}
