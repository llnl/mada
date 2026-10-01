export interface PlannedAgent {
  agent_name: string;
  description: string;
  domain: string;
  mcp_servers: string[];
  instructions: string;
  depends_on: string[];
}

export interface ExecutionStep {
  step_number: number;
  agents: string[];
  description: string;
  parallel: boolean;
}

export interface PlannedWorkflow {
  task_description: string;
  orchestration_type: "agent-as-tool" | "magentic";
  max_turns: number;
  max_stalls: number;
  model: string;
  agents: PlannedAgent[];
  execution_steps: ExecutionStep[];
  reasoning: string;
}

export interface GeneratedOutput {
  json_config: Record<string, unknown>;
  python_code: string;
}

export interface MCPServer {
  transport: string;
  url?: string;
  command?: string;
  description: string;
}

export interface AgentExecution {
  agent: string;
  status: "running" | "done";
  result: string;
}

/**
 * A rich chat message. Assistant messages can carry BOTH final response text
 * AND a list of agent activity that happened while producing it. Renderers
 * display agents inline (expandable) and text as the main body.
 */
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;               // true while assistant response is still coming in
  agents?: AgentExecution[];         // assistant only: agent activity during this turn
  status?: string;                   // assistant only: mada init status message
}

export type AppState =
  | "input"
  | "planning"
  | "review"
  | "generating"
  | "generated"
  | "executing";
