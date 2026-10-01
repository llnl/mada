"use client";

import ReactMarkdown from "react-markdown";
import type { ChatMessage } from "@/lib/types";

interface AssistantMessageProps {
  message: ChatMessage;
}

export default function AssistantMessage({ message }: AssistantMessageProps) {
  const { content, streaming, agents, status } = message;
  const hasAgents = agents && agents.length > 0;

  return (
    <div className="flex justify-start">
      <div className="w-full rounded-lg bg-gray-100 px-4 py-3 text-sm text-gray-900">
        {status && (
          <div className="mb-2 text-xs text-gray-500 italic">{status}</div>
        )}

        {hasAgents && (
          <div className="mb-2 space-y-1">
            {agents.map((agent, i) => (
              <AgentActivity key={`${agent.agent}-${i}`} agent={agent} />
            ))}
          </div>
        )}

        {(content || streaming) && (
          <div className="prose prose-sm max-w-none text-gray-900
                          prose-pre:bg-gray-800 prose-pre:text-gray-100
                          prose-code:text-mada-700 prose-code:bg-mada-50
                          prose-code:before:content-none prose-code:after:content-none
                          prose-img:rounded-lg prose-img:border prose-img:border-gray-200
                          prose-img:max-h-[600px]">
            <ReactMarkdown
              components={{
                // Open image URLs in a new tab on click, and constrain size
                img: ({ src, alt }) => (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={typeof src === "string" ? src : ""}
                    alt={alt || ""}
                    className="rounded-lg border border-gray-200 max-h-[600px] my-2"
                  />
                ),
                a: ({ href, children }) => (
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-mada-700 underline hover:text-mada-900"
                  >
                    {children}
                  </a>
                ),
              }}
            >
              {content}
            </ReactMarkdown>
            {streaming && (
              <span className="ml-1 inline-block h-3 w-1 animate-pulse bg-current" />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function AgentActivity({ agent }: { agent: import("@/lib/types").AgentExecution }) {
  // In agent-as-tool mode we don't get per-agent output from mada: specialist
  // outputs go silently back to the coordinator LLM as tool returns. Cards
  // just show start/done state with no body.
  const isRunning = agent.status === "running";
  const hasBody = agent.result && agent.result !== "(completed)";

  return (
    <div className={`flex items-center gap-2 rounded border px-2 py-1.5 text-xs ${
      isRunning ? "border-mada-300 bg-mada-50" : "border-gray-200 bg-white"
    }`}>
      {isRunning ? (
        <span className="h-2.5 w-2.5 animate-spin rounded-full border-2 border-mada-200 border-t-mada-600" />
      ) : (
        <span className="h-2.5 w-2.5 rounded-full bg-green-500" />
      )}
      <span className="font-semibold text-gray-900">{agent.agent}</span>
      <span className={`text-[10px] px-1.5 py-0 rounded-full ${
        isRunning ? "bg-mada-100 text-mada-700" : "bg-green-100 text-green-700"
      }`}>
        {isRunning ? "running" : "done"}
      </span>
      {hasBody && (
        <span className="ml-auto text-[10px] text-gray-400" title={agent.result}>
          {agent.result.length} chars
        </span>
      )}
    </div>
  );
}
