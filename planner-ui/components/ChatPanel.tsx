"use client";

import { useEffect, useRef, useState } from "react";
import type { ChatMessage } from "@/lib/types";
import AssistantMessage from "./AssistantMessage";

interface ChatPanelProps {
  messages: ChatMessage[];
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
  emptyHint?: string;
}

export default function ChatPanel({
  messages,
  onSend,
  disabled = false,
  placeholder = "",
  emptyHint,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || disabled) return;
    onSend(text);
    setInput("");
  };

  return (
    <div className="space-y-3">
      {messages.length > 0 ? (
        <div className="max-h-[75vh] overflow-y-auto space-y-3 pr-2">
          {messages.map((msg, i) =>
            msg.role === "user" ? (
              <UserMessage key={i} content={msg.content} />
            ) : (
              <AssistantMessage key={i} message={msg} />
            )
          )}
          <div ref={bottomRef} />
        </div>
      ) : emptyHint ? (
        <div className="py-8 text-center text-sm text-gray-400">{emptyHint}</div>
      ) : null}

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          className="input-field flex-1"
        />
        <button
          type="submit"
          disabled={disabled || !input.trim()}
          className="rounded-lg bg-mada-600 px-4 py-2 text-sm font-medium text-white
                     hover:bg-mada-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  );
}

function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-lg bg-mada-600 px-3 py-2 text-sm text-white">
        <pre className="whitespace-pre-wrap font-sans">{content}</pre>
      </div>
    </div>
  );
}
