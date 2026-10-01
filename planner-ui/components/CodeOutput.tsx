"use client";

import { useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";

import type { GeneratedOutput } from "@/lib/types";

interface CodeOutputProps {
  output: GeneratedOutput;
}

type Tab = "json" | "python";

export default function CodeOutput({ output }: CodeOutputProps) {
  const [activeTab, setActiveTab] = useState<Tab>("json");

  const jsonStr = JSON.stringify(output.json_config, null, 2);

  const tabs: { key: Tab; label: string }[] = [
    { key: "json", label: "JSON Config" },
    { key: "python", label: "Python Code" },
  ];

  const download = (content: string, filename: string) => {
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copy = async (content: string) => {
    await navigator.clipboard.writeText(content);
  };

  return (
    <div className="space-y-3">
      <div className="flex gap-1 border-b border-gray-200">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.key
                ? "border-mada-600 text-mada-700"
                : "border-transparent text-gray-500 hover:text-gray-700"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="max-h-[70vh] overflow-auto rounded-lg border border-gray-200">
        <SyntaxHighlighter
          language={activeTab === "json" ? "json" : "python"}
          style={oneLight}
          customStyle={{ margin: 0, fontSize: "12px" }}
          showLineNumbers
        >
          {activeTab === "json" ? jsonStr : output.python_code}
        </SyntaxHighlighter>
      </div>

      <div className="flex gap-2">
        <button
          onClick={() => download(jsonStr, "workflow_config.json")}
          className="btn-secondary"
        >
          Download JSON
        </button>
        <button
          onClick={() => download(output.python_code, "run_workflow.py")}
          className="btn-secondary"
        >
          Download Python
        </button>
        <button
          onClick={() =>
            copy(activeTab === "json" ? jsonStr : output.python_code)
          }
          className="btn-secondary"
        >
          Copy to Clipboard
        </button>
      </div>
    </div>
  );
}
