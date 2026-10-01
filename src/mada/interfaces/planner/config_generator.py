# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generate MADA orchestrator JSON config from a PlannedWorkflow."""

from __future__ import annotations

import os
from typing import Any

from .plan_schema import PlannedWorkflow

# Endpoint defaults per provider, used when API_BASE_URL is not set at runtime
_DEFAULT_BASE_URL_BY_PROVIDER = {
    "livai": "https://livai-api.llnl.gov/v1",
    "openai": "https://api.openai.com/v1",
}

# Boilerplate output-formatting rules appended to every specialist agent's
# instructions. Ensures consistent markdown output across all agents without
# users having to remember to ask for it in each task. The frontend renders
# markdown via react-markdown, so anything an agent emits in these forms
# displays properly in the chat.
_OUTPUT_FORMAT_INSTRUCTIONS = """

## Output formatting

Format your responses as Markdown. The UI renders Markdown, so:

- Use `**bold**` and `*italic*` for emphasis
- Use `# / ## / ###` for section headings
- Use bulleted or numbered lists for enumerations
- Use fenced code blocks with a language tag for code samples
- Use tables when comparing data

**Images (CRITICAL):** Whenever your response mentions or discusses an image, \
you MUST embed it inline with Markdown image syntax so the user actually sees \
it. Never just describe an image without embedding it; always include the \
image tag alongside your description.

- Public URL: `![short description](https://example.com/image.png)`
- Local file produced by a tool on the mada host: use the /api/files/ prefix \
  so the browser can fetch it through the planner backend. Example: a tool \
  writing `/tmp/laghos_results/run0/plot.png` should be embedded as:

      ![plot](/api/files/tmp/laghos_results/run0/plot.png)

  (Strip the leading slash from the absolute path and prepend /api/files/.)

- **ALWAYS use absolute file paths.** When a tool returns or you mention a \
  file location, use the full absolute path starting with `/`, not a \
  relative path like `figs/plot.png` or `./output/mesh.g`. Absolute paths \
  are required for the UI to display images and for other agents to find \
  the file. If a tool returns a relative path, prepend its working \
  directory to make it absolute before mentioning it.
- Base64 data URLs also work: `![alt](data:image/png;base64,...)`

**URL reproduction rules (VERY IMPORTANT):**
- If the user's task or a tool's output contains an image URL, reproduce that \
  URL EXACTLY, character-for-character, in your markdown image tag.
- Do NOT shorten, canonicalize, "correct," or paraphrase URLs.
- Do NOT substitute a URL you remember for the one that was actually provided.
- Do NOT emit `![alt]()` with an empty URL. Either include the real URL or \
  omit the image tag entirely.
- If you are unsure of a URL, ask the user rather than inventing one.

**Links:** Use `[link text](https://...)` for clickable references. Same \
verbatim reproduction rule applies.
"""


def _augment_instructions(instructions: str) -> str:
    """Append boilerplate output-formatting rules to an agent's instructions."""
    base = (instructions or "").rstrip()
    return f"{base}\n{_OUTPUT_FORMAT_INSTRUCTIONS}"


def generate_config(
    workflow: PlannedWorkflow,
    mcp_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a PlannedWorkflow into a MADA-compatible JSON config dict.

    Args:
        workflow: The planned workflow to convert.
        mcp_registry: Optional dict of known MCP servers keyed by name.
            Each value should have transport, url/command, description fields.

    Returns:
        A dict that can be passed to ``AppConfig.from_dict()`` or written as JSON.
    """
    mcp_registry = mcp_registry or {}

    # Model section: provider chosen by MADA_PROVIDER env var.
    # "livai" (default): LLNL LiteLLM proxy in front of Azure OpenAI
    #                    (needs /v1/chat/completions, not Responses API)
    # "openai":          public OpenAI endpoint (Responses API supported)
    provider = os.environ.get("MADA_PROVIDER", "livai").strip().lower()
    if provider not in _DEFAULT_BASE_URL_BY_PROVIDER:
        provider = "livai"
    default_base_url = _DEFAULT_BASE_URL_BY_PROVIDER[provider]

    config: dict[str, Any] = {
        "model": {
            "provider": provider,
            "model": workflow.model,
            "api_key": "${API_KEY}",
            "base_url": f"${{API_BASE_URL:-{default_base_url}}}",
        },
    }

    # Orchestration section
    config["orchestration"] = {
        "mode": workflow.orchestration_type,
    }

    # Agents section: augment each agent's instructions with markdown
    # formatting rules so the LLM emits images/tables/code inline
    # automatically without users having to ask for markdown syntax.
    config["agents"] = [
        {
            "agent_name": agent.agent_name,
            "description": agent.description,
            "domain": agent.domain,
            "mcp_servers": agent.mcp_servers,
            "instructions": _augment_instructions(agent.instructions),
        }
        for agent in workflow.agents
    ]

    # MCP servers section: only include servers referenced by agents
    referenced_servers: set[str] = set()
    for agent in workflow.agents:
        referenced_servers.update(agent.mcp_servers)

    if referenced_servers:
        mcp_servers: dict[str, Any] = {}
        for name in sorted(referenced_servers):
            if name in mcp_registry:
                entry = mcp_registry[name]
                server: dict[str, Any] = {
                    "transport": entry.get("transport", "streamable-http"),
                    "description": entry.get("description", ""),
                }
                if entry.get("url"):
                    server["url"] = entry["url"]
                if entry.get("command"):
                    server["command"] = entry["command"]
                mcp_servers[name] = server
            else:
                # Unknown server: create a placeholder
                mcp_servers[name] = {
                    "transport": "streamable-http",
                    "url": f"http://localhost:8080/mcp",
                    "description": f"{name} (auto-generated placeholder)",
                }
        config["mcp_servers"] = mcp_servers

    # Interface section
    title = workflow.task_description[:60]
    if len(workflow.task_description) > 60:
        title = title.rsplit(" ", 1)[0] + "..."
    config["interface"] = {
        "title": f"MADA: {title}",
        "description": workflow.task_description,
        "chat_placeholder": "Ask follow-up questions or request modifications...",
        "port": 7860,
        "share": False,
    }

    return config
