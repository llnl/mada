# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLM-based planner that generates a multi-agent workflow from a task description."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

from .plan_schema import DEFAULT_MODEL, PlannedWorkflow

LOG = logging.getLogger("mada-planner")

SYSTEM_PROMPT = """\
You are the MADA Workflow Planner. Your job is to design a multi-agent workflow \
for the user's task. You decide which agents are needed, what each agent does, \
which MCP tool-servers they connect to, and how they depend on each other.

## MADA Orchestrator Capabilities

The MADA orchestrator runs multiple AI agents that can call MCP tool-servers. \
Two orchestration strategies are available:

**Default to `agent-as-tool`** unless the task genuinely requires magentic's \
deliberation loop. agent-as-tool is significantly faster: it streams tool \
calls incrementally and can dispatch independent tools in parallel via \
asyncio.gather. Magentic serializes participant invocations through a \
manager LLM that itself takes 30-60s of silent deliberation per turn, \
which compounds badly on slower endpoints.

### agent-as-tool (default, prefer this)
A planning agent treats specialist agents as callable tools. Fits:
- Independent specialists that don't need to see each other's mid-turn reasoning
- Sequential pipelines where each stage consumes the previous stage's output
- Fan-out / fan-in patterns (LLM can dispatch parallel tool calls)
- Iterative workflows (LLM decides when to call an agent again with new inputs)
- Almost every scientific / HPC workflow

### magentic (use sparingly)
A manager agent coordinates participants through multi-turn deliberation. \
Only pick this when:
- Agents must critique or refine each other's outputs across multiple rounds
- The task is inherently a debate / consensus process
- You need the manager to autonomously replan mid-execution based on \
  intermediate results

If in doubt, choose agent-as-tool.

## Available MCP Servers

{mcp_server_descriptions}

If the task does not require any external tools, agents can have empty mcp_servers \
lists. Most reasoning/analysis/writing agents don't need MCP servers.

## Agent Design Guidelines

- Give each agent a clear, single responsibility
- Use descriptive names ending in "Agent" (e.g., "EducatorAgent")
- Write detailed instructions (system prompts) that define the agent's expertise, \
  perspective, and expected output format
- Set the domain field to a short keyword (e.g., "education", "critique", "synthesis")
- Only assign MCP servers when the agent genuinely needs external tools
- Use depends_on to model data flow: if AgentB needs AgentA's output, add "AgentA" \
  to AgentB's depends_on list
- Agents with no depends_on run first (or in parallel)
- Create execution_steps that group parallel agents together

## Prefer REUSABLE agents over per-iteration duplicates

The orchestrator can invoke the same agent multiple times with different inputs. \
NEVER create separate "First…Agent" and "Second…Agent" (or "…Iter1", "…Iter2") \
agents when the responsibility is identical and only the input changes. Instead:

- Design ONE agent per RESPONSIBILITY, not per invocation
- The agent's instructions should be parameterized ("stage a sweep with the \
  provided ranges and sample count", not "stage the first sweep with these \
  specific values")
- Show iteration in `execution_steps` by having the same agent appear across \
  multiple steps
- The reasoning field should note when an agent is meant to be reused across \
  iterations (e.g., "SweepRunnerAgent is invoked twice: once for the initial \
  sweep, once for the refined sweep")

BAD:  FirstSweepStagerAgent, FirstSweepRunnerAgent, FirstAnalyzerAgent, \
      SecondSweepStagerAgent, SecondSweepRunnerAgent, SecondAnalyzerAgent (6 agents)
GOOD: SimulationAgent, JobManagementAgent, AnalysisAgent, DesignAgent (4 agents, \
      first three invoked twice; DesignAgent decides the next iteration's \
      parameter ranges, sample count, and when to stop)

Use HIGH-LEVEL, DOMAIN-ORIENTED names. Not implementation-detail names, \
not tool names, not action names. Think of the agent as a professional \
specialist in a field, not a script that does a specific thing.

GOOD (high-level, domain-oriented):
  GeometryAgent, DesignAgent, JobManagementAgent, AnalysisAgent,
  SimulationAgent, VisualizationAgent, DataAgent, OptimizationAgent

BAD (too low-level, implementation-flavored):
  MeshCreatorAgent, RangeCalculatorAgent, FluxRunnerAgent, CubitCallerAgent,
  Params_txt_WriterAgent, SweepStagerAgent, QoIRetrieverAgent,
  JobSubmitterAgent, LaghosExecutorAgent

The BAD names all describe HOW the agent does its work (which MCP tool it \
calls, which file it writes, which subroutine it runs). The GOOD names \
describe WHAT DOMAIN the agent owns. The manager LLM can figure out the \
HOW from the agent's instructions and MCP servers.

Aim for the smallest set of specialists that covers the responsibilities. \
For most workflows this is 3-5 agents. If your plan has more than 6 agents, \
you are almost certainly over-decomposing; consolidate.

## Do NOT create manager / coordinator / orchestrator agents

Mada's orchestration strategies (magentic and agent-as-tool) ALREADY include \
an invisible manager LLM that decides which specialist to invoke next, in \
what order, with what inputs. Creating a visible "StudyManagerAgent", \
"CoordinatorAgent", "OrchestratorAgent", or "PlannerAgent" as a specialist \
is redundant: you'd just have the framework's manager calling your manager \
calling the workers, wasting LLM turns.

Design only the DOMAIN SPECIALISTS that actually do work with tools or produce \
domain artifacts. The manager is implicit.

## depends_on models DATA FLOW, not conceptual hierarchy

Only list AgentA in AgentB's depends_on if AgentB literally consumes AgentA's \
output (mesh file path, staged directory, run results, best-QoI values, etc.). \
Do NOT add "everyone depends on the top agent" or "all downstream agents \
depend on all upstream agents": that produces tangled graphs and doesn't \
match how the orchestrator actually routes data.

Typical shape for a linear pipeline: A -> B -> C -> D (each depends only \
on its immediate predecessor). Not: A -> B, A -> C, A -> D, B -> C, B -> D, \
C -> D.

For iteration/loops, the agent that starts each new iteration MUST list the \
previous-iteration's terminal agent in its depends_on; this creates the \
loop-back edge in the graph and communicates the iterative shape. The UI \
renders these cycles as distinct loop-back arrows; they are expected and \
correct for iterative workflows.

## Concrete example: adaptive parameter sweep with two iterations

For a task like "run a parameter sweep, find the best QoI, then refine ranges \
around the best value and sweep again," the RIGHT plan reuses agents across \
iterations with an explicit loop-back:

```
agents:
  - name: GeometryAgent        depends_on: []                    # runs once
  - name: SimulationAgent      depends_on: [GeometryAgent, DesignAgent]  # 2nd = loop-back
  - name: JobManagementAgent   depends_on: [SimulationAgent]
  - name: AnalysisAgent        depends_on: [JobManagementAgent]
  - name: DesignAgent          depends_on: [AnalysisAgent]       # decides next ranges
```

Note SimulationAgent's depends_on lists both GeometryAgent (used on iteration 1) \
AND DesignAgent (used on iteration 2+). That is not a mistake; it is the \
required way to express the loop-back edge for the graph to render correctly.

WRONG (over-decomposed, low-level names, no loop shown):
```
agents:
  - MeshCreatorAgent, FirstSweepStagerAgent, FirstJobRunnerAgent,
    FirstAnalyzerAgent, RangeCalculatorAgent, SecondSweepStagerAgent,
    SecondJobRunnerAgent, SecondAnalyzerAgent
    # 8 agents, all low-level, linear chain, hides the iteration
```

## Output Format

Return a JSON object matching this schema exactly:
- task_description: echo the user's task
- orchestration_type: "agent-as-tool" or "magentic"
- max_turns: integer (default 20, increase for complex multi-agent tasks)
- max_stalls: integer (default 3)
- model: the LLM model for agents (default "{default_model}")
- agents: list of agent objects with agent_name, description, domain, \
  mcp_servers, instructions, depends_on
- execution_steps: ordered list of steps with step_number, agents, description, parallel
- reasoning: 2-3 sentences explaining why you chose these agents and this strategy
"""


def _make_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a Pydantic JSON schema compatible with OpenAI strict structured output.

    OpenAI requires:
    - 'additionalProperties': false on every object
    - 'required' must list every key in 'properties'
    - No 'default' values (they use required + description instead)
    """
    schema = dict(schema)

    # Process $defs
    if "$defs" in schema:
        schema["$defs"] = {
            k: _make_strict_schema(v) for k, v in schema["$defs"].items()
        }

    # If this is an object type, enforce strict rules
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        if "properties" in schema:
            # All properties must be required
            schema["required"] = list(schema["properties"].keys())
            # Recursively fix nested schemas and remove defaults
            new_props = {}
            for k, v in schema["properties"].items():
                v = _make_strict_schema(v)
                v.pop("default", None)
                new_props[k] = v
            schema["properties"] = new_props

    # Process array items
    if "items" in schema:
        schema["items"] = _make_strict_schema(schema["items"])

    # Process anyOf / oneOf / allOf
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            schema[key] = [_make_strict_schema(s) for s in schema[key]]

    return schema


_IMAGE_EXTS = ("png", "jpg", "jpeg", "gif", "webp", "svg", "bmp", "tif", "tiff")
_IMAGE_EXT_ALT = "|".join(_IMAGE_EXTS)

# Match http(s) URLs ending in a common image extension. GREEDY so URLs like
# `.../foo.jpg/640px-foo.jpg` (Wikipedia thumbnail form) capture the WHOLE
# URL, not just the first `.jpg` fragment.
_IMAGE_URL_RE = re.compile(
    rf"https?://\S+\.(?:{_IMAGE_EXT_ALT})(?:\?\S*)?",
    re.IGNORECASE,
)

# Match absolute local paths ending in an image extension.
# The negative lookbehind excludes `:`, `\w`, and `/` so `https://foo.png`
# doesn't match starting at the second `/` (which would happen if we only
# excluded `\w` and `/`, because the char before is `:` from `https:`).
_LOCAL_PATH_RE = re.compile(
    rf"(?<![\w/:])(/[^\s]+?\.(?:{_IMAGE_EXT_ALT}))(?![\w])",
    re.IGNORECASE,
)

# Also match /api/files/<path> references (agents produce these to embed local
# tool output, but when a follow-up task references one, we want the LLM to
# see the actual image, not the URL string).
_API_FILES_RE = re.compile(
    rf"/api/files/([^\s]+?\.(?:{_IMAGE_EXT_ALT}))",
    re.IGNORECASE,
)


def _local_file_roots() -> list[Path]:
    """Same allowlist logic as api._resolve_file_roots, mirrored to avoid a
    cross-module import (this module has no dependency on api.py otherwise).
    Defaults: launch directory + /tmp/mada-jobs (parent of per-session sandboxes)."""
    env = os.environ.get("MADA_FILE_ROOTS")
    if env:
        raw = env.split(":")
    else:
        jobs_root = os.environ.get("MADA_JOBS_ROOT", "/tmp/mada-jobs")
        raw = [
            str(Path.cwd()),
            jobs_root,
        ]
    return [Path(p).resolve() for p in raw if p.strip()]


def _path_within_allowlist(candidate: Path) -> bool:
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, RuntimeError):
        return False
    for root in _local_file_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _guess_mime(path_or_url: str) -> str:
    mime, _ = mimetypes.guess_type(path_or_url)
    return mime or "application/octet-stream"


async def _url_to_data_uri(url: str, client: httpx.AsyncClient) -> str | None:
    """Fetch an HTTP(S) image URL and return a base64 data URL, or None on error."""
    try:
        response = await client.get(url, timeout=15.0, follow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        LOG.warning("Image URL fetch failed for %s: %s", url, exc)
        return None
    mime = response.headers.get("content-type", "").split(";")[0].strip()
    if not mime or not mime.startswith("image/"):
        mime = _guess_mime(url)
    encoded = base64.b64encode(response.content).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _path_to_data_uri(path: Path) -> str | None:
    """Read a local image file and return a base64 data URL, or None on error."""
    if not _path_within_allowlist(path):
        LOG.warning("Local image path rejected (not in allowlist): %s", path)
        return None
    try:
        data = path.read_bytes()
    except Exception as exc:
        LOG.warning("Local image read failed for %s: %s", path, exc)
        return None
    mime = _guess_mime(str(path))
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


async def _build_user_content(task_description: str) -> Any:
    """
    Turn a task description into an OpenAI Chat Completions "content" value.

    Detects three image sources:
      1. http(s) URLs: backend fetches, base64-encodes to a data URL
      2. Absolute local file paths (/tmp/plot.png etc.): read from disk with
         allowlist enforcement, then encoded as a data URL
      3. /api/files/<path>: treated as a local file path with the same allowlist

    Returns a plain string when no images are found (unchanged behavior).
    Otherwise returns a list of content parts so vision-capable models
    (gpt-4o, gpt-4o-mini, gpt-5) can actually see the images.
    """
    urls = list(dict.fromkeys(_IMAGE_URL_RE.findall(task_description)))
    local_paths = list(dict.fromkeys(_LOCAL_PATH_RE.findall(task_description)))
    api_files = list(dict.fromkeys(_API_FILES_RE.findall(task_description)))

    if not urls and not local_paths and not api_files:
        return task_description

    image_uris: list[str] = []

    if urls:
        async with httpx.AsyncClient() as client:
            for url in urls:
                uri = await _url_to_data_uri(url, client)
                if uri:
                    image_uris.append(uri)

    for p in local_paths:
        uri = _path_to_data_uri(Path(p))
        if uri:
            image_uris.append(uri)

    for rel in api_files:
        # /api/files/tmp/plot.png -> /tmp/plot.png
        uri = _path_to_data_uri(Path("/" + rel.lstrip("/")))
        if uri:
            image_uris.append(uri)

    if not image_uris:
        # All attempts failed; fall back to plain text so the LLM still gets
        # the task, even without images.
        return task_description

    parts: list[dict[str, Any]] = [{"type": "text", "text": task_description}]
    for uri in image_uris:
        parts.append({"type": "image_url", "image_url": {"url": uri}})
    return parts


def _build_mcp_descriptions(mcp_servers: dict[str, Any]) -> str:
    """Format MCP server registry into a readable list for the system prompt."""
    if not mcp_servers:
        return "No MCP servers are currently registered."
    lines = []
    for name, info in mcp_servers.items():
        desc = info.get("description", "No description")
        lines.append(f"- **{name}**: {desc}")
    return "\n".join(lines)


async def generate_plan(
    task_description: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = DEFAULT_MODEL,
    mcp_servers: dict[str, Any] | None = None,
) -> PlannedWorkflow:
    """Generate a multi-agent workflow plan using an LLM.

    Args:
        task_description: What the user wants to accomplish.
        api_key: OpenAI API key.
        base_url: OpenAI-compatible API base URL.
        model: Model to use for planning.
        mcp_servers: Dict of available MCP servers from the registry.

    Returns:
        A validated PlannedWorkflow.
    """
    mcp_servers = mcp_servers or {}
    system = SYSTEM_PROMPT.format(
        mcp_server_descriptions=_build_mcp_descriptions(mcp_servers),
        default_model=DEFAULT_MODEL,
    )

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    user_content = await _build_user_content(task_description)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "PlannedWorkflow",
                "strict": True,
                "schema": _make_strict_schema(PlannedWorkflow.model_json_schema()),
            },
        },
    )

    content = response.choices[0].message.content
    LOG.debug("Planner raw response: %s", content[:500])

    try:
        workflow = PlannedWorkflow.model_validate_json(content)
    except Exception:
        # Retry once with the error included
        LOG.warning("First parse failed, retrying with error feedback")
        retry_response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was not valid JSON matching the schema. "
                        f"Please fix it and return valid JSON only."
                    ),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "PlannedWorkflow",
                    "strict": True,
                    "schema": _make_strict_schema(PlannedWorkflow.model_json_schema()),
                },
            },
        )
        content = retry_response.choices[0].message.content
        workflow = PlannedWorkflow.model_validate_json(content)

    workflow = _add_loopback_deps(workflow)
    return workflow


async def refine_plan(
    current_workflow: PlannedWorkflow,
    feedback: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = DEFAULT_MODEL,
    mcp_servers: dict[str, Any] | None = None,
) -> PlannedWorkflow:
    """
    Refine an existing PlannedWorkflow given natural-language user feedback.

    Sends the SYSTEM_PROMPT, the previous plan as an assistant turn, and the
    user's feedback as a follow-up user turn. LLM returns a new plan that
    incorporates the feedback while preserving the parts the user didn't
    ask to change.
    """
    mcp_servers = mcp_servers or {}
    system = SYSTEM_PROMPT.format(
        mcp_server_descriptions=_build_mcp_descriptions(mcp_servers),
        default_model=DEFAULT_MODEL,
    )

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    previous_plan_json = current_workflow.model_dump_json(indent=2)
    original_task = current_workflow.task_description

    original_task_content = await _build_user_content(original_task)
    feedback_content = await _build_user_content(
        f"Please refine the previous plan based on this feedback:\n\n"
        f"{feedback}\n\n"
        "Return a complete, updated PlannedWorkflow JSON that "
        "incorporates the feedback. Preserve parts of the plan "
        "the feedback didn't ask you to change. Keep the original "
        "task_description unchanged."
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": original_task_content},
            {"role": "assistant", "content": previous_plan_json},
            {"role": "user", "content": feedback_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "PlannedWorkflow",
                "strict": True,
                "schema": _make_strict_schema(PlannedWorkflow.model_json_schema()),
            },
        },
    )

    content = response.choices[0].message.content
    LOG.debug("Refinement raw response: %s", content[:500])
    workflow = PlannedWorkflow.model_validate_json(content)

    # Ensure the original task description survives (LLMs sometimes rewrite it)
    if not workflow.task_description.strip():
        workflow.task_description = original_task

    workflow = _add_loopback_deps(workflow)
    return workflow


def _add_loopback_deps(workflow: PlannedWorkflow) -> PlannedWorkflow:
    """
    Detect agents reused across iterations in execution_steps and add
    depends_on entries for the loop-back edges the LLM tends to omit.

    Rule: if an agent appears in execution_steps more than once, add the
    last agent from the step immediately BEFORE its second appearance
    to its depends_on list (unless already present).
    """
    agents_by_name = {a.agent_name: a for a in workflow.agents}
    seen: set[str] = set()

    for i, step in enumerate(workflow.execution_steps):
        for agent_name in step.agents:
            if agent_name in seen and i > 0:
                # This is a re-invocation. Find the last agent from the
                # previous step; that's likely what feeds back into this one.
                prev_step = workflow.execution_steps[i - 1]
                if not prev_step.agents:
                    continue
                prev_terminal = prev_step.agents[-1]
                if prev_terminal == agent_name:
                    continue
                agent = agents_by_name.get(agent_name)
                if agent and prev_terminal not in agent.depends_on:
                    LOG.info(
                        "Auto-adding loop-back dep: %s depends_on += [%s]",
                        agent_name,
                        prev_terminal,
                    )
                    agent.depends_on = [*agent.depends_on, prev_terminal]
            seen.add(agent_name)

    return workflow
