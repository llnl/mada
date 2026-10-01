# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""FastAPI backend for the MADA Workflow Planner UI."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import re

import click
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .code_generator import generate_code
from .config_generator import generate_config
from .plan_schema import GeneratedOutput, PlannedWorkflow, PlanRequest
from .planner_agent import generate_plan, refine_plan

LOG = logging.getLogger("mada-planner-api")

CONFIGS_DIR = Path(__file__).resolve().parents[4] / "configs"
REGISTRY_PATH = CONFIGS_DIR / "mcp_server_registry.json"

app = FastAPI(title="MADA Workflow Planner API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_mcp_registry() -> dict[str, Any]:
    """Load the MCP server registry from disk."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}


# Allowlist of directories from which /api/files can serve.
#
# Defaults cover:
#   - the launch directory (where mada-planner-api was started): so files
#     already present in the repo/workspace are servable without extra config.
#     Works out-of-the-box for local dev (repo as CWD) and containers (WORKDIR).
#   - /tmp/mada-jobs: a mada-owned scratch directory for tool outputs. Agents
#     should write generated files (plots, meshes, etc.) here.
#
# Override with MADA_FILE_ROOTS (colon-separated) for tighter or different
# scoping, e.g. `export MADA_FILE_ROOTS=/data:/tmp` inside a container.
# Set MADA_JOBS_ROOT to relocate the job scratch dir (default /tmp/mada-jobs).
_LAUNCH_DIR = str(Path.cwd())
MADA_JOBS_ROOT = Path(os.environ.get("MADA_JOBS_ROOT", "/tmp/mada-jobs")).resolve()
MADA_JOBS_ROOT.mkdir(parents=True, exist_ok=True)

_DEFAULT_FILE_ROOTS = [
    _LAUNCH_DIR,
    str(MADA_JOBS_ROOT),
]


def _resolve_file_roots() -> list[Path]:
    """Return the list of directories from which /api/files may serve."""
    env = os.environ.get("MADA_FILE_ROOTS")
    raw = env.split(":") if env else _DEFAULT_FILE_ROOTS
    return [Path(p).resolve() for p in raw if p.strip()]


_FILE_ROOTS = _resolve_file_roots()


def _is_within_allowlist(candidate: Path) -> bool:
    """Return True if `candidate` (resolved) is inside any allowlisted root."""
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, RuntimeError):
        return False
    for root in _FILE_ROOTS:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


# ─── Image path auto-inlining ────────────────────────────────────────
#
# When agents (or the coordinator LLM) mention a local image path in their
# streamed response, we automatically append a markdown image tag so the
# image displays in the UI even if the LLM forgot to emit `![]()` syntax.
# This makes simulation-generated images (Laghos plots, Cubit mesh
# screenshots) reliably visible without depending on LLM discipline.

_IMAGE_EXTS = "png|jpg|jpeg|gif|webp|svg|bmp|tiff?"

# Absolute local paths ending in an image extension.
# Exclude `:` in lookbehind so `https://foo.png` doesn't match at the second `/`.
_LOCAL_IMAGE_PATH_RE = re.compile(
    rf"(?<![\w/:])(/[^\s\)]+?\.(?:{_IMAGE_EXTS}))\b",
    re.IGNORECASE,
)

# http(s) URLs ending in an image extension. GREEDY so URLs like
# `.../foo.jpg/640px-foo.jpg` (Wikipedia thumbnail form) capture the WHOLE
# URL, not just the first `.jpg` fragment. The `[^\s\)]+` grabs everything
# up to whitespace/paren, then the engine backtracks to find the last valid
# extension before that terminator.
_IMAGE_URL_RE = re.compile(
    rf"https?://[^\s\)]+\.(?:{_IMAGE_EXTS})(?:\?[^\s\)]*)?",
    re.IGNORECASE,
)


def _autoinline_local_images(text: str, source_context: str = "") -> str:
    """
    Detect image references and append markdown image tags so the UI
    renders them, even if the LLM forgot to emit `![]()` syntax, produced
    a broken image tag, or refused to include the URL at all.

    Scans in two places:
      - `text`: the LLM's response (agent narration + final synthesis)
      - `source_context`: the user's original task/followup message

    This dual scan matters because runtime agents sometimes refuse to
    reproduce URLs (saying "I can't retrieve images directly"), leaving
    the response text with no URL to grab. Falling back to the user's
    message ensures the image still shows.

    Handles two kinds of references:
      1. Absolute local paths (`/tmp/plot.png`): must be inside the
         MADA_FILE_ROOTS allowlist AND exist on disk. Emitted as
         `/api/files/...` markdown.
      2. Public http(s) image URLs: emitted verbatim as markdown.

    Deduplicates: if the same image is already embedded in markdown
    anywhere in the response text, don't add it again.
    """
    if not text and not source_context:
        return text

    additions: list[str] = []
    seen: set[str] = set()

    # Combined search space so we catch URLs the user mentioned even if
    # the agent's response omitted them.
    combined = f"{source_context}\n{text}"

    # Local paths: scanned from BOTH response text AND user context.
    # The allowlist check + file existence check are the real security
    # boundary. If a user names a path that's inside MADA_FILE_ROOTS and
    # actually exists, that's a legitimate reference we should honor;
    # otherwise agents that refuse to reproduce the path (as often happens
    # for image files) would leave the user unable to see their own file.
    for match in _LOCAL_IMAGE_PATH_RE.finditer(combined):
        raw_path = match.group(1)
        if raw_path in seen:
            continue
        seen.add(raw_path)

        api_url = f"/api/files{raw_path}"
        if f"]({api_url})" in text or f"](file://{raw_path})" in text:
            continue

        candidate = Path(raw_path)
        if not _is_within_allowlist(candidate):
            continue

        additions.append(f"\n\n![{candidate.name}]({api_url})")

    # Public URLs: scanned from BOTH response and user context
    for match in _IMAGE_URL_RE.finditer(combined):
        url = match.group(0)
        if url in seen:
            continue
        seen.add(url)

        # Skip if already embedded (in either response or user's markdown)
        if f"]({url})" in text:
            continue

        alt = url.rsplit("/", 1)[-1].split("?", 1)[0]
        additions.append(f"\n\n![{alt}]({url})")

    if not additions:
        return text
    return text + "".join(additions)


# ─── Endpoints ───────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/mcp-servers")
async def list_mcp_servers():
    """Return the MCP server registry."""
    return _load_mcp_registry()


@app.get("/api/files/{file_path:path}")
async def serve_file(file_path: str):
    """
    Serve a file from an allowlisted directory. Agents can embed images
    produced by tools (Cubit mesh screenshots, Laghos plots, etc.) using
    markdown like:

        ![mesh](/api/files/tmp/laghos_results/run0/plot.png)

    Path traversal is guarded by resolving the path and checking it falls
    under one of MADA_FILE_ROOTS (defaults to the backend launch directory
    and $MADA_JOBS_ROOT, which is /tmp/mada-jobs unless overridden).
    Set MADA_FILE_ROOTS='/tmp:/other' to override.
    """
    candidate = Path("/" + file_path.lstrip("/"))
    if not _is_within_allowlist(candidate):
        raise HTTPException(status_code=404, detail="File not found or not accessible")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(candidate)


@app.post("/api/plan", response_model=PlannedWorkflow)
async def create_plan(request: PlanRequest):
    """Generate a workflow plan from a task description."""
    api_key = os.environ.get("API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="API_KEY environment variable is not set")
    base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    mcp_registry = _load_mcp_registry()

    try:
        workflow = await generate_plan(
            task_description=request.task_description,
            api_key=api_key,
            base_url=base_url,
            model=request.model,
            mcp_servers=mcp_registry,
        )
        return workflow
    except Exception as exc:
        LOG.exception("Plan generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


class RefineRequest(BaseModel):
    workflow: PlannedWorkflow
    feedback: str
    model: str = "gpt-4o-mini"


@app.post("/api/refine-plan", response_model=PlannedWorkflow)
async def refine_plan_endpoint(request: RefineRequest):
    """Refine an existing plan given natural-language user feedback."""
    api_key = os.environ.get("API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=500, detail="API_KEY environment variable is not set"
        )
    base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    mcp_registry = _load_mcp_registry()

    try:
        refined = await refine_plan(
            current_workflow=request.workflow,
            feedback=request.feedback,
            api_key=api_key,
            base_url=base_url,
            model=request.model,
            mcp_servers=mcp_registry,
        )
        return refined
    except Exception as exc:
        LOG.exception("Plan refinement failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/generate", response_model=GeneratedOutput)
async def generate_output(workflow: PlannedWorkflow):
    """Generate JSON config and Python code from a plan."""
    mcp_registry = _load_mcp_registry()
    json_config = generate_config(workflow, mcp_registry)

    # Build a config filename from the first few words of the task
    slug = (
        workflow.task_description[:40]
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .strip("_")
    )
    config_filename = f"configs/{slug}.json"
    python_code = generate_code(workflow, config_filename)

    return GeneratedOutput(json_config=json_config, python_code=python_code)


@app.websocket("/ws/execute")
async def execute_workflow(websocket: WebSocket):
    """Execute a workflow and stream results over WebSocket.

    Protocol:
    1. Client sends a PlannedWorkflow JSON message.
    2. Server streams text chunks as they arrive.
    3. Server sends {"done": true} when complete.
    """
    await websocket.accept()
    try:
        # Wormhole community subtoken: forwarded to streamable-HTTP MCP servers
        # as X-Token so tools act on behalf of the actual user. Header first
        # (proxy-injected); query param fallback for browsers that can't set headers.
        bearer_token = websocket.headers.get("x-subtoken") or websocket.query_params.get("token")

        raw = await websocket.receive_text()
        workflow = PlannedWorkflow.model_validate_json(raw)

        mcp_registry = _load_mcp_registry()
        json_config = generate_config(workflow, mcp_registry)

        # Import here to avoid loading heavy deps at module level
        from mada.core.config import AppConfig
        from mada.core.database.session_manager import ChatSessionManager
        from mada.core.orchestration.stream_events import (
            error_message,
            response_replacement,
            tool_call_name,
        )
        from mada.core.orchestrator import MADAOrchestrator

        app_config = AppConfig.from_dict(json_config)
        session_manager = ChatSessionManager(app_config.database)
        session_manager.create_new_session()

        async with MADAOrchestrator(
            model_config=app_config.model,
            database_config=app_config.database,
            orchestration_config=app_config.orchestration,
            session_manager=session_manager,
            bearer_token=bearer_token,
        ) as orchestrator:
            status, tools = await orchestrator.initialize_orchestrator(
                agent_configs=app_config.agents,
                mcp_servers=app_config.mcp_servers,
            )
            await websocket.send_json({"type": "status", "message": status, "tools": len(tools)})

            # Same handling shape as openai_api.stream_response: magentic buffers
            # to absorb response_replacement events; other modes stream immediately.
            is_magentic = app_config.orchestration.mode == "magentic"
            buffered_text = ""

            # Track the currently-running agent so we can emit agent_result when
            # the next agent starts (mada doesn't emit a completion signal, so
            # this heuristic marks the previous agent complete when a new one
            # begins or when text streaming resumes).
            current_agent: str | None = None
            current_agent_output = ""

            # Accumulate everything the LLM has narrated this turn so we can
            # scan it for image paths at end-of-turn and auto-append markdown
            # image tags (see _autoinline_local_images).
            turn_narration = ""

            async def flush_autoinline(user_context: str = "") -> None:
                nonlocal turn_narration
                augmented = _autoinline_local_images(turn_narration, user_context)
                if augmented != (turn_narration or ""):
                    # Only send the *additions*: the original text already
                    # streamed. This keeps the UI's incremental behavior intact.
                    base_len = len(turn_narration or "")
                    additions = augmented[base_len:]
                    if additions:
                        await websocket.send_json({"type": "chunk", "content": additions})
                turn_narration = ""

            async def emit_agent_start(name: str) -> None:
                nonlocal current_agent, current_agent_output
                if current_agent is not None:
                    await websocket.send_json({
                        "type": "agent_result",
                        "agent": current_agent,
                        "result": current_agent_output.strip() or "(completed)",
                    })
                current_agent = name
                current_agent_output = ""
                await websocket.send_json({"type": "agent_start", "agent": name})

            async def close_current_agent(final_output: str = "") -> None:
                nonlocal current_agent, current_agent_output
                if current_agent is not None:
                    result = (final_output or current_agent_output).strip() or "(completed)"
                    await websocket.send_json({
                        "type": "agent_result",
                        "agent": current_agent,
                        "result": result,
                    })
                    current_agent = None
                    current_agent_output = ""

            async for chunk in orchestrator.process_message(workflow.task_description):
                # Invisible InternalToolCallSignal (magentic) -> structured event
                signal_name = tool_call_name(chunk)
                if signal_name:
                    await emit_agent_start(signal_name)
                    continue

                # Magentic can retract streamed text with a final replacement
                replacement = response_replacement(chunk)
                if replacement is not None:
                    buffered_text = replacement
                    continue

                # Terminal internal error signal
                err = error_message(chunk)
                if err is not None:
                    await close_current_agent()
                    if buffered_text:
                        await websocket.send_json({"type": "chunk", "content": buffered_text})
                        buffered_text = ""
                    await websocket.send_json({"type": "error", "message": err})
                    return

                text = str(chunk)
                if not text:
                    continue

                # agent-as-tool surfaces tool calls as visible "[Calling: name]"
                # chunks; translate to the same agent_start event for the UI.
                stripped = text.strip()
                if stripped.startswith("[Calling:") and stripped.endswith("]"):
                    await emit_agent_start(stripped[len("[Calling:"):-1].strip())
                    continue

                # Any real text chunk is the coordinator LLM's own narration
                # or final synthesis, NOT the specialist agent's output
                # (that goes back to the coordinator invisibly as the tool
                # return). Send it to the main assistant bubble, not to any
                # agent card, otherwise final synthesis gets misattributed
                # to whichever agent was called last.
                turn_narration += text
                if is_magentic:
                    buffered_text += text
                else:
                    await websocket.send_json({"type": "chunk", "content": text})

            # Flush any remaining state at the end of the workflow.
            await close_current_agent()
            if buffered_text:
                turn_narration += buffered_text
                await websocket.send_json({"type": "chunk", "content": buffered_text})

            # Scan the full narration for local image paths and auto-append
            # markdown image tags for any allowlisted files the LLM mentioned
            # but forgot to embed. Also scans the user's original task
            # description so URLs the user provided still get displayed even
            # when the agent refused to include them.
            await flush_autoinline(workflow.task_description)

            await websocket.send_json({"type": "done"})

            # Post-execution chat loop: keep the socket open and accept
            # follow-up messages routed to the same orchestrator instance.
            # Same session, same MCP connections, same conversation history.
            while True:
                raw_followup = await websocket.receive_text()
                try:
                    payload = json.loads(raw_followup)
                    followup_message = payload.get("message", "").strip()
                except json.JSONDecodeError:
                    followup_message = raw_followup.strip()

                if not followup_message:
                    continue

                await websocket.send_json({
                    "type": "followup_start",
                    "message": followup_message,
                })

                # Reset per-turn tracking state (nonlocals defined above)
                current_agent = None
                current_agent_output = ""
                buffered_text = ""
                followup_narration = ""

                async for chunk in orchestrator.process_message(followup_message):
                    signal_name = tool_call_name(chunk)
                    if signal_name:
                        await emit_agent_start(signal_name)
                        continue

                    replacement = response_replacement(chunk)
                    if replacement is not None:
                        buffered_text = replacement
                        continue

                    err = error_message(chunk)
                    if err is not None:
                        await close_current_agent()
                        if buffered_text:
                            await websocket.send_json({
                                "type": "followup_chunk",
                                "content": buffered_text,
                            })
                            buffered_text = ""
                        await websocket.send_json({"type": "error", "message": err})
                        return

                    text = str(chunk)
                    if not text:
                        continue

                    stripped = text.strip()
                    if stripped.startswith("[Calling:") and stripped.endswith("]"):
                        await emit_agent_start(stripped[len("[Calling:"):-1].strip())
                        continue

                    # Same rule as the initial workflow: coordinator narration
                    # goes to the main assistant bubble, not to agent cards.
                    followup_narration += text
                    if is_magentic:
                        buffered_text += text
                    else:
                        await websocket.send_json({
                            "type": "followup_chunk",
                            "content": text,
                        })

                await close_current_agent()
                if buffered_text:
                    followup_narration += buffered_text
                    await websocket.send_json({
                        "type": "followup_chunk",
                        "content": buffered_text,
                    })

                # Auto-inline any image paths/URLs mentioned in the response
                # OR in the user's follow-up message. The latter matters when
                # the agent refuses to reproduce a URL: we still show what
                # the user asked about.
                augmented = _autoinline_local_images(
                    followup_narration, followup_message
                )
                if augmented != (followup_narration or ""):
                    base_len = len(followup_narration or "")
                    additions = augmented[base_len:]
                    if additions:
                        await websocket.send_json({
                            "type": "followup_chunk",
                            "content": additions,
                        })

                await websocket.send_json({"type": "followup_done"})

    except WebSocketDisconnect:
        LOG.info("WebSocket client disconnected")
    except Exception as exc:
        LOG.exception("Execution error")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


# ─── CLI Entry Point ─────────────────────────────────────────────────


@click.command(name="mada-planner-api")
@click.option("-p", "--port", type=int, default=8000, help="Port to listen on")
@click.option("--host", type=str, default="0.0.0.0", help="Host to bind to")
def main(port: int, host: str):
    """Start the MADA Workflow Planner API server."""
    import uvicorn

    uvicorn.run(
        "mada.interfaces.planner.api:app",
        host=host,
        port=port,
        reload=True,
    )


if __name__ == "__main__":
    main()
