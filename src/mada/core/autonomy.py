"""
Shared autonomy helpers.

These helpers are intentionally UI-agnostic so both the Gradio wrapper and the
CLI can implement autonomous follow-ups, including timed waits, without
duplicating parsing logic.
"""

from __future__ import annotations

import re

ALLOWED_AUTONOMY_DECISIONS = {"CONTINUE", "ASK", "WAIT", "STOP"}
MAX_AUTONOMY_WAIT_SECONDS = 86400
MAX_AUTONOMY_FOLLOWUPS = 25

_AUTONOMY_INTERVAL_RE = re.compile(
    r"every\s+(\d+(?:\.\d+)?)\s*seconds",
    re.IGNORECASE,
)


def max_autonomy_followups(level: int) -> int:
    """
    Return the safety cap on autonomous follow-up turns.

    Autonomy level expresses how much freedom the controller has to decide
    whether additional follow-ups are needed. A fixed cap still prevents
    accidental infinite loops.
    """
    if level <= 0:
        return 0
    return MAX_AUTONOMY_FOLLOWUPS


def default_wait_seconds_from_user_message(message: str) -> int:
    """Infer a default wait interval from the user's prompt."""
    match = _AUTONOMY_INTERVAL_RE.search(message or "")
    if not match:
        return 20
    try:
        return int(float(match.group(1)))
    except (TypeError, ValueError):
        return 20


def parse_autonomy_control(decision_text: str) -> tuple[str, str, str, int, bool]:
    """Parse a strict autonomy controller response."""
    decision: str | None = None
    next_query = ""
    question = ""
    wait_seconds = 0

    for raw_line in (decision_text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue

        upper = line.upper()
        if upper.startswith("AUTONOMY_DECISION="):
            decision = line.split("=", 1)[1].strip().upper() or "STOP"
        elif upper.startswith("AUTONOMY_QUERY="):
            next_query = line.split("=", 1)[1].strip()
        elif upper.startswith("AUTONOMY_QUESTION="):
            question = line.split("=", 1)[1].strip()
        elif upper.startswith("AUTONOMY_WAIT_SECONDS="):
            raw_value = line.split("=", 1)[1].strip()
            try:
                wait_seconds = int(float(raw_value))
            except (TypeError, ValueError):
                wait_seconds = 0

    parse_ok = bool(decision) and decision in ALLOWED_AUTONOMY_DECISIONS
    if not parse_ok:
        return "STOP", "", "", 0, False

    if decision == "CONTINUE":
        return "CONTINUE", next_query, "", 0, True
    if decision == "WAIT":
        return "WAIT", next_query, "", wait_seconds, True
    if decision == "ASK":
        return "ASK", "", question, 0, True
    return "STOP", "", "", 0, True


def build_autonomy_enabled_prompt(
    user_prompt: str,
    *,
    level: int,
    followups_used: int,
    followups_max: int,
) -> str:
    """
    Prefix a prompt with an internal note that autonomy follow-ups are enabled.
    """
    normalized_level = int(level or 0)
    if normalized_level <= 0:
        return user_prompt

    header = (
        "[INTERNAL NOTE]\n"
        f"Autonomous follow-ups are ENABLED for this chat (level={normalized_level}).\n"
        "After you respond, the system may run additional follow-up turn(s) "
        "(bounded by a safety cap) and can wait in real time.\n"
        "The system can also perform timer-based repetition / periodic updates "
        "(e.g., 'every 20 seconds') by waiting and then issuing follow-up turns.\n"
        "You are NOT the autonomy controller. Do NOT output autonomy-control "
        "key/value lines like AUTONOMY_DECISION=..., and do NOT claim you cannot "
        "do follow-ups or waiting.\n"
        "Do NOT say things like:\n"
        '- "I can\'t continue sending messages on a timer by myself."\n'
        '- "I can\'t keep checking every N seconds."\n'
        '- "I can\'t send follow-up messages after this reply."\n'
        "Write a normal user-facing assistant reply. Only mention autonomy if "
        "the user explicitly asks.\n\n"
    )
    return f"{header}{user_prompt}"


def tail_text(text: str, max_chars: int) -> str:
    """Return the tail of `text` up to `max_chars` characters."""
    if not text:
        return ""
    try:
        limit = int(max_chars)
    except (TypeError, ValueError):
        limit = 0
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def build_autonomy_followup_prompt(
    next_query: str,
    *,
    original_request: str,
    last_reply: str,
    assistant_buffer: str,
    level: int,
    followups_used: int,
    followups_max: int,
    max_chars_original_request: int = 1200,
    max_chars_last_reply: int = 800,
    max_chars_transcript_tail: int = 2000,
) -> str:
    """
    Build a follow-up prompt that includes compact progress context.
    """
    context_block = (
        "[INTERNAL CONTEXT - DO NOT REPEAT]\n"
        "Continue from the current state. Do not restart the task and do not "
        "reprint already-completed output unless the user asks.\n\n"
        "Original user request:\n"
        f"{tail_text((original_request or '').strip(), max_chars_original_request)}\n\n"
        "Most recent assistant reply:\n"
        f"{tail_text((last_reply or '').strip(), max_chars_last_reply)}\n\n"
        "User-visible transcript tail:\n"
        f"{tail_text((assistant_buffer or '').strip(), max_chars_transcript_tail)}\n"
        "[END INTERNAL CONTEXT]\n\n"
        "Follow-up task:\n"
        f"{(next_query or '').strip()}"
    )
    return build_autonomy_enabled_prompt(
        context_block,
        level=level,
        followups_used=followups_used,
        followups_max=followups_max,
    )
