"""
Approval abstractions for manifest-based skill scripts.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Tuple

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillScriptApprovalRequest:
    """Approval request for one manifest-discovered skill script invocation."""

    skill_name: str
    script_name: str
    script_path: Path
    runner: str
    args: Tuple[str, ...] = ()
    timeout_seconds: int | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SkillScriptApprovalDecision:
    """Approval decision for one skill script invocation."""

    approved: bool
    reason: str = ""


class SkillScriptApprover(Protocol):
    """Interface for deciding whether one skill script may run."""

    def approve_skill_script(
        self,
        request: SkillScriptApprovalRequest,
    ) -> SkillScriptApprovalDecision:
        """Return an approval decision for one skill script invocation."""


class DenyAllSkillScriptApprover:
    """Default approver that denies all skill script execution."""

    def approve_skill_script(
        self,
        request: SkillScriptApprovalRequest,
    ) -> SkillScriptApprovalDecision:
        """
        Deny every skill script invocation.

        Args:
            request: Approval request describing the script.

        Returns:
            A denial decision explaining that no approver was configured.
        """
        LOG.debug(
            f"Denying script '{request.script_name}' for skill "
            f"'{request.skill_name}': no approver configured"
        )
        return SkillScriptApprovalDecision(
            approved=False,
            reason=(
                f"Skill script '{request.script_name}' for skill "
                f"'{request.skill_name}' was denied by the default approval policy."
            ),
        )


class PolicyBasedSkillScriptApprover:
    """Config-driven approver for skill-specific script approval policies."""

    def __init__(
        self,
        default_mode: str = "deny",
        skill_modes: dict[str, str] | None = None,
    ):
        """
        Initialize a policy approver.

        Args:
            default_mode: Mode applied when no override matches. Either
                "approve" or "deny"; unrecognized values fall back to "deny".
            skill_modes: Per-skill and per-script overrides keyed by
                "skill_name:script_name", "skill_name", or "*".
        """
        self.default_mode = self._normalize_mode(default_mode) or "deny"
        self.skill_modes = skill_modes or {}
        LOG.debug(
            f"Policy approver initialized with default mode '{self.default_mode}' "
            f"and {len(self.skill_modes)} override(s)"
        )

    def approve_skill_script(
        self,
        request: SkillScriptApprovalRequest,
    ) -> SkillScriptApprovalDecision:
        """
        Decide whether a script may run based on configured policy.

        Overrides are matched most specific first: an exact
        "skill_name:script_name" key, then "skill_name", then "*". When none
        match, the default mode applies.

        Args:
            request: Approval request describing the script.

        Returns:
            The policy's approval decision.
        """
        candidates = (
            f"{request.skill_name}:{request.script_name}",
            request.skill_name,
            "*",
        )

        mode = None
        for key in candidates:
            normalized = self._normalize_mode(self.skill_modes.get(key))
            if normalized is not None:
                mode = normalized
                LOG.debug(f"Matched approval override '{key}' -> '{mode}'")
                break

        if mode is None:
            mode = self.default_mode
            LOG.debug(f"No approval override matched; using default '{mode}'")

        if mode == "approve":
            return SkillScriptApprovalDecision(
                approved=True,
                reason=(
                    f"Skill script '{request.script_name}' for skill "
                    f"'{request.skill_name}' was approved by policy."
                ),
            )

        return SkillScriptApprovalDecision(
            approved=False,
            reason=(
                f"Skill script '{request.script_name}' for skill "
                f"'{request.skill_name}' was denied by policy."
            ),
        )

    @staticmethod
    def _normalize_mode(mode: str | None) -> str | None:
        """
        Normalize a configured approval mode.

        Args:
            mode: Raw mode value from configuration.

        Returns:
            "approve" or "deny", or None when the value is absent or
            unrecognized.
        """
        if mode is None:
            return None
        normalized = str(mode).strip().lower()
        if normalized in {"approve", "deny"}:
            return normalized
        return None


class PromptingSkillScriptApprover:
    """Interactive approver that asks a user to authorize each skill script."""

    def __init__(
        self,
        input_func: Callable[[str], str] = input,
        output_func: Callable[[str], None] = print,
    ):
        """
        Initialize an interactive approver.

        Args:
            input_func: Callable used to read the user's response.
            output_func: Callable used to display the approval prompt.
        """
        self.input_func = input_func
        self.output_func = output_func

    def approve_skill_script(
        self,
        request: SkillScriptApprovalRequest,
    ) -> SkillScriptApprovalDecision:
        """
        Ask the user whether a skill script may run.

        Args:
            request: Approval request describing the script.

        Returns:
            An approval decision reflecting the user's response. Anything other
            than an explicit yes is treated as a denial.
        """
        LOG.debug(
            f"Prompting for approval of script '{request.script_name}' "
            f"for skill '{request.skill_name}'"
        )
        args_display = " ".join(request.args) if request.args else "(none)"
        self.output_func("")
        self.output_func("Skill script approval required:")
        self.output_func(f"  Skill: {request.skill_name}")
        self.output_func(f"  Script: {request.script_name}")
        self.output_func(f"  Args: {args_display}")
        self.output_func(f"  Path: {Path(request.script_path)}")

        response = self.input_func("Approve script execution? [y/N]: ").strip().lower()
        if response in {"y", "yes"}:
            return SkillScriptApprovalDecision(
                approved=True,
                reason="Skill script was approved by the user.",
            )
        return SkillScriptApprovalDecision(
            approved=False,
            reason="Skill script was denied by the user.",
        )


def build_skill_script_approver(
    config: Any,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> SkillScriptApprover:
    """
    Build the skill script approver described by runtime configuration.

    A default mode of "prompt" selects an interactive approver that asks the
    user to authorize each script. Any other mode selects a policy approver
    driven by `skill_script_approval_modes`.

    Args:
        config: Skill runtime configuration.
        input_func: Callable used to read a user's approval response.
        output_func: Callable used to display approval prompts.

    Returns:
        An approver implementing the `SkillScriptApprover` protocol.
    """
    default_mode = (
        str(getattr(config, "default_skill_script_approval_mode", "prompt"))
        .strip()
        .lower()
    )
    skill_modes = dict(getattr(config, "skill_script_approval_modes", {}) or {})

    if default_mode == "prompt":
        LOG.debug("Building interactive skill script approver")
        return PromptingSkillScriptApprover(
            input_func=input_func,
            output_func=output_func,
        )

    LOG.debug(f"Building policy skill script approver with mode '{default_mode}'")
    return PolicyBasedSkillScriptApprover(
        default_mode=default_mode,
        skill_modes=skill_modes,
    )
