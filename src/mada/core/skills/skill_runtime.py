"""
Runtime helpers for manifest-based skills.
"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from mada.core.config import SkillRuntimeConfig
from .skill_manifest import (
    SkillManifestError,
    parse_skill_manifest,
)
from .utils import is_tool_allowed
from .skill_approval import (
    DenyAllSkillScriptApprover,
    SkillScriptApprovalRequest,
    SkillScriptApprover,
)
from .skill_registry import (
    DiscoveredSkillResource,
    DiscoveredSkillScript,
    SkillRegistry,
    SkillRegistryError,
)

LOG = logging.getLogger(__name__)


class SkillRuntimeError(Exception):
    """Raised when runtime access to manifest-based skill content fails."""


class SkillRuntime:
    """Validated runtime access to manifest-discovered skill content."""

    def __init__(
        self,
        skill_registry: SkillRegistry,
        config: SkillRuntimeConfig | None = None,
        script_approver: SkillScriptApprover | None = None,
    ):
        """
        Initialize runtime access to discovered skills.

        Args:
            skill_registry: Registry of discovered skills.
            config: Runtime limits and approval policy. Defaults are used when
                omitted.
            script_approver: Approver consulted before running a skill script.
                Defaults to denying every script.
        """
        self.skill_registry = skill_registry
        self.config = config or SkillRuntimeConfig()
        self.script_approver = script_approver or DenyAllSkillScriptApprover()

    def load_skill(self, skill_name: str) -> str:
        """Load the full SKILL.md body for a skill."""
        skill = self.skill_registry.get_skill(skill_name)
        self._require_allowed_tool(skill_name, skill.allowed_tools, "load_skill")
        LOG.info(f"Loading skill '{skill_name}' from '{skill.manifest_path}'")
        try:
            return parse_skill_manifest(skill.root_path).content
        except SkillManifestError as exc:
            raise SkillRuntimeError(str(exc)) from exc

    def read_skill_resource(self, skill_name: str, resource_path: str) -> str:
        """Read a discovered text resource for a skill with runtime validation."""
        normalized_path = self._normalize_resource_path(resource_path)

        try:
            skill = self.skill_registry.get_skill(skill_name)
            resource = self.skill_registry.get_resource(skill_name, normalized_path)
        except SkillRegistryError as exc:
            raise SkillRuntimeError(str(exc)) from exc

        self._require_allowed_tool(
            skill_name, skill.allowed_tools, "read_skill_resource"
        )
        self._validate_resource(skill.root_path, resource)
        LOG.debug(
            f"Reading resource '{normalized_path}' ({resource.size_bytes} bytes) "
            f"for skill '{skill_name}'"
        )
        try:
            return resource.file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SkillRuntimeError(
                f"Resource '{normalized_path}' for skill '{skill_name}' is not UTF-8 text."
            ) from exc

    def run_skill_script(
        self,
        skill_name: str,
        script_name: str,
        args: list[str] | None = None,
    ) -> dict[str, Any]:
        """Validate, approve, and execute one discovered skill script safely."""
        normalized_name = self._normalize_script_name(script_name)
        normalized_args = self._normalize_script_args(args)

        try:
            skill = self.skill_registry.get_skill(skill_name)
            script = self.skill_registry.get_script(skill_name, normalized_name)
        except SkillRegistryError as exc:
            raise SkillRuntimeError(str(exc)) from exc

        if not is_tool_allowed(skill.allowed_tools, "run_skill_script"):
            reason = (
                f"Skill '{skill_name}' does not allow use of tool 'run_skill_script' "
                "under its manifest policy."
            )
            return {
                "status": "denied",
                "skill_name": skill_name,
                "script_name": normalized_name,
                "approved": False,
                "executed": False,
                "outcome": reason,
            }

        self._validate_script(skill.root_path, script)
        approval_request = SkillScriptApprovalRequest(
            skill_name=skill_name,
            script_name=normalized_name,
            script_path=script.file_path,
            runner=script.runner,
            args=tuple(normalized_args),
            timeout_seconds=self.config.default_script_timeout_seconds,
        )
        decision = self.script_approver.approve_skill_script(approval_request)
        LOG.info(
            f"Script '{normalized_name}' for skill '{skill_name}' was "
            f"{'approved' if decision.approved else 'denied'}: {decision.reason}"
        )
        if not decision.approved:
            return {
                "status": "denied",
                "skill_name": skill_name,
                "script_name": normalized_name,
                "approved": False,
                "executed": False,
                "outcome": decision.reason,
            }

        return self._execute_script(
            skill_name=skill_name,
            skill_root=skill.root_path,
            script=script,
            normalized_args=normalized_args,
        )

    def _normalize_resource_path(self, resource_path: str) -> str:
        normalized = PurePosixPath(str(resource_path).strip())
        if not str(normalized) or str(normalized) == ".":
            raise SkillRuntimeError("Resource path must be a non-empty relative path.")
        if normalized.is_absolute():
            raise SkillRuntimeError("Resource path must be relative to the skill root.")
        if any(part in {"..", ""} for part in normalized.parts):
            raise SkillRuntimeError(
                "Resource path must not contain parent-directory traversal."
            )
        return normalized.as_posix()

    def _validate_resource(
        self,
        skill_root: Path,
        resource: DiscoveredSkillResource,
    ) -> None:
        if not resource.text_readable:
            raise SkillRuntimeError(
                f"Resource '{resource.path}' is not a supported text resource."
            )

        if resource.size_bytes > self.config.max_resource_bytes:
            raise SkillRuntimeError(
                f"Resource '{resource.path}' exceeds the configured size limit of "
                f"{self.config.max_resource_bytes} bytes."
            )

        expected_root = skill_root / resource.kind
        self._validate_owned_path(
            file_path=resource.file_path,
            expected_root=expected_root,
            expected_relative_path=resource.path,
            item_label="Resource",
        )

    def _normalize_script_name(self, script_name: str) -> str:
        normalized = PurePosixPath(str(script_name).strip())
        if not str(normalized) or str(normalized) == ".":
            raise SkillRuntimeError("Script name must be a non-empty relative path.")
        if normalized.is_absolute():
            raise SkillRuntimeError("Script name must be relative to the skill root.")
        if any(part in {"..", ""} for part in normalized.parts):
            raise SkillRuntimeError(
                "Script name must not contain parent-directory traversal."
            )
        return normalized.as_posix()

    def _normalize_script_args(self, args: list[str] | None) -> list[str]:
        if args is None:
            return []
        if not isinstance(args, list):
            raise SkillRuntimeError(
                "Script args must be provided as a list of strings."
            )
        normalized_args = []
        for arg in args:
            if not isinstance(arg, str):
                raise SkillRuntimeError("Each script arg must be a string.")
            normalized_args.append(arg)
        return normalized_args

    def _validate_script(
        self,
        skill_root: Path,
        script: DiscoveredSkillScript,
    ) -> None:
        expected_root = skill_root / "scripts"
        self._validate_owned_path(
            file_path=script.file_path,
            expected_root=expected_root,
            expected_relative_path=script.path,
            item_label="Script",
        )

    def _validate_owned_path(
        self,
        file_path: Path,
        expected_root: Path,
        expected_relative_path: str,
        item_label: str,
    ) -> None:
        resolved_expected_root = expected_root.resolve()
        resolved_expected_path = (
            resolved_expected_root.parent / expected_relative_path
        ).resolve()
        resolved_file_path = file_path.resolve()

        if not resolved_expected_path.is_relative_to(resolved_expected_root):
            raise SkillRuntimeError(
                f"{item_label} '{expected_relative_path}' resolved outside its expected skill-owned root."
            )

        if resolved_file_path != resolved_expected_path:
            raise SkillRuntimeError(
                f"{item_label} '{expected_relative_path}' resolved outside its expected skill-owned root."
            )

    def _require_allowed_tool(
        self,
        skill_name: str,
        allowed_tools: list[str],
        tool_name: str,
    ) -> None:
        if is_tool_allowed(allowed_tools, tool_name):
            return
        raise SkillRuntimeError(
            f"Skill '{skill_name}' does not allow use of tool '{tool_name}' "
            "under its manifest policy."
        )

    def _execute_script(
        self,
        skill_name: str,
        skill_root: Path,
        script: DiscoveredSkillScript,
        normalized_args: list[str],
    ) -> dict[str, Any]:
        command = self._build_script_command(script, normalized_args)
        timeout_seconds = self.config.default_script_timeout_seconds
        cwd = skill_root.resolve()
        LOG.info(
            f"Executing skill script '{script.path}' for skill '{skill_name}' "
            f"in '{cwd}' with a {timeout_seconds}s timeout"
        )
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=False,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            LOG.warning(
                f"Skill script '{script.path}' timed out after {timeout_seconds}s"
            )
            stdout, stdout_truncated = self._truncate_output(exc.stdout)
            stderr, stderr_truncated = self._truncate_output(exc.stderr)
            return {
                "status": "timeout",
                "skill_name": skill_name,
                "script_name": script.path,
                "approved": True,
                "executed": True,
                "runner": script.runner,
                "args": normalized_args,
                "cwd": str(cwd),
                "timeout_seconds": timeout_seconds,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "outcome": f"Skill script timed out after {timeout_seconds} seconds.",
            }

        stdout, stdout_truncated = self._truncate_output(completed.stdout)
        stderr, stderr_truncated = self._truncate_output(completed.stderr)
        result = {
            "skill_name": skill_name,
            "script_name": script.path,
            "approved": True,
            "executed": True,
            "runner": script.runner,
            "args": normalized_args,
            "cwd": str(cwd),
            "timeout_seconds": timeout_seconds,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "exit_code": completed.returncode,
        }

        if completed.returncode == 0:
            result["status"] = "success"
            result["outcome"] = "Skill script completed successfully."
            return result
        LOG.warning(
            f"Skill script '{script.path}' exited with status {completed.returncode}"
        )
        result["status"] = "nonzero_exit"
        result["outcome"] = (
            f"Skill script exited with non-zero status {completed.returncode}."
        )
        return result

    def _build_script_command(
        self,
        script: DiscoveredSkillScript,
        normalized_args: list[str],
    ) -> list[str]:
        if script.runner == "python":
            return [sys.executable, str(script.file_path), *normalized_args]
        if script.runner == "shell":
            bash = shutil.which("bash")
            if not bash:
                raise SkillRuntimeError(
                    "Cannot run shell skill scripts: bash is not installed or not on PATH."
                )
            return [bash, str(script.file_path), *normalized_args]

    def _truncate_output(self, data: bytes | str | None) -> tuple[str, bool]:
        if data is None:
            return "", False

        raw = data.encode("utf-8", errors="replace") if isinstance(data, str) else data
        truncated = len(raw) > self.config.max_script_output_bytes
        if truncated:
            raw = raw[: self.config.max_script_output_bytes]
        return raw.decode("utf-8", errors="replace"), truncated
