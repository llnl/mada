"""
Runtime tools for manifest-based skills.
"""

import logging
from typing import Any, Callable

from .skill_runtime import SkillRuntime

LOG = logging.getLogger(__name__)


def build_load_skill_tool(skill_runtime: SkillRuntime) -> Callable[[str], str]:
    """
    Build a runtime tool that loads full SKILL.md content by skill name.

    Args:
        skill_runtime: Runtime access layer for manifest-based skills.

    Returns:
        Callable tool that returns the full content for a named skill.
    """

    def load_skill(skill_name: str) -> str:
        """
        Load the full SKILL.md body for a manifest-discovered skill.

        Args:
            skill_name: Unique manifest skill name to load.

        Returns:
            Full markdown content of the requested skill.
        """
        LOG.debug(f"Tool 'load_skill' invoked for skill '{skill_name}'")
        return skill_runtime.load_skill(skill_name)

    # Keep the callable self-describing for the runtime/tooling layers.
    load_skill.__name__ = "load_skill"
    load_skill.name = "load_skill"
    load_skill.description = (
        "Load the full SKILL.md instructions for a manifest-based skill by name."
    )
    LOG.debug("Built runtime tool 'load_skill'")
    return load_skill


def build_read_skill_resource_tool(
    skill_runtime: SkillRuntime,
) -> Callable[[str, str], str]:
    """
    Build a runtime tool that loads one discovered skill resource on demand.

    Args:
        skill_runtime: Runtime access layer for manifest-based skills.

    Returns:
        Callable tool that returns validated text resource content.
    """

    def read_skill_resource(skill_name: str, resource_path: str) -> str:
        """
        Load one discovered resource file for a manifest-based skill.

        Args:
            skill_name: Unique manifest skill name.
            resource_path: Relative resource path such as 'references/guide.md'.

        Returns:
            Full text content of the requested resource.
        """
        LOG.debug(
            f"Tool 'read_skill_resource' invoked for skill '{skill_name}' "
            f"and resource '{resource_path}'"
        )
        return skill_runtime.read_skill_resource(skill_name, resource_path)

    read_skill_resource.__name__ = "read_skill_resource"
    read_skill_resource.name = "read_skill_resource"
    read_skill_resource.description = (
        "Read a discovered text resource for a manifest-based skill by skill name "
        "and relative resource path."
    )
    LOG.debug("Built runtime tool 'read_skill_resource'")
    return read_skill_resource


def build_run_skill_script_tool(
    skill_runtime: SkillRuntime,
) -> Callable[[str, str, list[str] | None], dict[str, Any]]:
    """
    Build a runtime tool that validates, approves, and executes one skill script.

    Args:
        skill_runtime: Runtime access layer for manifest-based skills.

    Returns:
        Callable tool that returns a structured execution result.
    """

    def run_skill_script(
        skill_name: str,
        script_name: str,
        args: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Validate, approve, and execute one discovered skill script.

        Args:
            skill_name: Unique manifest skill name.
            script_name: Relative script path such as 'scripts/report.py'.
            args: Optional argument list for the script process.

        Returns:
            Structured approval and execution result for the script.
        """
        LOG.info(
            f"Tool 'run_skill_script' invoked for skill '{skill_name}' "
            f"and script '{script_name}'"
        )
        return skill_runtime.run_skill_script(skill_name, script_name, args=args)

    run_skill_script.__name__ = "run_skill_script"
    run_skill_script.name = "run_skill_script"
    run_skill_script.description = (
        "Validate, request approval for, and execute a discovered manifest-based "
        "skill script by skill name and relative script path."
    )
    LOG.debug("Built runtime tool 'run_skill_script'")
    return run_skill_script
