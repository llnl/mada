# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Shared setup for manifest-based skills.

Builds the skill registry, runtime, and runtime tools from an application
configuration so every interface initializes skills the same way.

"""

import logging
from typing import Any, List, Tuple

from mada.core.config import AppConfig
from mada.core.skills.skill_approval import build_skill_script_approver
from mada.core.skills.skill_registry import SkillRegistry
from mada.core.skills.skill_runtime import SkillRuntime
from mada.core.skills.skill_tools import (
    build_load_skill_tool,
    build_read_skill_resource_tool,
    build_run_skill_script_tool,
)

LOG = logging.getLogger(__name__)


def initialize_skill_state(config: AppConfig) -> Tuple[SkillRegistry, List[Any]]:
    """
    Discover manifest-based skills and build their runtime tools.

    Args:
        config: Application configuration containing the resolved skills

    Returns:
        Tuple containing the populated skill registry and the list of runtime
        tools to expose to the planner.
    """
    skills_config = config.skills
    LOG.debug(
        f"Initializing skills from {len(skills_config.skill_paths)} configured path(s)"
    )
    skill_registry = SkillRegistry.discover(skills_config.skill_paths)
    skill_runtime = SkillRuntime(
        skill_registry,
        config=skills_config.skill_runtime,
        script_approver=build_skill_script_approver(skills_config.skill_runtime),
    )

    skill_tools = []
    if skill_registry.has_skills_for_tool("load_skill"):
        skill_tools.append(build_load_skill_tool(skill_runtime))
    if skill_registry.has_resources_for_tool("read_skill_resource"):
        skill_tools.append(build_read_skill_resource_tool(skill_runtime))
    if skill_registry.has_scripts_for_tool("run_skill_script"):
        skill_tools.append(build_run_skill_script_tool(skill_runtime))

    LOG.info(f"Skill setup complete with {len(skill_tools)} runtime tool(s) available")
    return skill_registry, skill_tools
