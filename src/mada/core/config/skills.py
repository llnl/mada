# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Configuration for manifest-based skills.

Defines `SkillsConfig`, which validates skill discovery roots and runtime
policy, and `SkillRuntimeConfig`, which holds limits and approval settings for
reading skill resources and executing skill scripts.
"""

import logging
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

LOG = logging.getLogger(__name__)


@dataclass
class SkillRuntimeConfig:
    """
    Runtime policy configuration for manifest-based skills.

    Attributes:
        default_script_timeout_seconds: Default timeout for skill script execution.
        max_script_output_bytes: Maximum captured stdout/stderr size for a skill script.
        max_resource_bytes: Maximum readable size for a skill resource file.
        default_skill_script_approval_mode: Approval mode applied when no
            specific rule matches a script. One of "prompt", "approve", or "deny".
        skill_script_approval_modes: Per-skill and per-script approval overrides.
            Keys are matched most specific first: "skill_name:script_name", then
            "skill_name", then "*" as a catch-all. Values are "approve" or "deny".
    """

    default_script_timeout_seconds: int = 30
    max_script_output_bytes: int = 32 * 1024
    max_resource_bytes: int = 32 * 1024
    default_skill_script_approval_mode: str = "prompt"
    skill_script_approval_modes: Dict[str, str] = field(default_factory=dict)


@dataclass
class SkillsConfig:
    """
    Configuration for discovering and running manifest-based skills.

    Attributes:
        skill_paths: Directories searched recursively for skill folders
            containing `SKILL.md`. Entries are normalized to absolute `Path`
            objects, resolved against `skill_path_base_dir` when relative.
        skill_runtime: Runtime limits and script approval policy.
    """

    skill_paths: List[Path] = field(default_factory=list)
    skill_runtime: SkillRuntimeConfig = field(default_factory=SkillRuntimeConfig)
    skill_path_base_dir: InitVar[str | Path | None] = None

    def __post_init__(self, skill_path_base_dir: str | Path | None) -> None:
        """
        Validate skill paths and normalize them to absolute `Path` objects.

        Args:
            skill_path_base_dir: Directory used to resolve relative skill paths. When
                omitted, relative paths are resolved against the process
                working directory.

        Raises:
            ValueError: If `skill_paths` is not a list, or any entry is not a
                non-empty string or path.
        """
        if not isinstance(self.skill_paths, list):
            raise ValueError("'skills.skill_paths' must be a list of paths.")

        base_dir = (
            Path(skill_path_base_dir).resolve() if skill_path_base_dir else Path.cwd()
        )
        resolved_paths: List[Path] = []

        for raw_path in self.skill_paths:
            if isinstance(raw_path, Path):
                skill_path = raw_path
            elif isinstance(raw_path, str) and raw_path.strip():
                skill_path = Path(raw_path.strip())
            else:
                raise ValueError(
                    "Each entry in 'skills.skill_paths' must be a non-empty path."
                )

            if not skill_path.is_absolute():
                skill_path = base_dir / skill_path
            resolved_paths.append(skill_path.resolve())

        self.skill_paths = resolved_paths
        LOG.debug(f"Resolved {len(resolved_paths)} skill discovery path(s)")


def load_skills_config(
    skills_block: Dict[str, Any] | None,
    skill_path_base_dir: str | Path | None = None,
) -> SkillsConfig:
    """
    Build a `SkillsConfig` from the nested `skills` configuration block.

    Args:
        skills_block: Raw `skills` mapping, or None when not configured.
        skill_path_base_dir: Directory used to resolve relative skill paths.

    Returns:
        A validated skills configuration.

    Raises:
        ValueError: If the block or its `skill_runtime` entry is not a mapping.
    """
    if skills_block is None:
        return SkillsConfig(skill_path_base_dir=skill_path_base_dir)

    if not isinstance(skills_block, dict):
        raise ValueError("'skills' must be an object")

    runtime_entry = skills_block.get("skill_runtime")
    if runtime_entry is None:
        runtime_config = SkillRuntimeConfig()
    elif isinstance(runtime_entry, dict):
        runtime_config = SkillRuntimeConfig(**runtime_entry)
    else:
        raise ValueError(
            "'skills.skill_runtime' must be a mapping of runtime settings."
        )

    return SkillsConfig(
        skill_paths=skills_block.get("skill_paths") or [],
        skill_runtime=runtime_config,
        skill_path_base_dir=skill_path_base_dir,
    )
