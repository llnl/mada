"""
Utilities for parsing manifest-based SKILL.md files.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

LOG = logging.getLogger(__name__)


class SkillManifestError(Exception):
    """Raised when a SKILL.md manifest cannot be parsed or validated."""


SUPPORTED_SKILL_TOOLS = {
    "load_skill",
    "read_skill_resource",
    "run_skill_script",
}

SUPPORTED_SKILL_MANIFEST_FIELDS = {
    "allowed_tools",
    "compatibility",
    "description",
    "license",
    "metadata",
    "name",
}


@dataclass(frozen=True)
class SkillManifest:
    """Parsed representation of a file-based skill manifest."""

    name: str
    description: str
    content: str
    path: Path
    manifest_path: Path
    license: str = ""
    compatibility: str = ""
    allowed_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _extract_frontmatter_parts(raw_text: str) -> Tuple[str, str]:
    """
    Split raw SKILL.md text into its YAML frontmatter and markdown body.

    Args:
        raw_text: Full text of a SKILL.md file.

    Returns:
        Tuple of the frontmatter text and the stripped markdown body.

    Raises:
        SkillManifestError: If the file is empty or the frontmatter delimiters
            are missing or unterminated.
    """
    lines: List[str] = raw_text.splitlines()
    if not lines:
        raise SkillManifestError("SKILL.md is empty.")

    if lines[0].strip() != "---":
        raise SkillManifestError(
            "SKILL.md must begin with YAML frontmatter delimited by '---'."
        )

    closing_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break

    if closing_index is None:
        raise SkillManifestError(
            "SKILL.md frontmatter is malformed: missing closing '---' delimiter."
        )

    frontmatter_text = "\n".join(lines[1:closing_index])
    body = "\n".join(lines[closing_index + 1 :]).strip()
    return frontmatter_text, body


def _parse_frontmatter(frontmatter_text: str) -> Dict[str, Any]:
    """
    Parse manifest frontmatter into a mapping.

    Args:
        frontmatter_text: YAML text extracted from between the delimiters.

    Returns:
        Parsed mapping, or an empty mapping when the frontmatter is blank.

    Raises:
        SkillManifestError: If the YAML is malformed or is not a mapping.
    """
    try:
        data = yaml.safe_load(frontmatter_text) if frontmatter_text.strip() else {}
    except yaml.YAMLError as exc:
        raise SkillManifestError(f"Malformed YAML frontmatter: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SkillManifestError(
            "Malformed YAML frontmatter: expected a top-level mapping."
        )
    return data


def _parse_allowed_tools(value: Any, manifest_path: Path) -> List[str]:
    """
    Validate and normalize a manifest's `allowed_tools` entry.

    An empty or absent entry means the skill permits every supported tool.

    Args:
        value: Raw `allowed_tools` value from the frontmatter.
        manifest_path: Path to the manifest, used in error messages.

    Returns:
        Normalized list of permitted tool names.

    Raises:
        SkillManifestError: If the value is not a list of unique, non-empty,
            supported tool names.
    """
    if value is None:
        return []

    if not isinstance(value, list):
        raise SkillManifestError(
            f"Skill manifest '{manifest_path}' has invalid 'allowed_tools': expected a list."
        )

    normalized_tools: List[str] = []
    seen_tools = set()
    for tool_name in value:
        if not isinstance(tool_name, str):
            raise SkillManifestError(
                f"Skill manifest '{manifest_path}' has invalid 'allowed_tools': "
                "entries must be non-empty strings."
            )
        normalized_name = tool_name.strip()
        if not normalized_name:
            raise SkillManifestError(
                f"Skill manifest '{manifest_path}' has invalid 'allowed_tools': "
                "entries must be non-empty strings."
            )
        if normalized_name not in SUPPORTED_SKILL_TOOLS:
            supported_tools = ", ".join(sorted(SUPPORTED_SKILL_TOOLS))
            raise SkillManifestError(
                f"Skill manifest '{manifest_path}' has invalid 'allowed_tools': "
                f"unknown tool '{normalized_name}'. Supported tools: {supported_tools}."
            )
        if normalized_name in seen_tools:
            raise SkillManifestError(
                f"Skill manifest '{manifest_path}' has invalid 'allowed_tools': "
                f"duplicate tool '{normalized_name}'."
            )
        seen_tools.add(normalized_name)
        normalized_tools.append(normalized_name)

    return normalized_tools


def _parse_string_field(
    manifest_data: Dict[str, Any],
    field_name: str,
    manifest_path: Path,
    *,
    required: bool = False,
) -> str:
    """
    Read and normalize one optional or required string field.

    Args:
        manifest_data: Parsed frontmatter mapping.
        field_name: Name of the field to read.
        manifest_path: Path to the manifest, used in error messages.
        required: If True, the field must resolve to a non-empty string.

    Returns:
        Stripped field value, or an empty string when absent and optional.

    Raises:
        SkillManifestError: If the value is not a string, or is empty while
            required.
    """
    value = manifest_data.get(field_name, "")
    if value is None:
        value = ""

    if value != "" and not isinstance(value, str):
        raise SkillManifestError(
            f"Skill manifest '{manifest_path}' has invalid '{field_name}': expected a string."
        )

    normalized_value = value.strip() if isinstance(value, str) else ""
    if required and not normalized_value:
        raise SkillManifestError(
            f"Skill manifest '{manifest_path}' must define a non-empty '{field_name}'."
        )

    return normalized_value


def parse_skill_manifest(skill_path: Path) -> SkillManifest:
    """
    Parse and validate the SKILL.md manifest for a single skill directory.

    Args:
        skill_path: Directory containing the SKILL.md file.

    Returns:
        A parsed SkillManifest instance.

    Raises:
        SkillManifestError: If the manifest is missing, malformed, or fails
            validation.
    """
    skill_path = Path(skill_path).resolve()
    manifest_path = skill_path / "SKILL.md"
    LOG.debug(f"Parsing skill manifest at '{manifest_path}'")

    if not manifest_path.exists():
        raise SkillManifestError(
            f"Skill directory '{skill_path}' does not contain required SKILL.md."
        )

    if not manifest_path.is_file():
        raise SkillManifestError(
            f"Manifest path '{manifest_path}' exists but is not a file."
        )

    raw_text = manifest_path.read_text(encoding="utf-8")
    frontmatter_text, content = _extract_frontmatter_parts(raw_text)
    manifest_data = _parse_frontmatter(frontmatter_text)

    unknown_fields = sorted(set(manifest_data) - SUPPORTED_SKILL_MANIFEST_FIELDS)
    if unknown_fields:
        raise SkillManifestError(
            f"Skill manifest '{manifest_path}' has unsupported fields: "
            f"{', '.join(unknown_fields)}."
        )

    name = _parse_string_field(
        manifest_data,
        "name",
        manifest_path,
        required=True,
    )
    description = _parse_string_field(
        manifest_data,
        "description",
        manifest_path,
        required=True,
    )
    license_value = _parse_string_field(manifest_data, "license", manifest_path)
    compatibility = _parse_string_field(manifest_data, "compatibility", manifest_path)
    allowed_tools = _parse_allowed_tools(
        manifest_data.get("allowed_tools"),
        manifest_path,
    )
    metadata = manifest_data.get("metadata", {})

    if not content:
        raise SkillManifestError(
            f"Skill manifest '{manifest_path}' must include non-empty markdown content "
            "after the YAML frontmatter."
        )

    if skill_path.name != name:
        raise SkillManifestError(
            f"Skill directory name '{skill_path.name}' must match manifest name '{name}'."
        )

    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise SkillManifestError(
            f"Skill manifest '{manifest_path}' has invalid 'metadata': expected a mapping."
        )

    LOG.debug(
        f"Parsed skill '{name}' with {len(allowed_tools) or 'all'} allowed tool(s)"
    )
    return SkillManifest(
        name=name,
        description=description,
        content=content,
        path=skill_path,
        manifest_path=manifest_path,
        license=license_value,
        compatibility=compatibility,
        allowed_tools=allowed_tools,
        metadata=metadata,
    )
