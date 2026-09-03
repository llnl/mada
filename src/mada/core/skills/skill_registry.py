"""
Discovery and indexing for manifest-based SKILL.md skills.
"""

from dataclasses import dataclass, field
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from .skill_manifest import parse_skill_manifest
from .utils import is_tool_allowed

LOG = logging.getLogger(__name__)


class SkillRegistryError(Exception):
    """Raised when manifest-based skill discovery or lookup fails."""


@dataclass(frozen=True)
class DiscoveredSkillResource:
    """Indexed metadata for one discovered skill resource file."""

    path: str
    kind: str
    file_path: Path
    size_bytes: int
    media_type: str
    text_readable: bool


@dataclass(frozen=True)
class DiscoveredSkillScript:
    """Indexed metadata for one discovered skill script file."""

    path: str
    runner: str
    file_path: Path


@dataclass(frozen=True)
class DiscoveredSkill:
    """Indexed manifest-based skill metadata used for lazy runtime loading."""

    name: str
    description: str
    manifest_path: Path
    root_path: Path
    license: str = ""
    compatibility: str = ""
    allowed_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, DiscoveredSkillResource] = field(default_factory=dict)
    scripts: Dict[str, DiscoveredSkillScript] = field(default_factory=dict)


class SkillRegistry:
    """Registry of manifest-discovered skills keyed by unique skill name."""

    TEXT_RESOURCE_EXTENSIONS = {
        ".csv",
        ".json",
        ".md",
        ".rst",
        ".tsv",
        ".txt",
        ".yaml",
        ".yml",
    }

    SUPPORTED_SCRIPT_RUNNERS = {
        ".py": "python",
        ".sh": "shell",
    }

    def __init__(self, skills: Iterable[DiscoveredSkill] = ()):
        self._skills: Dict[str, DiscoveredSkill] = {}

        for skill in skills:
            if skill.name in self._skills:
                existing = self._skills[skill.name]
                raise SkillRegistryError(
                    f"Duplicate skill name '{skill.name}' discovered at "
                    f"'{existing.manifest_path}' and '{skill.manifest_path}'."
                )
            self._skills[skill.name] = skill
            LOG.debug(
                f"Registered skill '{skill.name}' with "
                f"{len(skill.resources)} resource(s) and {len(skill.scripts)} script(s)"
            )

    @classmethod
    def discover(cls, roots: Iterable[Path]) -> "SkillRegistry":
        """Discover manifest-based skills under the provided root directories."""
        discovered: List[DiscoveredSkill] = []

        for root in roots:
            root_path = Path(root).resolve()
            if not root_path.exists():
                raise SkillRegistryError(
                    f"Skill discovery root '{root_path}' does not exist."
                )
            if not root_path.is_dir():
                raise SkillRegistryError(
                    f"Skill discovery root '{root_path}' is not a directory."
                )

            LOG.debug(f"Searching for skills under '{root_path}'")

            for manifest_path in sorted(root_path.rglob("SKILL.md")):
                skill_root = manifest_path.parent
                manifest = parse_skill_manifest(skill_root)
                discovered.append(
                    DiscoveredSkill(
                        name=manifest.name,
                        description=manifest.description,
                        manifest_path=manifest.manifest_path,
                        root_path=manifest.path,
                        license=manifest.license,
                        compatibility=manifest.compatibility,
                        allowed_tools=manifest.allowed_tools,
                        metadata=manifest.metadata,
                        resources=cls._discover_skill_resources(manifest.path),
                        scripts=cls._discover_skill_scripts(manifest.path),
                    )
                )

        LOG.info(f"Discovered {len(discovered)} manifest-based skill(s)")
        return cls(discovered)

    def list_skills(self) -> List[DiscoveredSkill]:
        """Return all discovered skills in deterministic name order."""
        return [self._skills[name] for name in sorted(self._skills)]

    def get_skill(self, name: str) -> DiscoveredSkill:
        """Return one discovered skill by name."""
        try:
            return self._skills[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._skills)) or "none"
            raise SkillRegistryError(
                f"Skill '{name}' was not found in the registry. Available skills: {available}."
            ) from exc

    def has_skills(self) -> bool:
        """Return True when the registry contains at least one skill."""
        return bool(self._skills)

    def has_skills_for_tool(self, tool_name: str = "load_skill") -> bool:
        """Return True when at least one discovered skill allows the tool."""
        return self._has_tool_for_content(tool_name, content_attr=None)

    def has_resources(self) -> bool:
        """Return True when at least one discovered skill has indexed resources."""
        return any(skill.resources for skill in self._skills.values())

    def has_scripts(self) -> bool:
        """Return True when at least one discovered skill has indexed scripts."""
        return any(skill.scripts for skill in self._skills.values())

    def has_resources_for_tool(self, tool_name: str = "read_skill_resource") -> bool:
        """Return True when at least one skill has resources and allows the tool."""
        return self._has_tool_for_content(tool_name, content_attr="resources")

    def has_scripts_for_tool(self, tool_name: str = "run_skill_script") -> bool:
        """Return True when at least one skill has scripts and allows the tool."""
        return self._has_tool_for_content(tool_name, content_attr="scripts")

    def get_resource(
        self, skill_name: str, resource_path: str
    ) -> DiscoveredSkillResource:
        """Return one discovered resource for a skill by normalized relative path."""
        skill = self.get_skill(skill_name)
        try:
            return skill.resources[resource_path]
        except KeyError as exc:
            available = ", ".join(sorted(skill.resources)) or "none"
            raise SkillRegistryError(
                f"Resource '{resource_path}' was not found for skill '{skill_name}'. "
                f"Available resources: {available}."
            ) from exc

    def get_script(self, skill_name: str, script_name: str) -> DiscoveredSkillScript:
        """Return one discovered script for a skill by normalized relative path."""
        skill = self.get_skill(skill_name)
        try:
            return skill.scripts[script_name]
        except KeyError as exc:
            available = ", ".join(sorted(skill.scripts)) or "none"
            raise SkillRegistryError(
                f"Script '{script_name}' was not found for skill '{skill_name}'. "
                f"Available scripts: {available}."
            ) from exc

    def skill_summaries(self) -> List[str]:
        """Return compact 'name: description' summaries for prompt advertisement."""
        return [f"{skill.name}: {skill.description}" for skill in self.list_skills()]

    def _has_tool_for_content(self, tool_name: str, content_attr: str | None) -> bool:
        """
        Return True when at least one skill has the given content and allows the tool.

        Args:
            tool_name: Runtime tool name to check.
            content_attr: Attribute name on `DiscoveredSkill` that must be
                truthy, such as "resources" or "scripts". None skips the
                content check entirely.

        Returns:
            True when at least one skill satisfies both conditions.
        """
        return any(
            (content_attr is None or getattr(skill, content_attr))
            and is_tool_allowed(skill.allowed_tools, tool_name)
            for skill in self._skills.values()
        )

    @classmethod
    def _iter_skill_files(
        cls,
        skill_root: Path,
        subdir_name: str,
        label: str,
    ) -> Iterator[Tuple[Path, str]]:
        """
        Yield visible files under one skill-owned subdirectory.

        Symlinks, directories, and hidden paths are skipped. A missing
        subdirectory yields nothing.

        Args:
            skill_root: Root directory of the skill.
            subdir_name: Skill-owned subdirectory to walk, such as "scripts".
            label: Human-readable name used in error messages.

        Yields:
            Tuples of the resolved candidate path and its normalized POSIX path
            relative to the skill root.

        Raises:
            SkillRegistryError: If the subdirectory exists but is not a directory.
        """
        subdir_root = skill_root / subdir_name
        if not subdir_root.exists():
            return
        if not subdir_root.is_dir():
            raise SkillRegistryError(
                f"Skill {label} root '{subdir_root}' exists but is not a directory."
            )

        for candidate in sorted(subdir_root.rglob("*")):
            if candidate.is_symlink() or not candidate.is_file():
                continue

            relative_path = candidate.relative_to(skill_root)
            if any(part.startswith(".") for part in relative_path.parts):
                continue

            yield candidate, relative_path.as_posix()

    @classmethod
    def _discover_skill_resources(
        cls, skill_root: Path
    ) -> Dict[str, DiscoveredSkillResource]:
        """
        Index readable resource files owned by one skill.

        Args:
            skill_root: Root directory of the skill.

        Returns:
            Mapping of normalized relative paths to discovered resources.
        """
        resources: Dict[str, DiscoveredSkillResource] = {}

        for kind in ("references", "assets"):
            for candidate, normalized_path in cls._iter_skill_files(
                skill_root, kind, "resource"
            ):
                media_type = (
                    mimetypes.guess_type(candidate.name)[0]
                    or "application/octet-stream"
                )
                resources[normalized_path] = DiscoveredSkillResource(
                    path=normalized_path,
                    kind=kind,
                    file_path=candidate.resolve(),
                    size_bytes=candidate.stat().st_size,
                    media_type=media_type,
                    text_readable=(
                        candidate.suffix.lower() in cls.TEXT_RESOURCE_EXTENSIONS
                        or media_type.startswith("text/")
                    ),
                )

        return resources

    @classmethod
    def _discover_skill_scripts(
        cls, skill_root: Path
    ) -> Dict[str, DiscoveredSkillScript]:
        """
        Index executable script files owned by one skill.

        Args:
            skill_root: Root directory of the skill.

        Returns:
            Mapping of normalized relative paths to discovered scripts.
        """
        scripts: Dict[str, DiscoveredSkillScript] = {}

        for candidate, normalized_path in cls._iter_skill_files(
            skill_root, "scripts", "scripts"
        ):
            runner = cls.SUPPORTED_SCRIPT_RUNNERS.get(candidate.suffix.lower())
            if not runner:
                continue

            scripts[normalized_path] = DiscoveredSkillScript(
                path=normalized_path,
                runner=runner,
                file_path=candidate.resolve(),
            )

        return scripts
