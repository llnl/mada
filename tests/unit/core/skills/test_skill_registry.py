# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from pathlib import Path

import pytest

from mada.core.skills.skill_registry import SkillRegistry, SkillRegistryError


def _write_skill(
    root: Path,
    name: str,
    allowed_tools: list[str] | None = None,
    resources: dict[str, str] | None = None,
    scripts: dict[str, str] | None = None,
) -> Path:
    """
    Write a skill directory with an optional set of resources and scripts.

    Args:
        root: Directory to create the skill folder inside.
        name: Skill directory and manifest name.
        allowed_tools: Optional manifest tool allowlist.
        resources: Mapping of relative resource paths to file contents.
        scripts: Mapping of relative script paths to file contents.

    Returns:
        Path to the created skill directory.
    """
    skill_dir = root / name
    skill_dir.mkdir(parents=True)

    frontmatter = f"name: {name}\ndescription: Skill {name}."
    if allowed_tools is not None:
        tool_lines = "\n".join(f"  - {tool}" for tool in allowed_tools)
        frontmatter += f"\nallowed_tools:\n{tool_lines}"

    (skill_dir / "SKILL.md").write_text(
        f"---\n{frontmatter}\n---\n\nDo the thing.\n",
        encoding="utf-8",
    )

    for relative_path, content in (resources or {}).items():
        target = skill_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    for relative_path, content in (scripts or {}).items():
        target = skill_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    return skill_dir


@pytest.mark.unit
class TestSkillRegistryDiscovery:
    def test_discovers_skills_under_root(self, tmp_path: Path):
        """Test that every skill folder under a root is discovered."""
        _write_skill(tmp_path, "alpha-skill")
        _write_skill(tmp_path, "beta-skill")

        registry = SkillRegistry.discover([tmp_path])

        assert registry.has_skills()
        assert [skill.name for skill in registry.list_skills()] == [
            "alpha-skill",
            "beta-skill",
        ]

    def test_empty_root_discovers_nothing(self, tmp_path: Path):
        """Test that a root with no skill folders yields an empty registry."""
        registry = SkillRegistry.discover([tmp_path])

        assert not registry.has_skills()
        assert registry.list_skills() == []

    def test_missing_root_raises(self, tmp_path: Path):
        """Test that a nonexistent discovery root is rejected."""
        with pytest.raises(SkillRegistryError, match="does not exist"):
            SkillRegistry.discover([tmp_path / "missing"])

    def test_duplicate_skill_names_raise(self, tmp_path: Path):
        """Test that two skills sharing a name are rejected."""
        first_root = tmp_path / "first"
        second_root = tmp_path / "second"
        _write_skill(first_root, "demo-skill")
        _write_skill(second_root, "demo-skill")

        with pytest.raises(SkillRegistryError, match="Duplicate skill name"):
            SkillRegistry.discover([first_root, second_root])


@pytest.mark.unit
class TestSkillRegistryIndexing:
    def test_indexes_resources_and_scripts(self, tmp_path: Path):
        """Test that resources and scripts are indexed by relative path."""
        _write_skill(
            tmp_path,
            "demo-skill",
            resources={"references/guide.md": "# Guide\n"},
            scripts={"scripts/run.py": "print('hi')\n"},
        )

        registry = SkillRegistry.discover([tmp_path])
        skill = registry.get_skill("demo-skill")

        assert "references/guide.md" in skill.resources
        assert "scripts/run.py" in skill.scripts
        assert skill.resources["references/guide.md"].text_readable is True
        assert skill.scripts["scripts/run.py"].runner == "python"

    def test_skips_hidden_and_unsupported_files(self, tmp_path: Path):
        """Test that hidden files and unsupported script runners are skipped."""
        _write_skill(
            tmp_path,
            "demo-skill",
            resources={"references/.hidden.md": "secret\n"},
            scripts={"scripts/notes.txt": "not a script\n"},
        )

        registry = SkillRegistry.discover([tmp_path])
        skill = registry.get_skill("demo-skill")

        assert skill.resources == {}
        assert skill.scripts == {}

    def test_unknown_skill_lookup_raises(self, tmp_path: Path):
        """Test that requesting an unknown skill is rejected."""
        _write_skill(tmp_path, "demo-skill")
        registry = SkillRegistry.discover([tmp_path])

        with pytest.raises(SkillRegistryError, match="was not found in the registry"):
            registry.get_skill("nope")

    def test_unknown_resource_lookup_raises(self, tmp_path: Path):
        """Test that requesting an unknown resource is rejected."""
        _write_skill(tmp_path, "demo-skill")
        registry = SkillRegistry.discover([tmp_path])

        with pytest.raises(SkillRegistryError, match="was not found for skill"):
            registry.get_resource("demo-skill", "references/missing.md")


@pytest.mark.unit
class TestSkillRegistryToolPolicy:
    def test_skill_without_allowlist_permits_every_tool(self, tmp_path: Path):
        """Test that omitting allowed_tools permits all runtime tools."""
        _write_skill(
            tmp_path,
            "demo-skill",
            resources={"references/guide.md": "# Guide\n"},
            scripts={"scripts/run.py": "print('hi')\n"},
        )

        registry = SkillRegistry.discover([tmp_path])

        assert registry.has_skills_for_tool("load_skill")
        assert registry.has_resources_for_tool("read_skill_resource")
        assert registry.has_scripts_for_tool("run_skill_script")

    def test_allowlist_restricts_tools(self, tmp_path: Path):
        """Test that allowed_tools blocks tools it does not name."""
        _write_skill(
            tmp_path,
            "demo-skill",
            allowed_tools=["load_skill"],
            scripts={"scripts/run.py": "print('hi')\n"},
        )

        registry = SkillRegistry.discover([tmp_path])

        assert registry.has_skills_for_tool("load_skill")
        assert not registry.has_scripts_for_tool("run_skill_script")

    def test_skill_summaries_are_name_and_description(self, tmp_path: Path):
        """Test that summaries used for planner advertisement are compact."""
        _write_skill(tmp_path, "demo-skill")

        registry = SkillRegistry.discover([tmp_path])

        assert registry.skill_summaries() == ["demo-skill: Skill demo-skill."]
