# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from pathlib import Path

import pytest

from mada.core.skills.skill_manifest import (
    SkillManifestError,
    parse_skill_manifest,
)


def _write_skill(
    root: Path, name: str, frontmatter: str, body: str = "Do the thing."
) -> Path:
    """
    Write a minimal skill directory containing a SKILL.md manifest.

    Args:
        root: Directory to create the skill folder inside.
        name: Skill directory name, which must match the manifest name.
        frontmatter: YAML frontmatter body, without the '---' delimiters.
        body: Markdown content placed after the frontmatter.

    Returns:
        Path to the created skill directory.
    """
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\n{frontmatter}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


@pytest.mark.unit
class TestParseSkillManifest:
    def test_parses_minimal_manifest(self, tmp_path: Path):
        """Test that a manifest with only required fields parses correctly."""
        skill_dir = _write_skill(
            tmp_path,
            "demo-skill",
            "name: demo-skill\ndescription: A demo skill.",
        )

        manifest = parse_skill_manifest(skill_dir)

        assert manifest.name == "demo-skill"
        assert manifest.description == "A demo skill."
        assert manifest.content == "Do the thing."
        assert manifest.allowed_tools == []

    def test_parses_optional_fields(self, tmp_path: Path):
        """Test that optional manifest fields are read and normalized."""
        skill_dir = _write_skill(
            tmp_path,
            "demo-skill",
            (
                "name: demo-skill\n"
                "description: A demo skill.\n"
                "license: Apache-2.0\n"
                "compatibility: mada>=0.2\n"
                "allowed_tools:\n"
                "  - load_skill\n"
                "  - run_skill_script\n"
                "metadata:\n"
                "  owner: docs-team"
            ),
        )

        manifest = parse_skill_manifest(skill_dir)

        assert manifest.license == "Apache-2.0"
        assert manifest.compatibility == "mada>=0.2"
        assert manifest.allowed_tools == ["load_skill", "run_skill_script"]
        assert manifest.metadata == {"owner": "docs-team"}

    def test_missing_manifest_raises(self, tmp_path: Path):
        """Test that a directory without SKILL.md is rejected."""
        empty_dir = tmp_path / "no-manifest"
        empty_dir.mkdir()

        with pytest.raises(SkillManifestError, match="does not contain required"):
            parse_skill_manifest(empty_dir)

    def test_missing_frontmatter_raises(self, tmp_path: Path):
        """Test that a manifest without YAML frontmatter is rejected."""
        skill_dir = tmp_path / "demo-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("No frontmatter here.\n", encoding="utf-8")

        with pytest.raises(
            SkillManifestError, match="must begin with YAML frontmatter"
        ):
            parse_skill_manifest(skill_dir)

    def test_unterminated_frontmatter_raises(self, tmp_path: Path):
        """Test that frontmatter without a closing delimiter is rejected."""
        skill_dir = tmp_path / "demo-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: demo-skill\ndescription: A demo skill.\n",
            encoding="utf-8",
        )

        with pytest.raises(SkillManifestError, match="missing closing"):
            parse_skill_manifest(skill_dir)

    def test_missing_required_field_raises(self, tmp_path: Path):
        """Test that a manifest without a description is rejected."""
        skill_dir = _write_skill(tmp_path, "demo-skill", "name: demo-skill")

        with pytest.raises(SkillManifestError, match="non-empty 'description'"):
            parse_skill_manifest(skill_dir)

    def test_directory_name_must_match_manifest_name(self, tmp_path: Path):
        """Test that a mismatched directory and manifest name is rejected."""
        skill_dir = _write_skill(
            tmp_path,
            "wrong-name",
            "name: demo-skill\ndescription: A demo skill.",
        )

        with pytest.raises(SkillManifestError, match="must match manifest name"):
            parse_skill_manifest(skill_dir)

    def test_unknown_field_raises(self, tmp_path: Path):
        """Test that unsupported manifest fields are rejected."""
        skill_dir = _write_skill(
            tmp_path,
            "demo-skill",
            "name: demo-skill\ndescription: A demo skill.\nunexpected: true",
        )

        with pytest.raises(SkillManifestError, match="unsupported fields"):
            parse_skill_manifest(skill_dir)

    def test_unknown_allowed_tool_raises(self, tmp_path: Path):
        """Test that an unrecognized entry in allowed_tools is rejected."""
        skill_dir = _write_skill(
            tmp_path,
            "demo-skill",
            (
                "name: demo-skill\n"
                "description: A demo skill.\n"
                "allowed_tools:\n"
                "  - not_a_real_tool"
            ),
        )

        with pytest.raises(SkillManifestError, match="unknown tool"):
            parse_skill_manifest(skill_dir)

    def test_empty_content_raises(self, tmp_path: Path):
        """Test that a manifest with no markdown body is rejected."""
        skill_dir = _write_skill(
            tmp_path,
            "demo-skill",
            "name: demo-skill\ndescription: A demo skill.",
            body="",
        )

        with pytest.raises(SkillManifestError, match="non-empty markdown content"):
            parse_skill_manifest(skill_dir)
