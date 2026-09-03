# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from pathlib import Path

import pytest

from mada.core.config import SkillRuntimeConfig
from mada.core.skills.skill_approval import (
    DenyAllSkillScriptApprover,
    PolicyBasedSkillScriptApprover,
    PromptingSkillScriptApprover,
    build_skill_script_approver,
)
from mada.core.skills.skill_registry import SkillRegistry
from mada.core.skills.skill_runtime import SkillRuntime, SkillRuntimeError


def _write_skill(
    root: Path,
    name: str = "demo-skill",
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

    for relative_path, content in {**(resources or {}), **(scripts or {})}.items():
        target = skill_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    return skill_dir


def _runtime(root: Path, **runtime_kwargs) -> SkillRuntime:
    """
    Build a SkillRuntime over every skill discovered under a root.

    Args:
        root: Discovery root containing skill folders.
        **runtime_kwargs: Overrides passed to `SkillRuntime`.

    Returns:
        A runtime wired to the discovered registry.
    """
    registry = SkillRegistry.discover([root])
    return SkillRuntime(registry, **runtime_kwargs)


@pytest.mark.unit
class TestLoadSkill:
    def test_loads_manifest_content(self, tmp_path: Path):
        """Test that the markdown body of a skill is returned."""
        _write_skill(tmp_path)

        assert _runtime(tmp_path).load_skill("demo-skill") == "Do the thing."

    def test_disallowed_tool_raises(self, tmp_path: Path):
        """Test that a manifest allowlist excluding load_skill is enforced."""
        _write_skill(tmp_path, allowed_tools=["run_skill_script"])

        with pytest.raises(SkillRuntimeError, match="does not allow use of tool"):
            _runtime(tmp_path).load_skill("demo-skill")


@pytest.mark.unit
class TestReadSkillResource:
    def test_reads_text_resource(self, tmp_path: Path):
        """Test that an indexed text resource is returned verbatim."""
        _write_skill(tmp_path, resources={"references/guide.md": "# Guide\n"})

        content = _runtime(tmp_path).read_skill_resource(
            "demo-skill", "references/guide.md"
        )

        assert content == "# Guide\n"

    def test_traversal_path_raises(self, tmp_path: Path):
        """Test that parent-directory traversal in a resource path is rejected."""
        _write_skill(tmp_path, resources={"references/guide.md": "# Guide\n"})

        with pytest.raises(SkillRuntimeError, match="parent-directory traversal"):
            _runtime(tmp_path).read_skill_resource("demo-skill", "../../etc/passwd")

    def test_absolute_path_raises(self, tmp_path: Path):
        """Test that an absolute resource path is rejected."""
        _write_skill(tmp_path, resources={"references/guide.md": "# Guide\n"})

        with pytest.raises(SkillRuntimeError, match="must be relative"):
            _runtime(tmp_path).read_skill_resource("demo-skill", "/etc/passwd")

    def test_oversized_resource_raises(self, tmp_path: Path):
        """Test that a resource above the configured size limit is rejected."""
        _write_skill(tmp_path, resources={"references/guide.md": "x" * 100})
        config = SkillRuntimeConfig(max_resource_bytes=10)

        with pytest.raises(
            SkillRuntimeError, match="exceeds the configured size limit"
        ):
            _runtime(tmp_path, config=config).read_skill_resource(
                "demo-skill", "references/guide.md"
            )


@pytest.mark.unit
class TestRunSkillScript:
    def test_denied_by_default_approver(self, tmp_path: Path):
        """Test that the default approver blocks script execution."""
        _write_skill(tmp_path, scripts={"scripts/run.py": "print('hi')\n"})

        result = _runtime(tmp_path).run_skill_script("demo-skill", "scripts/run.py")

        assert result["status"] == "denied"
        assert result["approved"] is False
        assert result["executed"] is False

    def test_approved_script_runs(self, tmp_path: Path):
        """Test that an approved script executes and captures stdout."""
        _write_skill(
            tmp_path, scripts={"scripts/run.py": "print('hello from skill')\n"}
        )
        approver = PolicyBasedSkillScriptApprover(default_mode="approve")

        result = _runtime(tmp_path, script_approver=approver).run_skill_script(
            "demo-skill", "scripts/run.py"
        )

        assert result["status"] == "success"
        assert result["exit_code"] == 0
        assert "hello from skill" in result["stdout"]

    def test_nonzero_exit_is_reported(self, tmp_path: Path):
        """Test that a failing script is reported without raising."""
        _write_skill(
            tmp_path,
            scripts={"scripts/run.py": "import sys\nsys.exit(3)\n"},
        )
        approver = PolicyBasedSkillScriptApprover(default_mode="approve")

        result = _runtime(tmp_path, script_approver=approver).run_skill_script(
            "demo-skill", "scripts/run.py"
        )

        assert result["status"] == "nonzero_exit"
        assert result["exit_code"] == 3

    def test_disallowed_tool_is_denied(self, tmp_path: Path):
        """Test that a manifest allowlist excluding run_skill_script is enforced."""
        _write_skill(
            tmp_path,
            allowed_tools=["load_skill"],
            scripts={"scripts/run.py": "print('hi')\n"},
        )
        approver = PolicyBasedSkillScriptApprover(default_mode="approve")

        result = _runtime(tmp_path, script_approver=approver).run_skill_script(
            "demo-skill", "scripts/run.py"
        )

        assert result["status"] == "denied"
        assert result["executed"] is False


@pytest.mark.unit
class TestApprovalPolicy:
    def test_deny_all_approver_denies(self):
        """Test that the default approver denies every request."""
        approver = DenyAllSkillScriptApprover()
        decision = approver.approve_skill_script(_approval_request())

        assert decision.approved is False

    def test_per_script_override_wins_over_skill(self):
        """Test that a skill:script key takes precedence over a skill key."""
        approver = PolicyBasedSkillScriptApprover(
            default_mode="deny",
            skill_modes={
                "demo-skill": "approve",
                "demo-skill:scripts/run.py": "deny",
            },
        )

        decision = approver.approve_skill_script(_approval_request())

        assert decision.approved is False

    def test_skill_override_wins_over_wildcard(self):
        """Test that a skill key takes precedence over the '*' catch-all."""
        approver = PolicyBasedSkillScriptApprover(
            default_mode="deny",
            skill_modes={"demo-skill": "approve", "*": "deny"},
        )

        decision = approver.approve_skill_script(_approval_request())

        assert decision.approved is True

    def test_default_mode_applies_when_nothing_matches(self):
        """Test that the default mode is used when no override matches."""
        approver = PolicyBasedSkillScriptApprover(default_mode="approve")

        decision = approver.approve_skill_script(_approval_request())

        assert decision.approved is True

    def test_unrecognized_mode_falls_back_to_deny(self):
        """Test that an invalid default mode is treated as deny."""
        approver = PolicyBasedSkillScriptApprover(default_mode="maybe")

        decision = approver.approve_skill_script(_approval_request())

        assert decision.approved is False


@pytest.mark.unit
class TestBuildSkillScriptApprover:
    def test_prompt_mode_builds_interactive_approver(self):
        """Test that 'prompt' selects the interactive approver."""
        config = SkillRuntimeConfig(default_skill_script_approval_mode="prompt")

        approver = build_skill_script_approver(config)

        assert isinstance(approver, PromptingSkillScriptApprover)

    def test_policy_mode_builds_policy_approver(self):
        """Test that a non-prompt mode selects the policy approver."""
        config = SkillRuntimeConfig(
            default_skill_script_approval_mode="approve",
            skill_script_approval_modes={"demo-skill": "deny"},
        )

        approver = build_skill_script_approver(config)

        assert isinstance(approver, PolicyBasedSkillScriptApprover)
        assert approver.default_mode == "approve"
        assert approver.skill_modes == {"demo-skill": "deny"}

    def test_prompting_approver_denies_on_no(self):
        """Test that anything other than an explicit yes is a denial."""
        approver = PromptingSkillScriptApprover(
            input_func=lambda _: "n",
            output_func=lambda _: None,
        )

        decision = approver.approve_skill_script(_approval_request())

        assert decision.approved is False

    def test_prompting_approver_approves_on_yes(self):
        """Test that an explicit yes approves the script."""
        approver = PromptingSkillScriptApprover(
            input_func=lambda _: "y",
            output_func=lambda _: None,
        )

        decision = approver.approve_skill_script(_approval_request())

        assert decision.approved is True


def _approval_request():
    """
    Build a representative approval request for policy tests.

    Returns:
        An approval request for 'scripts/run.py' in 'demo-skill'.
    """
    from mada.core.skills.skill_approval import SkillScriptApprovalRequest

    return SkillScriptApprovalRequest(
        skill_name="demo-skill",
        script_name="scripts/run.py",
        script_path=Path("/tmp/demo-skill/scripts/run.py"),
        runner="python",
    )
