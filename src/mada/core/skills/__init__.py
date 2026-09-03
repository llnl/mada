# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Manifest-based skills package.

This package provides discovery, validated runtime access, and approval
handling for manifest-based skills. A skill is a directory containing a
`SKILL.md` file, along with optional `references/`, `assets/`, and `scripts/`
subdirectories. It exposes the runtime tool builders used to expose skill
content to the planning agent.

Modules:
    skill_approval:
        Defines approvers for skill script execution, including
        [`PromptingSkillScriptApprover`][core.skills.skill_approval.PromptingSkillScriptApprover],
        [`PolicyBasedSkillScriptApprover`][core.skills.skill_approval.PolicyBasedSkillScriptApprover],
        and the [`build_skill_script_approver`][core.skills.skill_approval.build_skill_script_approver]
        factory.
    skill_manifest:
        Provides [`parse_skill_manifest`][core.skills.skill_manifest.parse_skill_manifest]
        for parsing and validating `SKILL.md` frontmatter and content.
    skill_registry:
        Provides [`SkillRegistry`][core.skills.skill_registry.SkillRegistry]
        for discovering and indexing skills, their resources, and their
        scripts under configured discovery roots.
    skill_runtime:
        Provides [`SkillRuntime`][core.skills.skill_runtime.SkillRuntime] for
        validated, policy-checked access to skill content, resources, and
        script execution.
    skill_setup:
        Provides [`initialize_skill_state`][core.skills.skill_setup.initialize_skill_state]
        for building a skill registry and its runtime tools from application
        configuration.
    skill_tools:
        Provides the runtime tool builders (`build_load_skill_tool`,
        `build_read_skill_resource_tool`, `build_run_skill_script_tool`) that
        expose skill content to the planning agent.
    utils:
        Provides shared helpers, including
        [`is_tool_allowed`][core.skills.utils.is_tool_allowed] for checking a
        skill's manifest tool allowlist.
"""
