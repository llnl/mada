# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from mada.core.autonomy import (
    build_autonomy_enabled_prompt,
    build_autonomy_followup_prompt,
    tail_text,
)


@pytest.mark.unit
class TestBuildAutonomyEnabledPrompt:
    def test_level_zero_returns_prompt_unchanged(self):
        prompt = "USER:\nHello"
        assert (
            build_autonomy_enabled_prompt(
                prompt,
                level=0,
                followups_used=0,
                followups_max=3,
            )
            == prompt
        )

    def test_level_positive_prefixes_internal_note_and_preserves_prompt(self):
        prompt = "USER:\nHello"
        out = build_autonomy_enabled_prompt(
            prompt,
            level=5,
            followups_used=1,
            followups_max=3,
        )

        assert out.endswith(prompt)
        assert out.startswith("[INTERNAL NOTE]")
        assert "Autonomous follow-ups are ENABLED" in out
        assert "bounded by a safety cap" in out
        assert "timer-based" in out or "periodic" in out
        assert "I can't continue sending messages on a timer by myself." in out


@pytest.mark.unit
class TestTailText:
    def test_empty_returns_empty(self):
        assert tail_text("", 10) == ""

    def test_non_positive_limit_returns_empty(self):
        assert tail_text("abc", 0) == ""
        assert tail_text("abc", -1) == ""

    def test_shorter_than_limit_returns_original(self):
        assert tail_text("abc", 10) == "abc"

    def test_longer_than_limit_returns_tail(self):
        assert tail_text("0123456789", 4) == "6789"


@pytest.mark.unit
class TestBuildAutonomyFollowupPrompt:
    def test_includes_internal_context_and_followup_task(self):
        out = build_autonomy_followup_prompt(
            "Continue printing pi digits",
            original_request="Print a new digit of pi every 10 seconds up to 8 digits",
            last_reply="Digits so far: 3.1415",
            assistant_buffer="... transcript ...",
            level=3,
            followups_used=1,
            followups_max=5,
        )
        assert out.startswith("[INTERNAL NOTE]")
        assert "[INTERNAL CONTEXT - DO NOT REPEAT]" in out
        assert "Original user request:" in out
        assert "Most recent assistant reply:" in out
        assert "User-visible transcript tail:" in out
        assert "Follow-up task:" in out
        assert "Continue printing pi digits" in out
