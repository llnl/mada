# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Durable, transactionally reserved simulation budgets and critique records."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path


def execution_requested(text: str, mode: str = "auto") -> bool:
    """Conservative request routing, never inferred from earlier chat messages.

    Analysis mode always wins. Auto requires an explicit execution request;
    ambiguous requests stay read-only. This is not a general intent classifier.
    """
    text = text.strip().lower().replace("’", "'")
    if mode == "analysis" or re.search(
        r"\b(?:do not|don't|dont|never|without)\b[^.\n]{0,100}"
        r"\b(?:run|runs|running|rerun|submit|submitting|generate|execute|launch|start|starting|continue|resume|new simulations?|new jobs?)\b"
        r"|\bno\s+(?:(?:new|more|additional|further|extra)\s+)?"
        r"(?:simulations?|jobs?|runs?|submissions?|executions?|running|submitting)\b"
        r"|\bno\s+(?:need|permission|authorization)\s+to\s+"
        r"(?:run|submit|generate|execute|launch|start|continue|resume)\b"
        r"|\banalysis[- ]only\b",
        text,
    ):
        return False
    return any(
        not re.search(
            r"\b(?:analysis|review|summary|post[- ]processing|existing results)\b",
            match.group(),
        )
        for match in re.finditer(
            r"(?:^|[.!?,;\n]\s*)(?:(?:please|can you|could you|i want to|let's)\s+)?"
            r"(?:run|execute|launch|start|continue|resume|submit|generate)\b[^.!?\n]{0,160}"
            r"\b(?:campaign|simulations?|experiments?|jobs?)\b[^.!?;\n]*",
            text,
        )
    )


def run_identity(run: dict) -> str:
    # Ignore display IDs: renaming a submitted run must not make it new work.
    value = {key: run[key] for key in ("run_location", "command", "args") if key in run}
    return json.dumps(value or run, sort_keys=True, separators=(",", ":"))


class CampaignLedger:
    def __init__(self, path: str, campaign_id: str, policy: dict):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.campaign_id = campaign_id
        self.policy = {
            "limit": policy["limit"],
            "initial_limit": policy.get("initial_limit", policy["limit"]),
            "critique_agents": policy.get("critique_agents", []),
        }
        with self._transaction() as state:
            if state["policy"] != self.policy:
                raise ValueError(
                    "Campaign policy changed; existing budget cannot be reset or increased"
                )

    @contextmanager
    def _transaction(self):
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS campaigns (id TEXT PRIMARY KEY, state TEXT NOT NULL)"
            )
            row = connection.execute(
                "SELECT state FROM campaigns WHERE id=?", (self.campaign_id,)
            ).fetchone()
            state = (
                json.loads(row[0])
                if row
                else {
                    "policy": self.policy,
                    "runs": [],
                    "critiques": {},
                    "results_available": False,
                    "comparable": False,
                }
            )
            if state["policy"] != self.policy:
                raise ValueError(
                    "Campaign policy changed; existing budget cannot be reset or increased"
                )
            yield state
            connection.execute(
                "INSERT OR REPLACE INTO campaigns VALUES (?, ?)",
                (self.campaign_id, json.dumps(state, allow_nan=False)),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def snapshot(self) -> dict:
        with self._transaction() as state:
            return json.loads(json.dumps(state))

    @staticmethod
    def _run_ids(runs: list) -> set[str]:
        if (
            not isinstance(runs, list)
            or not runs
            or any(not isinstance(run, dict) for run in runs)
        ):
            raise ValueError("Persistent campaigns require explicit run objects")
        ids = [run.get("id") for run in runs]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (str, int))
            or not str(value).strip()
            for value in ids
        ):
            raise ValueError("Persistent runs require a nonempty string or integer id")
        normalized = {str(value) for value in ids}
        if len(normalized) != len(ids):
            raise ValueError("Persistent run IDs must be unique")
        return normalized

    def import_submitted(self, runs: list) -> int:
        """Explicit administrative migration of known pre-existing submissions.

        This never submits work or resets a nonempty ledger. Repeating the same
        import is idempotent. Callers must reconcile actual scheduler history.
        """
        if (
            not isinstance(runs, list)
            or not 1 <= len(runs) <= self.policy["limit"]
            or any(not isinstance(run, dict) or not run for run in runs)
        ):
            raise ValueError(
                "Import requires explicit previously submitted runs within the campaign limit"
            )
        self._run_ids(runs)
        identities = {run_identity(run) for run in runs}
        if len(identities) != len(runs):
            raise ValueError("Import contains duplicate runs")
        with self._transaction() as state:
            if state["runs"]:
                if {run_identity(run) for run in state["runs"]} != identities:
                    raise ValueError(
                        "Cannot replace an existing campaign ledger with a different import"
                    )
            else:
                state["runs"] = runs
                state["imported_submissions"] = True
            return len(state["runs"])

    def reserve(self, runs: list) -> int:
        run_ids = self._run_ids(runs)
        identities = [run_identity(run) for run in runs]
        with self._transaction() as state:
            existing = {run_identity(run) for run in state["runs"]}
            count = len(state["runs"])
            if len(set(identities)) != len(identities) or existing.intersection(
                identities
            ):
                raise ValueError(
                    "Duplicate submission blocked; reconcile the original jobs"
                )
            if run_ids.intersection(str(run["id"]) for run in state["runs"]):
                raise ValueError("Persistent run IDs must be unique across submissions")
            if count + len(runs) > self.policy["limit"]:
                raise ValueError("Simulation budget exceeded for this campaign")
            if count + len(runs) > self.policy["initial_limit"]:
                if count < self.policy["initial_limit"]:
                    raise ValueError(
                        "Initial batch is too large; reserve the follow-up slot"
                    )
                if not state["results_available"] or not state["comparable"]:
                    raise ValueError(
                        "Follow-up locked: completed, comparable initial results are required"
                    )
                required = self.policy["critique_agents"]
                if not required or not all(
                    state["critiques"].get(agent, {}).get("recommend_followup")
                    for agent in required
                ):
                    raise ValueError(
                        "Follow-up locked: each required specialist must record a critique recommending it"
                    )
            state["runs"].extend(runs)
            # A new observation cycle invalidates any previous approvals.
            state["critiques"] = {}
            state["results_available"] = False
            state["comparable"] = False
            return len(state["runs"])

    def observe(self, report: dict) -> None:
        """Accept evidence only for the exact reserved manifest, never other runs."""
        manifest = report.get("manifest_runs")
        results = report.get("runs")
        if not isinstance(manifest, list) or not isinstance(results, list):
            return
        if any(not isinstance(run, dict) for run in manifest + results):
            return
        with self._transaction() as state:
            if len(state["runs"]) < self.policy["initial_limit"]:
                return
            if any(
                run.get("id") is None or not str(run["id"]).strip()
                for run in state["runs"] + manifest + results
            ):
                return
            reserved = {str(run["id"]): run_identity(run) for run in state["runs"]}
            reported = {str(run["id"]): run_identity(run) for run in manifest}
            if (
                len(reserved) != len(state["runs"])
                or len(reported) != len(manifest)
                or reported != reserved
                or len(results) != len(reserved)
                or {str(run["id"]) for run in results} != reserved.keys()
            ):
                return
            ready = all(
                run.get("time_integration_complete_in_log")
                and run.get("velocity_probe")
                for run in results
            )
            comparable = (
                report.get("comparison", {}).get("same_final_saved_time") is True
            )
            digest = hashlib.sha256(
                json.dumps(
                    {
                        "runs": [
                            {
                                key: run.get(key)
                                for key in (
                                    "id",
                                    "parameters",
                                    "velocity_probe",
                                    "time_integration_complete_in_log",
                                )
                            }
                            for run in sorted(
                                results, key=lambda run: str(run.get("id"))
                            )
                        ],
                        "comparison": {"same_final_saved_time": comparable},
                    },
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            if state.get("evidence_digest") != digest:
                state["critiques"] = {}
            state["evidence_digest"] = digest
            state["results_available"] = bool(ready)
            state["comparable"] = bool(ready and comparable)

    def critique(self, agent: str, summary: str, recommend_followup: bool) -> dict:
        if not summary.strip():
            raise ValueError("A public critique summary is required")
        with self._transaction() as state:
            if agent not in self.policy["critique_agents"]:
                raise ValueError("This agent is not a required campaign critic")
            if not state["results_available"]:
                raise ValueError(
                    "Read the completed initial campaign results before recording critiques"
                )
            state["critiques"][agent] = {
                "summary": summary.strip()[:4000],
                "recommend_followup": recommend_followup,
            }
            released = (
                len(state["runs"]) < self.policy["limit"]
                and state["comparable"]
                and all(
                    state["critiques"].get(name, {}).get("recommend_followup")
                    for name in self.policy["critique_agents"]
                )
            )
            return {
                "agent": agent,
                "critique_recorded": True,
                "followup_released": bool(released),
            }


def main():
    import argparse

    from mada.core.config import load_orchestration_config

    parser = argparse.ArgumentParser(
        description="Inspect a campaign budget or import known submitted runs; never submits jobs"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--import-submitted",
        type=Path,
        help="Manifest of runs verified to have already been submitted",
    )
    args = parser.parse_args()
    config = load_orchestration_config(
        json.loads(args.config.read_text())["orchestration"]
    )
    config.resolve_campaign_paths(args.config.parent)
    policy = config.simulation_budget
    if not policy or not policy.get("campaign_id"):
        parser.error(
            "The configuration must define a persistent simulation_budget with campaign_id"
        )
    ledger = CampaignLedger(policy["ledger_path"], policy["campaign_id"], policy)
    if args.import_submitted:
        manifest = json.loads(args.import_submitted.read_text())
        ledger.import_submitted(
            manifest.get("runs") if isinstance(manifest, dict) else manifest
        )
    state = ledger.snapshot()
    print(
        json.dumps(
            {
                "campaign_id": policy["campaign_id"],
                "reserved_simulations": len(state["runs"]),
                "limit": policy["limit"],
                "critiques": state["critiques"],
                "new_simulations_submitted": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
