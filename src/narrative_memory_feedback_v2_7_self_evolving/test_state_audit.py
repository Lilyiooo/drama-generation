from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .analyze_state_continuity import compute_trajectory_metrics, internal_ledger_diagnostics
from .state_audit import validate_episode_audit, validate_finalize, validate_verification


class StateAuditValidationTests(unittest.TestCase):
    def test_episode_validation_cleans_all_sections(self) -> None:
        value = validate_episode_audit({
            "new_states": [{
                "domain": "knowledge", "subject": "甲", "attribute": "knows_answer",
                "value": "知道门已锁", "description": "甲已经知道门被锁住",
                "certainty": "confirmed", "importance": 2, "evidence": "甲说「门锁了」",
            }],
            "state_events": [{
                "state_id": "ST0001", "event": "updated", "new_value": "停职",
                "evidence": "通知书写明停职", "cause_state_ids": [],
            }],
            "candidate_issues": [{
                "issue_type": "knowledge_leak", "domains": ["knowledge"],
                "state_ids": ["ST0001"], "description": "乙使用了未获得的信息",
                "evidence": "乙直接说出密码", "severity": "major", "confidence": 0.8,
            }],
        }, active_ids={"ST0001"})
        self.assertEqual(value["new_states"][0]["domain"], "KNOWLEDGE")
        self.assertEqual(value["state_events"][0]["event"], "UPDATED")
        self.assertEqual(value["candidate_issues"][0]["candidate_id"], "C01")

    def test_episode_validation_rejects_unknown_state_id(self) -> None:
        with self.assertRaises(ValueError):
            validate_episode_audit({
                "new_states": [],
                "state_events": [{
                    "state_id": "ST9999", "event": "RESOLVED", "new_value": "",
                    "evidence": "已经完成", "cause_state_ids": [],
                }],
                "candidate_issues": [],
            }, active_ids={"ST0001"})

    def test_verification_and_finalize_require_complete_id_coverage(self) -> None:
        decisions = validate_verification({
            "decisions": [{"candidate_id": "C01", "verdict": "CONFIRMED", "evidence": "冲突", "confidence": 1}],
        }, {"C01"})
        self.assertEqual(decisions["C01"]["verdict"], "CONFIRMED")
        statuses = validate_finalize({
            "open_state_statuses": [{"state_id": "ST0002", "status": "STALE", "reason": "已经完成", "confidence": 1}],
        }, {"ST0002"})
        self.assertEqual(statuses["ST0002"]["status"], "STALE")


class StateContinuityMetricTests(unittest.TestCase):
    def test_metrics_penalize_confirmed_issue_and_stale_open(self) -> None:
        summary = {
            "trajectory_id": "S01__C__R01", "story_id": "S01", "run_id": "R01",
            "condition": "C", "episode_count": 40,
            "states": [
                {"state_id": "ST0001", "domain": "WORLD_FACT", "introduced_episode": "E01", "history": []},
                {"state_id": "ST0002", "domain": "GOAL", "introduced_episode": "E02", "history": []},
            ],
            "confirmed_issues": [{
                "episode_id": "E03", "issue_type": "FACT_CONTRADICTION",
                "severity": "CRITICAL", "domains": ["WORLD_FACT"],
            }],
            "final_open_statuses": {"ST0002": {"status": "STALE"}},
        }
        metrics = compute_trajectory_metrics(summary)
        self.assertLess(metrics["trajectory_state_continuity_score"], 100)
        self.assertEqual(metrics["critical_issue_count"], 1)
        self.assertEqual(metrics["stale_open_rate"], 1.0)

    def test_internal_ledger_diagnostics_match_write_and_removal(self) -> None:
        summary = {
            "trajectory_id": "S01__C__R01", "story_id": "S01", "run_id": "R01",
            "states": [{
                "state_id": "ST0001", "domain": "GOAL", "introduced_episode": "E01",
                "description": "甲需要找到钥匙", "value": "找到钥匙",
                "history": [{"episode_id": "E02", "event": "RESOLVED"}],
            }],
        }
        update_rows = [
            {
                "trajectory_id": "S01__C__R01", "episode_id": "E01",
                "state_delta": {
                    "character_state": [], "relationship_state": [], "known_information": [],
                    "unknown_information": [], "confirmed_facts": [], "unresolved_threads": [],
                    "resources_and_evidence": [], "current_goals": ["甲需要找到钥匙"], "timeline": [],
                },
                "lifecycle_removed": {"resolved_goals": [], "resolved_unknown": [], "retracted_facts": []},
            },
            {
                "trajectory_id": "S01__C__R01", "episode_id": "E02",
                "state_delta": {field: [] for field in (
                    "character_state", "relationship_state", "known_information", "unknown_information",
                    "confirmed_facts", "unresolved_threads", "resources_and_evidence", "current_goals", "timeline",
                )},
                "lifecycle_removed": {"resolved_goals": ["甲需要找到钥匙"], "resolved_unknown": [], "retracted_facts": []},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state_updates.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in update_rows), encoding="utf-8")
            result = internal_ledger_diagnostics(
                summaries=[summary], internal_updates_path=path, runs_filter=None,
            )
        self.assertEqual(result["state_write_precision"], 1.0)
        self.assertEqual(result["state_removal_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
