"""Metrics and internal-ledger diagnostics for ``state_audit`` outputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .io_utils import read_jsonl, write_json
from .schema import ROOT
from .state_audit import CLOSING_EVENTS, FIELD_DOMAIN, STATE_DOMAINS, _overlap


DOMAIN_WEIGHTS = {
    "WORLD_FACT": 0.18,
    "CHARACTER": 0.16,
    "KNOWLEDGE": 0.14,
    "RESOURCE_EVIDENCE": 0.12,
    "GOAL": 0.10,
    "UNKNOWN": 0.10,
    "TIMELINE": 0.08,
    "RELATIONSHIP": 0.07,
    "CAUSAL_THREAD": 0.05,
}
SEVERITY_WEIGHT = {"MINOR": 0.5, "MAJOR": 1.0, "CRITICAL": 2.0}


def _episode_number(episode_id: str) -> int:
    return int(str(episode_id)[1:])


def compute_trajectory_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    states = summary.get("states", [])
    issues = summary.get("confirmed_issues", [])
    episode_count = int(summary.get("episode_count", 40))
    domain_activity: dict[str, set[str]] = defaultdict(set)
    domain_issue_episodes: dict[str, set[str]] = defaultdict(set)
    issue_types: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    issue_episodes: set[str] = set()
    transition_count = 0

    for state in states:
        domain = str(state.get("domain", ""))
        introduced = str(state.get("introduced_episode", ""))
        if domain in STATE_DOMAINS and introduced.startswith("E") and introduced != "E00":
            domain_activity[domain].add(introduced)
        for event in state.get("history", []):
            episode_id = str(event.get("episode_id", ""))
            if episode_id.startswith("E") and episode_id != "E00":
                domain_activity[domain].add(episode_id)
                transition_count += 1

    for issue in issues:
        episode_id = str(issue.get("episode_id", ""))
        if episode_id.startswith("E"):
            issue_episodes.add(episode_id)
        issue_types[str(issue.get("issue_type", "UNKNOWN"))] += 1
        severity_counts[str(issue.get("severity", "MAJOR"))] += 1
        for domain in issue.get("domains", []):
            if domain in STATE_DOMAINS and episode_id.startswith("E"):
                domain_issue_episodes[domain].add(episode_id)

    final_statuses = summary.get("final_open_statuses", {})
    stale_ids = {state_id for state_id, value in final_statuses.items() if value.get("status") == "STALE"}
    partial_ids = {state_id for state_id, value in final_statuses.items() if value.get("status") == "PARTIAL"}
    if stale_ids:
        issue_episodes.add(f"E{episode_count:02d}")
    state_index = {item.get("state_id"): item for item in states}
    for state_id in stale_ids:
        domain = state_index.get(state_id, {}).get("domain")
        if domain in STATE_DOMAINS:
            domain_activity[domain].add(f"E{episode_count:02d}")
            domain_issue_episodes[domain].add(f"E{episode_count:02d}")

    domain_scores: dict[str, float | None] = {}
    available_weight = 0.0
    weighted_sum = 0.0
    for domain, weight in DOMAIN_WEIGHTS.items():
        relevant = domain_activity.get(domain, set()) | domain_issue_episodes.get(domain, set())
        if not relevant:
            domain_scores[domain] = None
            continue
        failed = domain_issue_episodes.get(domain, set()) & relevant
        score = 100.0 * (1.0 - len(failed) / len(relevant))
        domain_scores[domain] = round(score, 4)
        weighted_sum += weight * score
        available_weight += weight

    tscs = weighted_sum / available_weight if available_weight else None
    burden = sum(SEVERITY_WEIGHT.get(str(item.get("severity", "MAJOR")), 1.0) for item in issues)
    burden += len(stale_ids)
    non_initial_states = [item for item in states if item.get("introduced_episode") != "E00"]
    prospective_open_count = len(final_statuses)
    return {
        "trajectory_id": summary["trajectory_id"],
        "story_id": summary["story_id"],
        "run_id": summary["run_id"],
        "condition": summary.get("condition"),
        "episode_count": episode_count,
        "trajectory_state_continuity_score": round(tscs, 4) if tscs is not None else None,
        "domain_scores": domain_scores,
        "issue_free_episode_rate": round(1.0 - len(issue_episodes) / episode_count, 6) if episode_count else None,
        "confirmed_issue_count": len(issues),
        "confirmed_issue_rate_per_10ep": round(len(issues) * 10 / episode_count, 6) if episode_count else None,
        "severity_weighted_issue_burden_per_10ep": round(burden * 10 / episode_count, 6) if episode_count else None,
        "major_or_critical_issue_count": severity_counts["MAJOR"] + severity_counts["CRITICAL"],
        "critical_issue_count": severity_counts["CRITICAL"],
        "external_state_count": len(non_initial_states),
        "initial_state_count": len(states) - len(non_initial_states),
        "transition_count": transition_count,
        "prospective_open_count": prospective_open_count,
        "stale_open_count": len(stale_ids),
        "stale_open_rate": round(len(stale_ids) / prospective_open_count, 6) if prospective_open_count else None,
        "partial_open_count": len(partial_ids),
        "issue_type_counts": dict(issue_types),
        "severity_counts": dict(severity_counts),
    }


def _flatten_state_value(value: Any, *, subject: str = "STORY") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(subject, value)] if value.strip() else []
    if isinstance(value, list):
        rows: list[tuple[str, str]] = []
        for item in value:
            rows.extend(_flatten_state_value(item, subject=subject))
        return rows
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            rows.extend(_flatten_state_value(item, subject=str(key)))
        return rows
    return []


def _internal_additions(updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for update in updates:
        delta = update.get("state_delta", {})
        for field, domain in FIELD_DOMAIN.items():
            for subject, text in _flatten_state_value(delta.get(field, [])):
                rows.append({
                    "domain": domain,
                    "episode_id": update["episode_id"],
                    "text": f"{subject}：{text}" if subject != "STORY" else text,
                    "field": field,
                })
    return rows


def _internal_removals(updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mapping = {
        "resolved_goals": "GOAL",
        "resolved_unknown": "UNKNOWN",
        "retracted_facts": "WORLD_FACT",
    }
    for update in updates:
        removed = update.get("lifecycle_removed", {})
        for field, domain in mapping.items():
            for text in removed.get(field, []) or []:
                rows.append({"domain": domain, "episode_id": update["episode_id"], "text": str(text), "field": field})
    return rows


def _external_additions(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "state_id": item["state_id"],
            "domain": item["domain"],
            "episode_id": item["introduced_episode"],
            "text": item.get("description") or item.get("value", ""),
        }
        for item in summary.get("states", [])
        if item.get("introduced_episode") != "E00"
    ]


def _external_closures(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in summary.get("states", []):
        for event in state.get("history", []):
            if event.get("event") in CLOSING_EVENTS:
                rows.append({
                    "state_id": state["state_id"],
                    "domain": state["domain"],
                    "episode_id": event["episode_id"],
                    "text": state.get("description") or state.get("value", ""),
                    "event": event["event"],
                })
    return rows


def _domains_compatible(left: str, right: str) -> bool:
    if left == right:
        return True
    return {left, right} <= {"WORLD_FACT", "KNOWLEDGE"}


def _greedy_matches(external: list[dict[str, Any]], internal: list[dict[str, Any]], threshold: float) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for ext_index, ext in enumerate(external):
        for int_index, item in enumerate(internal):
            if not _domains_compatible(str(ext["domain"]), str(item["domain"])):
                continue
            score = _overlap(str(ext["text"]), str(item["text"]))
            if score >= threshold:
                candidates.append((score, ext_index, int_index))
    candidates.sort(reverse=True)
    ext_used: set[int] = set()
    int_used: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, ext_index, int_index in candidates:
        if ext_index in ext_used or int_index in int_used:
            continue
        ext_used.add(ext_index)
        int_used.add(int_index)
        matches.append((ext_index, int_index, score))
    return matches


def internal_ledger_diagnostics(
    *, summaries: list[dict[str, Any]], internal_updates_path: Path, runs_filter: set[str] | None
) -> dict[str, Any]:
    updates = read_jsonl(internal_updates_path)
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in updates:
        run_id = str(row.get("trajectory_id", "")).split("__")[-1]
        if runs_filter and run_id not in runs_filter:
            continue
        by_trajectory[str(row["trajectory_id"])].append(row)
    summary_index = {str(item["trajectory_id"]): item for item in summaries}
    totals: Counter[str] = Counter()
    write_scores: list[float] = []
    removal_scores: list[float] = []
    trajectory_rows: list[dict[str, Any]] = []

    for trajectory_id, rows in sorted(by_trajectory.items()):
        summary = summary_index.get(trajectory_id)
        if not summary:
            continue
        rows.sort(key=lambda item: _episode_number(item["episode_id"]))
        external_add = _external_additions(summary)
        internal_add = _internal_additions(rows)
        write_matches = _greedy_matches(external_add, internal_add, threshold=0.18)
        write_scores.extend(score for _, _, score in write_matches)

        external_close = _external_closures(summary)
        internal_remove = _internal_removals(rows)
        removal_matches = _greedy_matches(external_close, internal_remove, threshold=0.18)
        removal_scores.extend(score for _, _, score in removal_matches)
        exact = early = late = 0
        for ext_index, int_index, _ in removal_matches:
            external_ep = _episode_number(external_close[ext_index]["episode_id"])
            internal_ep = _episode_number(internal_remove[int_index]["episode_id"])
            if internal_ep == external_ep:
                exact += 1
            elif internal_ep < external_ep:
                early += 1
            else:
                late += 1
        unmatched_internal = len(internal_remove) - len(removal_matches)
        unmatched_external = len(external_close) - len(removal_matches)
        premature = early + unmatched_internal
        stale = late + unmatched_external
        totals.update({
            "external_add": len(external_add),
            "internal_add": len(internal_add),
            "write_match": len(write_matches),
            "external_close": len(external_close),
            "internal_remove": len(internal_remove),
            "removal_match": len(removal_matches),
            "exact_removal": exact,
            "early_removal": early,
            "late_removal": late,
            "premature": premature,
            "stale": stale,
        })
        trajectory_rows.append({
            "trajectory_id": trajectory_id,
            "external_state_count": len(external_add),
            "internal_state_count": len(internal_add),
            "write_match_count": len(write_matches),
            "external_closure_count": len(external_close),
            "internal_removal_count": len(internal_remove),
            "exact_removal_count": exact,
            "premature_removal_count": premature,
            "stale_internal_count": stale,
        })

    return {
        "state_write_precision": round(totals["write_match"] / totals["internal_add"], 6) if totals["internal_add"] else None,
        "state_write_recall": round(totals["write_match"] / totals["external_add"], 6) if totals["external_add"] else None,
        "state_removal_precision": round(totals["exact_removal"] / totals["internal_remove"], 6) if totals["internal_remove"] else None,
        "state_removal_recall": round(totals["exact_removal"] / totals["external_close"], 6) if totals["external_close"] else None,
        "premature_removal_rate": round(totals["premature"] / totals["internal_remove"], 6) if totals["internal_remove"] else None,
        "stale_internal_rate": round(totals["stale"] / totals["external_close"], 6) if totals["external_close"] else None,
        "mean_write_match_score": round(mean(write_scores), 6) if write_scores else None,
        "mean_removal_match_score": round(mean(removal_scores), 6) if removal_scores else None,
        "counts": dict(totals),
        "trajectories": trajectory_rows,
        "matching_note": "description/value 2-gram 分域贪心匹配，只作为内部账本机制诊断，不作为轨迹状态连续性主分",
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_fields = [
        "trajectory_state_continuity_score",
        "issue_free_episode_rate",
        "confirmed_issue_rate_per_10ep",
        "severity_weighted_issue_burden_per_10ep",
        "stale_open_rate",
    ]
    result: dict[str, Any] = {"trajectory_count": len(rows)}
    for field in scalar_fields:
        values = [float(item[field]) for item in rows if item.get(field) is not None]
        result[field] = round(mean(values), 6) if values else None
    for domain in DOMAIN_WEIGHTS:
        values = [float(item["domain_scores"][domain]) for item in rows if item["domain_scores"].get(domain) is not None]
        result[f"domain_{domain.lower()}"] = round(mean(values), 6) if values else None
    result["total_confirmed_issues"] = sum(int(item["confirmed_issue_count"]) for item in rows)
    result["total_external_states"] = sum(int(item["external_state_count"]) for item in rows)
    return result


def analyze(
    *, audit_dir: Path, output: Path, internal_updates_path: Path | None, runs_filter: set[str] | None
) -> dict[str, Any]:
    summaries = read_jsonl(audit_dir / "state_audit_trajectories.jsonl")
    if runs_filter:
        summaries = [item for item in summaries if item.get("run_id") in runs_filter]
    metrics = [compute_trajectory_metrics(item) for item in summaries]
    result: dict[str, Any] = {
        "metric": "Trajectory State Continuity Audit",
        "abbreviation": "TSCS",
        "score_definition": "受评状态域中，无确认问题的活跃集比例按 DOMAIN_WEIGHTS 加权；无活动域不进入分母",
        "domain_weights": DOMAIN_WEIGHTS,
        "runs": sorted(runs_filter) if runs_filter else None,
        "aggregate": _aggregate(metrics),
        "trajectories": metrics,
    }
    if internal_updates_path:
        result["internal_ledger_diagnostics"] = internal_ledger_diagnostics(
            summaries=summaries,
            internal_updates_path=internal_updates_path,
            runs_filter=runs_filter,
        )
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--internal-updates", type=Path)
    parser.add_argument("--runs", type=str)
    args = parser.parse_args()
    runs_filter = {item.strip() for item in args.runs.split(",") if item.strip()} if args.runs else None
    result = analyze(
        audit_dir=args.audit_dir,
        output=args.output,
        internal_updates_path=args.internal_updates,
        runs_filter=runs_filter,
    )
    aggregate = result["aggregate"]
    print(
        f"state_trajectories={aggregate['trajectory_count']} "
        f"TSCS={aggregate['trajectory_state_continuity_score']} "
        f"issues={aggregate['total_confirmed_issues']} output={args.output}"
    )


if __name__ == "__main__":
    main()
