from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .io_utils import read_jsonl, write_json

CLOSURE_VALUES = {
    "FULFILLED": 1.0,
    "FAILED_WITH_CONSEQUENCE": 1.0,
    "ABANDONED_WITH_CONSEQUENCE": 1.0,
    "PARTIAL": 0.5,
    "DROPPED": 0.0,
    "CONTRADICTED": 0.0,
}
RESPONSE_VALUES = {
    "FULFILLED": 1.0,
    "FAILED_WITH_CONSEQUENCE": 1.0,
    "ABANDONED_WITH_CONSEQUENCE": 1.0,
    "PROGRESSED": 0.75,
    "MENTIONED": 0.20,
    "NONE": 0.0,
    "CONTRADICTED": 0.0,
}


def _bigrams(text: str) -> set[str]:
    clean = "".join(str(text).split())
    return {clean[i:i + 2] for i in range(max(0, len(clean) - 1))}


def _overlap(left: str, right: str) -> float:
    a, b = _bigrams(left), _bigrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _terminal_history_event(history_item: dict[str, Any]) -> str:
    verification = history_item.get("verification")
    if isinstance(verification, dict) and verification.get("verdict") == "CONFIRMED":
        return str(verification.get("terminal_event", "NONE"))
    event = str(history_item.get("event", "NONE"))
    return event if event in RESPONSE_VALUES else "NONE"


def _weighted_mean(pairs: list[tuple[float, float]]) -> float | None:
    denominator = sum(weight for _, weight in pairs)
    if denominator <= 0:
        return None
    return sum(value * weight for value, weight in pairs) / denominator


def compute_trajectory_metrics(summary: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    obligations = summary.get("obligations", [])
    closure_pairs: list[tuple[float, float]] = []
    response_pairs: list[tuple[float, float]] = []
    consistency_pairs: list[tuple[float, float]] = []
    distances: list[int] = []
    distance_bucket_closed = {"short_1_3": 0, "medium_4_8": 0, "long_9_plus": 0}
    distance_bucket_matured = {"short_1_3": 0, "medium_4_8": 0, "long_9_plus": 0}
    type_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    importance_counts: Counter[int] = Counter()

    for obligation in obligations:
        importance = float(obligation.get("importance", 1))
        status = str(obligation.get("status", "OPEN_VALID"))
        type_counts[str(obligation.get("type", "UNKNOWN"))] += 1
        status_counts[status] += 1
        importance_counts[int(importance)] += 1
        if status in CLOSURE_VALUES:
            closure_pairs.append((CLOSURE_VALUES[status], importance))
        history = obligation.get("history", [])
        for event in history:
            if event.get("relevance") not in {"OPPORTUNITY", "DUE"}:
                continue
            event_name = _terminal_history_event(event)
            response_pairs.append((RESPONSE_VALUES.get(event_name, 0.0), importance))

        drift_values = [str(event.get("drift", "NONE")) for event in history]
        if status == "CONTRADICTED" or "MAJOR" in drift_values:
            consistency = 0.0
        elif "MINOR" in drift_values:
            consistency = 0.75
        else:
            consistency = 1.0
        consistency_pairs.append((consistency, importance))

        resolution_episode = obligation.get("resolution_episode")
        introduced_episode = obligation.get("introduced_episode")
        terminal_status = status in {"FULFILLED", "FAILED_WITH_CONSEQUENCE", "ABANDONED_WITH_CONSEQUENCE"}
        if resolution_episode and introduced_episode:
            distance = int(resolution_episode[1:]) - int(introduced_episode[1:])
            distances.append(distance)
            bucket = "short_1_3" if distance <= 3 else ("medium_4_8" if distance <= 8 else "long_9_plus")
            distance_bucket_matured[bucket] += 1
            if terminal_status:
                distance_bucket_closed[bucket] += 1
        elif status in {"DROPPED", "PARTIAL", "CONTRADICTED"} and introduced_episode:
            observed_distance = 40 - int(introduced_episode[1:])
            bucket = "short_1_3" if observed_distance <= 3 else ("medium_4_8" if observed_distance <= 8 else "long_9_plus")
            distance_bucket_matured[bucket] += 1

    closure = _weighted_mean(closure_pairs)
    response = _weighted_mean(response_pairs)
    consistency = _weighted_mean(consistency_pairs)
    components = [
        (0.5, closure),
        (0.3, response),
        (0.2, consistency),
    ]
    available_weight = sum(weight for weight, value in components if value is not None)
    ocs = None
    # 至少有成熟义务或相关兑现机会才能计算 OCS；纯开放/右删失轨迹不可评价，不能仅凭一致性得到满分。
    if (closure is not None or response is not None) and available_weight > 0:
        ocs = 100.0 * sum(weight * value for weight, value in components if value is not None) / available_weight
        ocs = max(0.0, min(100.0, ocs))

    open_counts = [int(row.get("open_count", 0)) for row in episodes]
    episode_count = int(summary.get("episode_count", len(episodes) or 40))
    material_count = sum(1 for item in obligations if int(item.get("importance", 1)) >= 2)
    matured_count = sum(status_counts[x] for x in CLOSURE_VALUES)
    terminal_count = sum(status_counts[x] for x in ("FULFILLED", "FAILED_WITH_CONSEQUENCE", "ABANDONED_WITH_CONSEQUENCE"))
    opportunity_count = len(response_pairs)
    response_success = sum(1 for value, _ in response_pairs if value >= 0.75)

    return {
        "trajectory_id": summary["trajectory_id"],
        "story_id": summary["story_id"],
        "run_id": summary["run_id"],
        "condition": summary.get("condition"),
        "episode_count": episode_count,
        "obligation_count": len(obligations),
        "ocs": round(ocs, 4) if ocs is not None else None,
        "valid_payoff_rate": round(closure, 6) if closure is not None else None,
        "opportunity_response_rate": round(response, 6) if response is not None else None,
        "consistency_rate": round(consistency, 6) if consistency is not None else None,
        "matured_count": matured_count,
        "matured_rate": round(matured_count / len(obligations), 6) if obligations else None,
        "terminal_closure_count": terminal_count,
        "dropped_rate": round(status_counts["DROPPED"] / matured_count, 6) if matured_count else None,
        "contradiction_rate": round(status_counts["CONTRADICTED"] / len(obligations), 6) if obligations else None,
        "opportunity_count": opportunity_count,
        "opportunity_success_rate": round(response_success / opportunity_count, 6) if opportunity_count else None,
        "mean_payoff_distance": round(mean(distances), 6) if distances else None,
        "distance_bucket_closed": distance_bucket_closed,
        "distance_bucket_matured": distance_bucket_matured,
        "short_payoff_rate": round(distance_bucket_closed["short_1_3"] / distance_bucket_matured["short_1_3"], 6) if distance_bucket_matured["short_1_3"] else None,
        "medium_payoff_rate": round(distance_bucket_closed["medium_4_8"] / distance_bucket_matured["medium_4_8"], 6) if distance_bucket_matured["medium_4_8"] else None,
        "long_range_payoff_count": distance_bucket_closed["long_9_plus"],
        "long_range_matured_count": distance_bucket_matured["long_9_plus"],
        "long_range_payoff_rate": round(distance_bucket_closed["long_9_plus"] / distance_bucket_matured["long_9_plus"], 6) if distance_bucket_matured["long_9_plus"] else None,
        "open_valid_rate": round(status_counts["OPEN_VALID"] / len(obligations), 6) if obligations else None,
        "censored_rate": round(status_counts["CENSORED"] / len(obligations), 6) if obligations else None,
        "material_obligation_density_per_10ep": round(material_count * 10 / episode_count, 6),
        "obligation_type_count": len(type_counts),
        "type_counts": dict(type_counts),
        "status_counts": dict(status_counts),
        "importance_counts": {str(k): v for k, v in importance_counts.items()},
        "mean_open_load": round(mean(open_counts), 6) if open_counts else None,
        "peak_open_load": max(open_counts) if open_counts else None,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_fields = [
        "ocs", "valid_payoff_rate", "opportunity_response_rate", "consistency_rate",
        "matured_rate", "dropped_rate", "contradiction_rate", "opportunity_success_rate",
        "mean_payoff_distance", "short_payoff_rate", "medium_payoff_rate", "long_range_payoff_rate", "open_valid_rate", "censored_rate",
        "material_obligation_density_per_10ep", "obligation_type_count", "mean_open_load", "peak_open_load",
    ]
    result: dict[str, Any] = {"trajectory_count": len(rows)}
    for field in metric_fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        result[field] = round(mean(values), 6) if values else None
    result["total_obligations"] = sum(int(row["obligation_count"]) for row in rows)
    result["total_matured"] = sum(int(row["matured_count"]) for row in rows)
    return result


def _paired_permutation(differences: list[float], *, iterations: int = 20000, seed: int = 20260819) -> float | None:
    if not differences:
        return None
    observed = abs(mean(differences))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        permuted = mean(value if rng.random() < 0.5 else -value for value in differences)
        if abs(permuted) >= observed:
            extreme += 1
    return round((extreme + 1) / (iterations + 1), 6)


def _story_stratified_bootstrap(
    paired: list[tuple[str, float]],
    *,
    iterations: int = 10000,
    seed: int = 20260819,
) -> list[float] | None:
    by_story: dict[str, list[float]] = defaultdict(list)
    for story_id, value in paired:
        by_story[story_id].append(value)
    if not by_story:
        return None
    rng = random.Random(seed)
    samples: list[float] = []
    stories = sorted(by_story)
    for _ in range(iterations):
        values: list[float] = []
        for story_id in stories:
            source = by_story[story_id]
            values.extend(rng.choice(source) for _ in range(len(source)))
        samples.append(mean(values))
    samples.sort()
    return [round(samples[int(0.025 * (len(samples) - 1))], 6), round(samples[int(0.975 * (len(samples) - 1))], 6)]


def compare_conditions(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    left_index = {(row["story_id"], row["run_id"]): row for row in left}
    right_index = {(row["story_id"], row["run_id"]): row for row in right}
    common = sorted(set(left_index) & set(right_index))
    fields = [
        "ocs", "valid_payoff_rate", "opportunity_response_rate", "consistency_rate",
        "dropped_rate", "contradiction_rate", "long_range_payoff_rate",
        "material_obligation_density_per_10ep", "mean_open_load",
    ]
    comparisons: dict[str, Any] = {}
    for field in fields:
        pairs: list[tuple[str, float]] = []
        for key in common:
            left_value = left_index[key].get(field)
            right_value = right_index[key].get(field)
            if left_value is None or right_value is None:
                continue
            pairs.append((key[0], float(left_value) - float(right_value)))
        differences = [value for _, value in pairs]
        comparisons[field] = {
            "paired_n": len(differences),
            "delta": round(mean(differences), 6) if differences else None,
            "bootstrap_95ci": _story_stratified_bootstrap(pairs),
            "paired_permutation_p": _paired_permutation(differences),
        }
    return {
        "left": left_name,
        "right": right_name,
        "paired_trajectory_count": len(common),
        "matched_keys": [{"story_id": story, "run_id": run} for story, run in common],
        "metrics": comparisons,
    }


def _match_internal_to_external(
    external: list[dict[str, Any]],
    internal: list[dict[str, Any]],
) -> tuple[dict[str, str], list[tuple[str, str, float]]]:
    candidates: list[tuple[float, str, str]] = []
    for ext in external:
        for item in internal:
            description_score = _overlap(ext.get("description", ""), item.get("description", ""))
            payoff_score = _overlap(ext.get("required_payoff", ""), item.get("required_payoff", ""))
            score = 0.7 * description_score + 0.3 * payoff_score
            if score >= 0.28:
                candidates.append((score, ext["obligation_id"], item["obligation_id"]))
    candidates.sort(reverse=True)
    ext_used: set[str] = set()
    int_used: set[str] = set()
    mapping: dict[str, str] = {}
    matched: list[tuple[str, str, float]] = []
    for score, ext_id, int_id in candidates:
        if ext_id in ext_used or int_id in int_used:
            continue
        ext_used.add(ext_id)
        int_used.add(int_id)
        mapping[int_id] = ext_id
        matched.append((int_id, ext_id, score))
    return mapping, matched


def internal_ledger_diagnostics(
    *,
    audit_summaries: list[dict[str, Any]],
    internal_updates_path: Path,
    runs_filter: set[str] | None,
) -> dict[str, Any]:
    updates = read_jsonl(internal_updates_path)
    by_trajectory: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in updates:
        run_id = str(row["trajectory_id"]).split("__")[-1]
        if runs_filter and run_id not in runs_filter:
            continue
        by_trajectory[(row["story_id"], run_id)].append(row)
    summary_index = {(row["story_id"], row["run_id"]): row for row in audit_summaries}
    totals = Counter()
    match_scores: list[float] = []
    trajectory_rows: list[dict[str, Any]] = []

    for key, rows in sorted(by_trajectory.items()):
        if key not in summary_index:
            continue
        rows.sort(key=lambda row: int(row["episode_id"][1:]))
        internal_added: list[dict[str, Any]] = []
        internal_resolved: list[dict[str, Any]] = []
        for row in rows:
            internal_added.extend(row.get("added", []))
            for item in row.get("resolved", []):
                internal_resolved.append({**item, "audit_resolution_episode": row["episode_id"]})
        external = summary_index[key].get("obligations", [])
        mapping, matches = _match_internal_to_external(external, internal_added)
        external_index = {item["obligation_id"]: item for item in external}
        true_resolution = 0
        premature_resolution = 0
        for item in internal_resolved:
            ext_id = mapping.get(item["obligation_id"])
            if not ext_id:
                premature_resolution += 1
                continue
            ext = external_index[ext_id]
            if ext.get("resolution_episode") == item["audit_resolution_episode"] and ext.get("status") in {
                "FULFILLED", "FAILED_WITH_CONSEQUENCE", "ABANDONED_WITH_CONSEQUENCE",
            }:
                true_resolution += 1
            else:
                premature_resolution += 1
        matched_external = {ext_id for _, ext_id, _ in matches}
        matched_internal = {int_id for int_id, _, _ in matches}
        ext_terminal = [item for item in external if item.get("status") in {
            "FULFILLED", "FAILED_WITH_CONSEQUENCE", "ABANDONED_WITH_CONSEQUENCE",
        }]
        resolved_external_ids = {
            mapping[item["obligation_id"]]
            for item in internal_resolved
            if item["obligation_id"] in mapping
        }
        stale_open = sum(1 for item in ext_terminal if item["obligation_id"] not in resolved_external_ids)
        totals.update({
            "external": len(external),
            "internal": len(internal_added),
            "matched_external": len(matched_external),
            "matched_internal": len(matched_internal),
            "internal_resolved": len(internal_resolved),
            "true_resolution": true_resolution,
            "premature_resolution": premature_resolution,
            "external_terminal": len(ext_terminal),
            "stale_open": stale_open,
        })
        match_scores.extend(score for _, _, score in matches)
        trajectory_rows.append({
            "story_id": key[0],
            "run_id": key[1],
            "external_count": len(external),
            "internal_count": len(internal_added),
            "matched_count": len(matches),
            "premature_resolution_count": premature_resolution,
            "stale_open_count": stale_open,
        })

    return {
        "write_precision": round(totals["matched_internal"] / totals["internal"], 6) if totals["internal"] else None,
        "write_recall": round(totals["matched_external"] / totals["external"], 6) if totals["external"] else None,
        "resolution_precision": round(totals["true_resolution"] / totals["internal_resolved"], 6) if totals["internal_resolved"] else None,
        "resolution_recall": round(totals["true_resolution"] / totals["external_terminal"], 6) if totals["external_terminal"] else None,
        "premature_resolve_rate": round(totals["premature_resolution"] / totals["internal_resolved"], 6) if totals["internal_resolved"] else None,
        "stale_open_rate": round(totals["stale_open"] / totals["external_terminal"], 6) if totals["external_terminal"] else None,
        "mean_match_score": round(mean(match_scores), 6) if match_scores else None,
        "counts": dict(totals),
        "trajectories": trajectory_rows,
        "matching_note": "基于 description/payoff 2-gram 贪心匹配，仅作机制诊断，不作为 OCS 主指标",
    }


def load_metrics(audit_dir: Path, runs_filter: set[str] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = read_jsonl(audit_dir / "obligation_audit_trajectories.jsonl")
    episodes = read_jsonl(audit_dir / "obligation_audit_episodes.jsonl")
    if runs_filter:
        summaries = [row for row in summaries if row["run_id"] in runs_filter]
        episodes = [row for row in episodes if row["run_id"] in runs_filter]
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        by_trajectory[row["trajectory_id"]].append(row)
    metrics = [compute_trajectory_metrics(row, by_trajectory.get(row["trajectory_id"], [])) for row in summaries]
    return metrics, summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-audit-dir", type=Path, required=True)
    parser.add_argument("--s3-audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=str, default="R02,R03")
    parser.add_argument("--s3-internal-updates", type=Path)
    args = parser.parse_args()
    runs_filter = {x.strip() for x in args.runs.split(",") if x.strip()} if args.runs else None
    s1_metrics, s1_summaries = load_metrics(args.s1_audit_dir, runs_filter)
    s3_metrics, s3_summaries = load_metrics(args.s3_audit_dir, runs_filter)
    result = {
        "metric": "Obligation Continuity Score",
        "abbreviation": "OCS",
        "formula": "100 * (0.5*C + 0.3*R + 0.2*S), renormalized when a component is unavailable",
        "runs": sorted(runs_filter) if runs_filter else None,
        "S1_lifecycle": {"aggregate": _aggregate(s1_metrics), "trajectories": s1_metrics},
        "S3_obligation": {"aggregate": _aggregate(s3_metrics), "trajectories": s3_metrics},
        "comparison": compare_conditions(s3_metrics, s1_metrics, left_name="S3_obligation", right_name="S1_lifecycle"),
    }
    if args.s3_internal_updates:
        result["S3_internal_ledger_diagnostics"] = internal_ledger_diagnostics(
            audit_summaries=s3_summaries,
            internal_updates_path=args.s3_internal_updates,
            runs_filter=runs_filter,
        )
    write_json(args.output, result)
    comparison = result["comparison"]["metrics"]["ocs"]
    print(f"S1 trajectories={len(s1_metrics)} S3 trajectories={len(s3_metrics)}")
    print(f"OCS delta={comparison['delta']} 95%CI={comparison['bootstrap_95ci']} p={comparison['paired_permutation_p']}")
    print(f"summary_written={args.output}")


if __name__ == "__main__":
    main()
