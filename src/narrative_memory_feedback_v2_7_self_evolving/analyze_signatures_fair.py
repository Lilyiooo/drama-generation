"""位置公平的叙事重复分析（执行计划：选项1 card_echo + A 固定窗口评估）。

背景：标准 analyze_signatures 的 all_pair 指标存在「位置混淆」——
后续集（如 ep8）可比的前驱更多（1-7），仅凭「对历史的最大相似度」判定复用，
会随位置升高而假性膨胀（更多历史=更高随机命中）。本脚本提供两类修正：

1) card_echo（选项1，零噪声、位置公平）：每集只与「它实际检索到的卡」比对。
   - 零噪声版：本集 macro_move 是否命中其 retrieved_cluster_ids（cluster_id 即 move_id）。
   - 适用条件：凡有检索的组（C/D/E…）；A 组协议禁止检索 -> 返回 N/A。
2) 固定窗口（A 组等价修正）：epN 仅与「前 k 集」比对，所有位置窗口相同，
   消除历史池增长带来的分母偏差。

输出：
  runs/<dir>/narrative_signature_fair_v1_1.json
用法：
  python analyze_signatures_fair.py <run_dir> [--window 3] [--card-echo]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from statistics import mean

from narrative_signature import (
    SIGNATURE_SCHEMA_VERSION, effective_count, entropy, structured_similarity,
)

RDIR = Path(__file__).resolve().parent / "runs"


def _ep_index(ep: str) -> int:
    return int(ep[1:])


def _load_sigs(run_dir: Path) -> list[dict]:
    out = run_dir / "narrative_signatures_v1_1.jsonl"
    return [json.loads(l) for l in out.open(encoding="utf-8")]


def _load_contexts(run_dir: Path) -> dict[str, dict]:
    p = run_dir / "contexts.jsonl"
    if not p.exists():
        return {}
    return {r["job_id"]: r for r in (json.loads(l) for l in p.open(encoding="utf-8"))}


def _load_extractions(run_dir: Path) -> dict[str, dict]:
    p = run_dir / "extractions.jsonl"
    if not p.exists():
        return {}
    return {r["job_id"]: r for r in (json.loads(l) for l in p.open(encoding="utf-8"))}


def _core(left: dict, right: dict) -> float:
    return structured_similarity(left["signature"], right["signature"])["core_structural_similarity"]


def _dim_metrics(rows: list[dict]) -> dict:
    dims = ["macro_move", "information_delta", "turn_type", "resolution_type",
            "trigger_type", "obstacle_type", "state_change_target",
            "causal_relation", "hook_type", "interaction_pattern", "relationship_change"]
    out = {}
    for f in dims:
        vals = [r["signature"][f] for r in rows]
        out[f] = {
            "entropy": round(entropy(vals), 4),
            "effective_count": round(effective_count(vals), 4),
            "dominant_share": round(max(vals.count(v) for v in set(vals)) / len(vals), 4) if vals else 0.0,
        }
    return out


def analyze(run_dir: Path, window: int = 3, strict_window: bool = False) -> dict:
    rows = _load_sigs(run_dir)
    rows = sorted(rows, key=lambda r: (r["trajectory_id"], _ep_index(r["episode_id"])))
    trajs = defaultdict(list)
    for r in rows:
        trajs[r["trajectory_id"]].append(r)

    all_pairs, adjacent, fixed_max, fixed_mean = [], [], [], []
    all_hist_max, reuse_window, reuse_hist = [], [], []
    per_pos_window, per_pos_hist = defaultdict(list), defaultdict(list)

    for tid, eps in trajs.items():
        eps = sorted(eps, key=lambda r: _ep_index(r["episode_id"]))
        for i, r in enumerate(eps):
            hist = eps[:i]
            win = eps[max(0, i - window):i]
            all_core = [_core(r, h) for h in hist]
            win_core = [_core(r, h) for h in win]
            if all_core:
                all_pairs.append(statistics.mean(all_core))
                mh = max(all_core)
                all_hist_max.append(mh)
                reuse_hist.append(mh >= 0.55)
                per_pos_hist[i + 1].append(mh)
            # strict 模式：窗口未填满的开头几集（ep2/ep3）池子偏小、max 偏低，
            # 会引入与全历史相反方向的偏差，严格可比时应剔除。
            if strict_window and len(win) < window:
                win_core = []
            if win_core:
                fixed_max.append(max(win_core))
                fixed_mean.append(statistics.mean(win_core))
                mw = max(win_core)
                reuse_window.append(mw >= 0.55)
                per_pos_window[i + 1].append(mw)
        for i in range(1, len(eps)):
            adjacent.append(_core(eps[i - 1], eps[i]))

    # card_echo（零噪声）：本集 macro_move 是否命中其检索到的 cluster_id。
    # macro_move 优先取自签名，缺失时回退 extractions.primary_move_id（无需 LLM）。
    ctx = _load_contexts(run_dir)
    exts = _load_extractions(run_dir)
    ce_hits = ce_tot = 0
    ce_present = False
    for r in rows:
        c = ctx.get(r["job_id"])
        if not c:
            continue
        clusters = c.get("retrieved_cluster_ids") or []
        if not clusters:
            continue
        ce_present = True
        ce_tot += 1
        mm = r["signature"].get("macro_move")
        if mm is None:
            mm = exts.get(r["job_id"], {}).get("result", {}).get("primary_move_id")
        if mm is not None and mm in clusters:
            ce_hits += 1
    card_echo = (
        {"episodes_with_retrieval": ce_tot, "hits": ce_hits,
         "rate": round(ce_hits / ce_tot, 4) if ce_tot else None,
         "method": "macro_move(签名/抽取) ∈ retrieved_cluster_ids，零噪声"}
        if ce_present else
        {"note": "本组无检索（如 A 组协议禁止检索），card_echo 不适用；改用固定窗口。"}
    )

    return {
        "schema_version": SIGNATURE_SCHEMA_VERSION,
        "run_dir": run_dir.name,
        "window": window,
        "strict_window": strict_window,
        "episode_count": len(rows),
        "trajectory_count": len(trajs),
        "standard_confounded": {
            "all_pair_mean_core_similarity": round(mean(all_pairs), 4) if all_pairs else None,
            "adjacent_mean_core_similarity": round(mean(adjacent), 4) if adjacent else None,
            "mean_max_over_all_history": round(mean(all_hist_max), 4) if all_hist_max else None,
            "reuse_rate_max_all_history_ge_0.55": round(mean(reuse_hist), 4) if reuse_hist else None,
        },
        "position_fair_fixed_window": {
            "mean_max_window_core_similarity": round(mean(fixed_max), 4) if fixed_max else None,
            "mean_window_core_similarity": round(mean(fixed_mean), 4) if fixed_mean else None,
            "reuse_rate_window_ge_0.55": round(mean(reuse_window), 4) if reuse_window else None,
            "episodes_counted": len(reuse_window),
            "per_position_max_window_mean": {str(k): round(mean(v), 4) for k, v in sorted(per_pos_window.items())},
        },
        "confound_showcase_max_all_history_per_position": {
            str(k): round(mean(v), 4) for k, v in sorted(per_pos_hist.items())
        },
        "card_echo_macro_zero_noise": card_echo,
        "dimension_metrics": _dim_metrics(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--strict-window", action="store_true",
                    help="只统计窗口填满的集（丢弃每条轨迹开头 window-1 集），保证严格可比")
    args = ap.parse_args()
    summary = analyze(args.run_dir, window=args.window, strict_window=args.strict_window)
    suffix = "_strict" if args.strict_window else ""
    out = args.run_dir / f"narrative_signature_fair_v1_1{suffix}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    sc = summary["standard_confounded"]
    fw = summary["position_fair_fixed_window"]
    print(f"\n===== 位置公平分析：{summary['run_dir']} (window={summary['window']}) =====")
    print(f"集数={summary['episode_count']}  轨迹数={summary['trajectory_count']}")
    print("\n[受混淆] 全历史指标（位置越高越虚高）：")
    print(f"  all_pair_mean           = {sc['all_pair_mean_core_similarity']}")
    print(f"  mean_max_over_all_hist  = {sc['mean_max_over_all_history']}")
    print(f"  reuse_rate(全历史>=0.55) = {sc['reuse_rate_max_all_history_ge_0.55']}")
    print("\n[位置公平] 固定窗口(k=3)指标：")
    print(f"  mean_max_window         = {fw['mean_max_window_core_similarity']}")
    print(f"  reuse_rate(窗口>=0.55)   = {fw['reuse_rate_window_ge_0.55']}")
    print(f"  adjacent_mean(本身公平)  = {sc['adjacent_mean_core_similarity']}")
    print("\n[选项1] card_echo（零噪声，仅对有检索的组）：")
    print(" ", summary["card_echo_macro_zero_noise"])
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
