"""本地验证：当前重写后的检索器以 R0_current 口径重放，是否与历史 B 选卡一致。

不调用任何 API。逐集重放每条 B 轨迹（冻结高多样池，exposures 随集递增），
比较有序 retrieved_memory_ids 与历史 contexts.jsonl 记录。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from narrative_memory_feedback_v2_2.memory import MemoryPool
from narrative_memory_feedback_v2_2.schema import ROOT, load_protocol

B_DIRS = [
    ROOT / "runs" / "nmf_v3_multistory_v6_b",       # R01
    ROOT / "runs" / "nmf_v3_multistory_v6_b_r02",   # R02
    ROOT / "runs" / "nmf_v3_multistory_v6_b_r03",   # R03
]
TARGET_STORIES = {"S01", "S02", "S03", "S04"}


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    config, stories, plans, memory_cards, taxonomy = load_protocol()
    stories_by_id = {s["story_id"]: s for s in stories}
    plans_by_key = {(p["story_id"], p["episode_id"]): p for p in plans}
    episode_ids = config["episode_ids"]

    # 历史 B 的选卡（有序 memory_ids），键为 job_id
    hist_selected: dict[str, list[str]] = {}
    for d in B_DIRS:
        for row in read_jsonl(d / "contexts.jsonl"):
            hist_selected[row["job_id"]] = list(row.get("retrieved_memory_ids", []))

    # R0 检索设置：基准权重 + 空 override
    r0_settings = {**config["retrieval"], **config.get("retrieval_overrides", {}).get("R0_current", {})}

    # 按轨迹重放：story x run
    runs = ["R01", "R02", "R03"]
    total = 0
    matched = 0
    mismatches: list[dict] = []

    for story_id in sorted(TARGET_STORIES):
        story = stories_by_id[story_id]
        for run_id in runs:
            # 用 R0_current 条件建池（池标签 high_diverse，与 B 相同）；冻结池无写回
            pool = MemoryPool(memory_cards, condition="R0_current", story_id=story_id)
            for episode_id in episode_ids:
                plan = plans_by_key[(story_id, episode_id)]
                query = " ".join([story["genre"], plan["episode_goal"], *plan["open_decisions"]])
                cards, _events = pool.retrieve(query, settings=r0_settings)
                replay_ids = [c["memory_id"] for c in cards]

                job_id = f"{story_id}__B_high_diverse_frozen__{run_id}__{episode_id}"
                if job_id not in hist_selected:
                    continue
                total += 1
                hist_ids = hist_selected[job_id]
                if replay_ids == hist_ids:
                    matched += 1
                else:
                    mismatches.append({
                        "job_id": job_id,
                        "replay": replay_ids,
                        "history": hist_ids,
                        "same_set": set(replay_ids) == set(hist_ids),
                    })

    print(f"compared_jobs={total} matched={matched} mismatched={total - matched}")
    if total:
        print(f"exact_order_match_rate={matched / total:.4f}")
    set_match = sum(1 for m in mismatches if m["same_set"])
    print(f"mismatched_but_same_set={set_match} (仅排序不同)")
    print(f"mismatched_different_set={len(mismatches) - set_match}")
    for m in mismatches[:12]:
        tag = "SET-EQ/ORDER-DIFF" if m["same_set"] else "SET-DIFF"
        print(f"  [{tag}] {m['job_id']}")
        print(f"      replay : {m['replay']}")
        print(f"      history: {m['history']}")


if __name__ == "__main__":
    main()
