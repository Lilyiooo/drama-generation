"""叙事机制分布同质化：各臂 primary_move_id 分布对比 (S01-S04 x 3 run)。

指标：
  unique_primary/run : 每组(故事x run,16集)使用的不同 primary move 数 (越高越多样)
  xep_primary_repeat : 相邻集 primary move 相同比例 (越高越同质)
  primary_evenness   : 分布熵归一化到 [0,1] (越高越均匀)
  run_gini_primary   : 组内 move 频次 Gini (越高越集中)
  cross_run_jaccard  : 同故事不同 run 的 primary 集合 Jaccard (越高跨run越同质)
"""
import json
from collections import defaultdict, Counter
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
ARMS = {
    "A_state_only":   "nmf_v3_multistory_v6",
    "R0_current":     "nmf_v3_multistory_v6_b",
    "R1_no_exposure": "nmf_r1_no_exposure_gpt41",
}
KEEP = {"S01", "S02", "S03", "S04"}


def story_of(jid):
    return jid.split("__")[0]


def load_ext(dirname):
    d = RUNS / dirname
    rows = [json.loads(l) for l in open(d / "extractions.jsonl")]
    return {r["job_id"]: r["result"] for r in rows if story_of(r["job_id"]) in KEEP}


def entropy_norm(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    import math
    h = -sum((n / total) * math.log2(n / total) for n in counter.values() if n > 0)
    k = len(counter)
    return h / math.log2(k) if k > 1 else 0.0


def gini(vals):
    vals = sorted(v for v in vals if v > 0)
    n = len(vals)
    if n == 0:
        return 0.0
    s = sum(vals)
    if s == 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(vals))
    return (2 * cum) / (n * s) - (n + 1) / n


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def analyze(ext):
    groups = defaultdict(dict)
    for jid, res in ext.items():
        p = jid.split("__")
        story, run, ep = p[0], p[2], int(p[3][1:])
        groups[(story, run)][ep] = res.get("primary_move_id")

    uniq, xrep, even, gins, cr_jac = [], [], [], [], []
    for (story, run), eps in groups.items():
        moves = [eps[e] for e in sorted(eps)]
        c = Counter(moves)
        uniq.append(len(c))
        rep = sum(1 for i in range(1, len(moves)) if moves[i] == moves[i - 1]) / (len(moves) - 1) if len(moves) > 1 else 0
        xrep.append(rep)
        even.append(entropy_norm(c))
        gins.append(gini(list(c.values())))

    # cross-run: per story, mean Jaccard of primary-set across run pairs
    by_story = defaultdict(dict)
    for (story, run), eps in groups.items():
        by_story[story][run] = set(eps[e] for e in eps)
    for story, runs in by_story.items():
        rl = list(runs.keys())
        for i in range(len(rl)):
            for j in range(i + 1, len(rl)):
                cr_jac.append(jac(by_story[story][rl[i]], by_story[story][rl[j]]))

    def m(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "unique_primary_per_run": m(uniq),
        "xep_primary_repeat": m(xrep),
        "primary_evenness": m(even),
        "run_gini_primary": m(gins),
        "cross_run_jaccard": m(cr_jac),
    }


def main():
    print(f"{'arm':16} {'uniq/run':>9} {'xep_repeat':>10} {'evenness':>9} {'runGini':>8} {'cross_run_jac':>13}")
    for arm, d in ARMS.items():
        m = analyze(load_ext(d))
        print(f"{arm:16} {m['unique_primary_per_run']:9.2f} {m['xep_primary_repeat']:10.3f} {m['primary_evenness']:9.3f} {m['run_gini_primary']:8.3f} {m['cross_run_jaccard']:13.3f}")


if __name__ == "__main__":
    main()
