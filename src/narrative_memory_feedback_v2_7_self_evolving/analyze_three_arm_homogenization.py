"""三臂（A / R0 / R1）同质化对比分析。

- A_state_only    : nmf_v3_multistory_v6            (无记忆基线, cards=[])
- R0_current(=B)  : nmf_v3_multistory_v6_b          (检索+曝光, 已验证与当前检索器等价)
- R1_no_exposure  : nmf_r1_no_exposure_gpt41        (检索, 曝光置零)

统一取 S01-S04 × 3 run 对比。指标：
  检索层(retrieval): cross_episode_repeat / top_cluster_share / run_Gini
  表层(surface)    : surface_repeat(相邻集4-gram Jaccard) / lexical_diversity(1-重复率)
"""
from collections import defaultdict, Counter
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
ARMS = {
    "A_state_only":   "nmf_v3_multistory_v6",
    "R0_current":     "nmf_v3_multistory_v6_b",
    "R1_no_exposure": "nmf_r1_no_exposure_gpt41",
}
KEEP_STORIES = {"S01", "S02", "S03", "S04"}


def story_of(job_id):
    return job_id.split("__")[0]


def load(dirname):
    d = RUNS / dirname
    gens = [r for r in (json_loads(l) for l in open(d / "generations.jsonl")) if r["story_id"] in KEEP_STORIES]
    ctx = {r["job_id"]: r for r in (json_loads(l) for l in open(d / "contexts.jsonl")) if story_of(r["job_id"]) in KEEP_STORIES}
    return gens, ctx


def json_loads(line):
    import json
    return json.loads(line)


def four_grams(text, n=4):
    text = "".join(ch for ch in text if ch.strip())
    return [text[i:i + n] for i in range(len(text) - n + 1)] if len(text) >= n else []


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


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


def analyze(gens, ctx):
    # group by (story, run)
    groups = defaultdict(dict)
    for r in gens:
        parts = r["job_id"].split("__")
        story, run, ep = parts[0], parts[2], int(parts[3][1:])
        groups[(story, run)][ep] = r

    # retrieval metrics
    ret_repeat, cluster_counter, run_ginis = [], Counter(), []
    has_retrieval = False
    for (story, run), eps in groups.items():
        mem_sets, clu_counter = [], Counter()
        for ep in sorted(eps):
            c = ctx.get(eps[ep]["job_id"], {})
            mids = c.get("retrieved_memory_ids") or []
            cids = c.get("retrieved_cluster_ids") or []
            if mids:
                has_retrieval = True
            mem_sets.append(set(mids))
            for cid in cids:
                clu_counter[cid] += 1
        for i in range(1, len(mem_sets)):
            ret_repeat.append(jaccard(mem_sets[i - 1], mem_sets[i]))
        if clu_counter:
            for cid, n in clu_counter.items():
                cluster_counter[cid] += n
            run_ginis.append(gini(list(clu_counter.values())))

    # surface metrics (char 4-gram)
    surf_repeat, lex_divs = [], []
    for (story, run), eps in groups.items():
        gram_sets, all_grams = [], []
        for ep in sorted(eps):
            grams = four_grams(eps[ep]["script"])
            gram_sets.append(set(grams))
            all_grams.extend(grams)
        for i in range(1, len(gram_sets)):
            surf_repeat.append(jaccard(gram_sets[i - 1], gram_sets[i]))
        if all_grams:
            lex_divs.append(len(set(all_grams)) / len(all_grams))

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "has_retrieval": has_retrieval,
        "cross_episode_repeat_retrieval": mean(ret_repeat),
        "top_cluster_share": (cluster_counter.most_common(1)[0][1] / sum(cluster_counter.values())) if cluster_counter else float("nan"),
        "run_Gini": mean(run_ginis),
        "surface_repeat": mean(surf_repeat),
        "lexical_diversity": mean(lex_divs),
        "n_groups": len(groups),
    }


def main():
    print(f"{'arm':16} {'ret?':5} {'xep_retrieval':>14} {'top_clu':>8} {'runGini':>8} {'surface_repeat':>14} {'lex_div':>8}")
    results = {}
    for arm, dirname in ARMS.items():
        gens, ctx = load(dirname)
        m = analyze(gens, ctx)
        results[arm] = m
        print(f"{arm:16} {str(m['has_retrieval']):5} {m['cross_episode_repeat_retrieval']:14.3f} {m['top_cluster_share']:8.3f} {m['run_Gini']:8.3f} {m['surface_repeat']:14.3f} {m['lexical_diversity']:8.3f}")
    print()
    # 解读
    if results["R0_current"]["has_retrieval"] and results["R1_no_exposure"]["has_retrieval"]:
        dr = results["R0_current"]["cross_episode_repeat_retrieval"] - results["R1_no_exposure"]["cross_episode_repeat_retrieval"]
        print(f"检索跨集重复: R0={results['R0_current']['cross_episode_repeat_retrieval']:.3f}  R1={results['R1_no_exposure']['cross_episode_repeat_retrieval']:.3f}  (差 {dr:+.3f})")
        print(f"表层跨集重复: A={results['A_state_only']['surface_repeat']:.3f}  R0={results['R0_current']['surface_repeat']:.3f}  R1={results['R1_no_exposure']['surface_repeat']:.3f}")
        print(f"词汇多样性  : A={results['A_state_only']['lexical_diversity']:.3f}  R0={results['R0_current']['lexical_diversity']:.3f}  R1={results['R1_no_exposure']['lexical_diversity']:.3f}")


if __name__ == "__main__":
    main()
