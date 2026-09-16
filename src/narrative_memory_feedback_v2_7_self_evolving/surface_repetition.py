"""场景+语言表层重复检测（离线，不消耗 API）。

结构性叙事签名（narrative_signature）捕捉的是 mechanism / obstacle / turn_type
等抽象骨架，对"每集都场景一""短句堆叠"这类**表层**重复不敏感。本模块补充一个
更具体、可直接感知的重复指标，分两层：

  场景层（scene）：
    1. opening_label_ratio        —— 开头含"场景X"标题的集比例（剧本正常格式，仅参考，不计入 SRI）
    2. opening_time_establishing_ratio —— 开头用固定时辰定场词（晨光/午后…）的集比例（套路定场信号）
    3. opening_template_score     —— 各集开头片段两两 4-gram Jaccard 均值（越高越说明每集开头雷同）

  语言层（language）：
    4. punct_gap_avg              —— 标点间平均字数（越短越"AI 味"）
    5. short_clause_ratio         —— 超短分句（<8 字）占比
    6. cross_ep_4gram_rep         —— 本集 4-gram 落在前面已生成集的比例（生成时视角，衡量随集数累积复用；旧版含所有集，会掩盖后段升高趋势）
    7. top_repeated_phrases       —— 跨集最重复的 4 字短语清单（直接可见"在重复什么"）

综合指标 surface_repetition_index（SRI，0–1，越高重复越重）= 上述 2/3/4/6 归一化后等权平均（场景标题与固有地点词均不计入）。

输出：
  <run>/surface_repetition.jsonl       —— 每集一行
  <run>/surface_repetition_summary.json —— 每轨迹一个对象 + 指标定义
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .api_client import OpenAICompatibleClient
from .io_utils import read_jsonl, utc_now, write_json
from .narrative_signature import compute_all_boundaries
from .schema import ROOT

# 中文标点 + 空白（用作分句切分符）
_PUNCT = set("。！？；：，、…—（）《》「」『』""''· \t\n\r")
# 只保留汉字，去掉标点/空白/英文，用于 4-gram 比较
_CJK = re.compile(r"[一-鿿]")
# 常见固定定场词，分两类：
#   时辰类（套路定场信号，跨题材通用，应尽量避免每集雷同）——计入 SRI 的场景层
_TIME_ESTABLISHING = ("晨光", "清晨", "日出", "朝阳", "暮色", "黄昏", "夜色", "傍晚", "午后", "晨雾", "夜半")
#   地点/物件类（故事固有词，如本故事的草料/羊圈；仅作参考，不计入 SRI 以免把剧情固有词误判为套路）
_LOCATION = ("羊圈", "草料", "路线图", "院内", "屋内", "帐篷", "蒙古包", "羊棚", "草料棚")
_SCENE_LABEL = re.compile(r"场景[一二三四五六七八九十]")

N = 120          # 开头片段取前 N 个汉字
SHORT = 8        # 超短分句阈值（字）

# ── formulaic_density 排除词 ──
# 场景标签碎片：出现在 4-gram 中说明该短语是剧本格式词而非自然语言
_SCENE_FRAGMENTS = {
    "场景", "闪回", "人物", "集终", "本集", "第集", "集完",
    "日人物", "夜人物", "画外", "字幕", "转场",
    "内景", "外景", "旁白", "独白", "镜头", "特写",
}


def _keep_cjk(text: str) -> str:
    return "".join(_CJK.findall(text))


def _clause_gaps(text: str) -> list[int]:
    chunks = re.split(r"[。！？；：，、…—（）《》「」『』""''· \t\n\r]", text)
    return [len(c) for c in chunks if c.strip()]


def _ngrams(s: str, k: int = 4) -> list[str]:
    return [s[i:i + k] for i in range(len(s) - k + 1)] if len(s) >= k else []


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _max_self_window_jaccard(text: str, window: int = 200, step: int = 50) -> float:
    """单集内非重叠窗口的最大 4-gram Jaccard，检测集内自我重复。

    对同一集的所有滑动窗口，比较不相交（起始位置差 ≥ window）
    的窗口对，返回最高的 4-gram Jaccard。
    """
    cjk = _keep_cjk(text)
    if len(cjk) < window * 2:
        return 0.0
    win_sets = _sliding_window_gram_sets(cjk, window, step)
    n = len(win_sets)
    # 计算哪些窗口对不相交：起始位置差 ≥ window
    step_chars = _step_until_disjoint(len(cjk), window, step)
    best = 0.0
    for i in range(n):
        # 只跟后面足够远的窗口比，避免重复计算
        for j in range(i + step_chars, n):
            sim = _jaccard(win_sets[i], win_sets[j])
            if sim > best:
                best = sim
    return best


def _step_until_disjoint(total_len: int, window: int, step: int) -> int:
    """两个窗口起始位置差多少个 step 才算不相交（差 ≥ window chars）。"""
    return max(1, (window + step - 1) // step)


def _max_window_jaccard(
    text_a: str, text_b: str,
    window: int = 200, step: int = 50,
) -> float:
    """两篇文本的滑动窗口最大 4-gram Jaccard，检测最相似段落。

    对 text_a 和 text_b 各取 window 字窗口，步长 step，
    计算所有交叉窗口对的 4-gram Jaccard，返回最大值。
    代表"这两集里最像的那段有多像"。
    """
    cjk_a = _keep_cjk(text_a)
    cjk_b = _keep_cjk(text_b)
    if len(cjk_a) < window or len(cjk_b) < window:
        return 0.0
    # 预计算各窗口的 4-gram 集合
    win_sets_a = _sliding_window_gram_sets(cjk_a, window, step)
    win_sets_b = _sliding_window_gram_sets(cjk_b, window, step)
    if not win_sets_a or not win_sets_b:
        return 0.0
    best = 0.0
    for sa in win_sets_a:
        for sb in win_sets_b:
            j = _jaccard(sa, sb)
            if j > best:
                best = j
    return best


def _sliding_window_gram_sets(text: str, window: int, step: int) -> list[set[str]]:
    """取 text 的滑动窗口，返回每个窗口的 4-gram set。"""
    results: list[set[str]] = []
    i = 0
    while i + window <= len(text):
        chunk = text[i:i + window]
        grams = _ngrams(chunk)
        if grams:
            results.append(set(grams))
        i += step
    return results


# ── formulaic_density 辅助函数 ──

def _extract_character_names(scripts: list[str]) -> set[str]:
    """从剧本中提取角色名。

    支持三种格式：
      - drama:     人物：顾千雪、王董、李总  /  顾千雪（动作）：台词
      - custom:    **莫岚** 独行  /  人物：秦朗、莫岚
      - D_diverse: 无人物：标签，从对话前缀提取（角色名+——台词）
    """
    names: set[str] = set()
    for script in scripts:
        # ① "人物：" 标签
        for m in re.finditer(r"人物[：:]\s*([^\n]+)", script):
            # 用顿号/逗号/空格分割，取2-3字中文名
            for name in re.findall(r"[\u4e00-\u9fff]{2,3}", m.group(1)):
                names.add(name)
        # ② 对话前缀：角色名(动作)：台词  /  角色名：台词
        for m in re.finditer(r"(?:^|\n)([\u4e00-\u9fff]{2,3})[（(][^)]*[)）][：:]", script):
            names.add(m.group(1))
        # ③ **角色名** 独行 (custom 格式)
        for m in re.finditer(r"\*\*([\u4e00-\u9fff]{2,3})\*\*", script):
            names.add(m.group(1))
        # ④ 角色名\n——台词 (D_diverse 格式)
        for m in re.finditer(r"(?:^|\n)([\u4e00-\u9fff]{2,3})\n——", script):
            names.add(m.group(1))
        # ⑤ 角色名（动作）\n——台词
        for m in re.finditer(r"(?:^|\n)([\u4e00-\u9fff]{2,3})[（(][^)]*[)）]\n——", script):
            names.add(m.group(1))
    return names


def _detect_proper_noun_fragments(
    per_ep_cjk: list[str],
    char_names: set[str],
    n_eps: int,
) -> set[str]:
    """检测故事专有名词片段。

    启发式：某 2-3 字片段在单集中出现 ≥5 次，
    且在其他集中仅在 ≤20% 的集中出现过 → 判定为故事专有名词。
    """
    if n_eps <= 2:
        return set()

    # 统计每个 2-3 字 fragment 在各集中的出现次数
    ep_frag_counts: list[dict[str, int]] = []
    for ep_cjk in per_ep_cjk:
        counts: dict[str, int] = defaultdict(int)
        for k in (2, 3):
            for i in range(len(ep_cjk) - k + 1):
                frag = ep_cjk[i:i + k]
                counts[frag] = counts.get(frag, 0) + 1
        ep_frag_counts.append(counts)

    proper_nouns: set[str] = set()
    max_other_eps = max(1, int(n_eps * 0.2))

    for ep_idx, counts in enumerate(ep_frag_counts):
        for frag, cnt in counts.items():
            if cnt < 5:
                continue
            # 跳过角色名和场景标签
            if frag in char_names or frag in _SCENE_FRAGMENTS:
                continue
            # 检查在其他集中是否很少出现
            n_other_eps = sum(
                1 for j, ec in enumerate(ep_frag_counts)
                if j != ep_idx and frag in ec
            )
            if n_other_eps <= max_other_eps:
                proper_nouns.add(frag)

    return proper_nouns


def _contains_any(gram: str, exclusion_set: set[str]) -> bool:
    """检查 4-gram 是否包含排除集合中的任意子串。"""
    for ex in exclusion_set:
        if ex in gram:
            return True
    return False


def _compute_formulaic_density(
    per_ep_grams: list[set[str]],
    n_eps: int,
    char_names: set[str],
    proper_nouns: set[str],
) -> tuple[float, list[dict[str, Any]]]:
    """计算 formulaic_density 和 top formulaic phrases。

    formulaic_density = N_formulaic / N_total_4grams
    其中 formulaic 4-gram 需同时满足：
      a) 覆盖 ≥15% 集数（至少 3 集）
      b) 单集出现 ≤2 次
      c) 不含角色名
      d) 不含场景标签碎片
      e) 不含故事专有名词
    """
    min_eps = 3  # 固定绝对门槛，避免长序列被 15% 规则过度惩罚

    # 统计每个 4-gram 在各集的出现次数
    gram_ep_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for ep_idx, grams in enumerate(per_ep_grams):
        for gram in grams:
            gram_ep_counts[gram][ep_idx] += 1

    # 排除词集合
    all_exclusions = char_names | _SCENE_FRAGMENTS | proper_nouns

    total_grams = sum(len(gs) for gs in per_ep_grams)
    formulaic_grams: dict[str, int] = {}  # gram -> episode_count

    for gram, ep_counts in gram_ep_counts.items():
        n_eps_covered = len(ep_counts)
        # a) 覆盖足够多集
        if n_eps_covered < min_eps:
            continue
        # b) 单集不超过 2 次
        if any(cnt > 2 for cnt in ep_counts.values()):
            continue
        # c/d/e) 不含排除词
        if _contains_any(gram, all_exclusions):
            continue
        formulaic_grams[gram] = n_eps_covered

    n_formulaic = len(formulaic_grams)
    density = n_formulaic / total_grams if total_grams > 0 else 0.0

    # Top formulaic phrases
    sorted_grams = sorted(formulaic_grams.items(), key=lambda x: (-x[1], x[0]))[:20]
    top_formulaic = [
        {"phrase": g, "episode_count": c, "episode_share": round(c / n_eps, 3)}
        for g, c in sorted_grams
    ]

    return density, top_formulaic


# ── 专有名词自动检测（可选，消耗 API） ──
_PROPER_NOUN_CACHE = "proper_nouns_cache.json"
_STORY_TITLES: dict[str, str] | None = None


def _load_story_titles() -> dict[str, str]:
    """从 stories.json 读取 story_id → title 映射（模块级缓存）。"""
    global _STORY_TITLES
    if _STORY_TITLES is None:
        try:
            stories = json.loads((ROOT / "data" / "stories.json").read_text(encoding="utf-8"))
            _STORY_TITLES = {s["story_id"]: s["title"] for s in stories if s.get("title")}
        except (OSError, json.JSONDecodeError, KeyError):
            _STORY_TITLES = {}
    return _STORY_TITLES


def _title_exclusions(title: str) -> set[str]:
    """把剧名及其 3 字子串加入排除，覆盖剧名与「场景」标签交界产生的 4-gram 片段。

    例如「雨夜门禁」会产生「雨夜门禁」「夜门禁场」两个漏网 4-gram，
    排除「雨夜门禁」「雨夜门」「夜门禁」后二者均被覆盖。
    """
    title = (title or "").strip()
    ex: set[str] = set()
    if not title:
        return ex
    ex.add(title)
    for i in range(len(title) - 2):
        ex.add(title[i:i + 3])
    return {x for x in ex if x}


def _sample_scripts(scripts: list[str], per_script_chars: int = 400, cap: int = 20000) -> str:
    """对多集剧本采样，取每集开头若干字，避免全量送入模型。"""
    parts = [s[:per_script_chars] for s in scripts]
    merged = "\n".join(parts)
    return merged[:cap]


def _detect_proper_nouns_api(scripts: list[str], story_id: str, client: OpenAICompatibleClient) -> set[str]:
    """调用 LLM 提取故事世界专有名词（地名/机构/物件/事件/头衔等），排除角色名。"""
    sample = _sample_scripts(scripts)
    prompt = f"""你是一名剧本分析师。下面是一部连续剧（{story_id}）若干集的剧本节选。

请提取该故事世界中反复出现的「专有名词」，包括：地名、机构/组织/公司名、物件专名、事件专名、职业/头衔专名、特定称谓等。

要求：
1. 只提取属于该故事设定、且会在剧情中反复出现的专有名词，不要提取普通动词、形容词、量词、通用场景词（如"屋内""门外""电话"）
2. 不要提取角色名（角色名会单独处理）
3. 每个专名用最简形式（2-6 个汉字），去重
4. 只输出一个 JSON 数组，不要任何解释，例如：["恒森冷链","旧港区","喷淋系统","保险理赔"]

剧本节选：
{sample}

只输出 JSON 数组："""
    raw = client.complete(
        [{"role": "user", "content": prompt}],
        settings={"temperature": 0.0, "max_output_tokens": 800},
        seed=20260813, purpose="proper_noun_detection",
    )
    return _parse_proper_nouns(raw)


def _parse_proper_nouns(raw: str) -> set[str]:
    """从模型输出中解析专有名词 JSON 数组，失败则返回空集。"""
    if not raw:
        return set()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return {str(x).strip() for x in parsed if str(x).strip()}
    except (json.JSONDecodeError, ValueError):
        pass
    # 兜底：尝试提取 JSON 数组片段
    m = re.search(r"\[[^\]]*\]", raw, re.S)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return {str(x).strip() for x in parsed if str(x).strip()}
        except (json.JSONDecodeError, ValueError):
            pass
    return set()


def _load_or_detect_proper_nouns(
    run_dir: Path,
    story_id: str,
    scripts: list[str],
    detect: bool,
    client: OpenAICompatibleClient | None,
) -> set[str]:
    """读取专名缓存；无缓存且开启检测时调用 API，并写回缓存。"""
    cache_path = run_dir / _PROPER_NOUN_CACHE
    cache: dict[str, list[str]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            cache = {}
    if story_id in cache:
        return set(cache[story_id])
    if not detect:
        return set()
    if client is None:
        return set()
    nouns = _detect_proper_nouns_api(scripts, story_id, client)
    cache[story_id] = sorted(nouns)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return nouns


def _trajectory_id(job_id: str) -> str:
    return "__".join(job_id.split("__")[:-1])


def _episode_id(job_id: str) -> str:
    return job_id.split("__")[-1]


def _episode_record(job_id: str, script: str, other_ngram_sets: list[set[str]]) -> dict[str, Any]:
    cjk = _keep_cjk(script)
    gaps = _clause_gaps(script)
    punct_gap = mean(gaps) if gaps else 0.0
    short_ratio = (sum(1 for g in gaps if g < SHORT) / len(gaps)) if gaps else 0.0
    self_rep = (1 - len(set(_ngrams(cjk))) / len(_ngrams(cjk))) if len(cjk) >= 4 else 0.0
    grams = set(_ngrams(cjk))
    # 生成时视角：仅与前面已生成集比较，衡量"随集数累积复用"。
    cross_rep = 0.0
    if grams and other_ngram_sets:
        union_other = set().union(*other_ngram_sets)
        cross_rep = len(grams & union_other) / len(grams) if grams else 0.0
    opening = _keep_cjk(script)[:N]
    self_win_rep = _max_self_window_jaccard(script)
    return {
        "job_id": job_id,
        "episode_id": _episode_id(job_id),
        "opening_120": opening,
        "opening_has_scene_label": bool(_SCENE_LABEL.search(script[:120])),
        "opening_has_time_establishing": any(w in script[:120] for w in _TIME_ESTABLISHING),
        "opening_has_location_word": any(w in script[:120] for w in _LOCATION),
        "punct_gap_avg": round(punct_gap, 3),
        "short_clause_ratio": round(short_ratio, 4),
        "self_4gram_rep": round(self_rep, 4),
        "cross_ep_4gram_rep": round(cross_rep, 4),
        "self_window_repeat": round(self_win_rep, 4),
    }


def analyze(run_dir: Path, detect_proper_nouns: bool = False) -> dict[str, Any]:
    gens = read_jsonl(run_dir / "generations.jsonl")
    client = OpenAICompatibleClient.from_environment("evaluation") if detect_proper_nouns else None
    by_traj: dict[str, list[dict[str, str]]] = defaultdict(list)
    for g in gens:
        by_traj[_trajectory_id(g["job_id"])].append(g)

    per_episode: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []

    for traj_id, rows in sorted(by_traj.items()):
        rows.sort(key=lambda r: r["job_id"])
        # 预计算每集 CJK 4-gram 集合，用于跨集复用
        cjk_sets = [_keep_cjk(r["script"]) for r in rows]
        ngram_sets = [_ngrams(c) for c in cjk_sets]
        episode_recs: list[dict[str, Any]] = []
        for i, r in enumerate(rows):
            others = ngram_sets[:i]  # 生成时视角：仅前面已生成集，避免上帝视角掩盖后段累积复用
            rec = _episode_record(r["job_id"], r["script"], others)
            episode_recs.append(rec)
            per_episode.append(rec)

        # ── 全局表面相似度（替换旧版开篇偏好） ──
        # ① 全篇两两 4-gram Jaccard：所有集对的全文本相似度均值
        ngram_sets_as_set = [set(gs) for gs in ngram_sets]
        global_pairs = [(ngram_sets_as_set[i], ngram_sets_as_set[j])
                        for i in range(len(ngram_sets_as_set)) for j in range(i + 1, len(ngram_sets_as_set))]
        global_pairwise_jaccard = mean(_jaccard(a, b) for a, b in global_pairs) if global_pairs else 0.0

        # 旧版开篇指标（仍输出但不计入新 SRI）
        openings = [_keep_cjk(r["script"])[:N] for r in rows]
        opening_grams = [set(_ngrams(o)) for o in openings]
        opening_pairs = [(opening_grams[i], opening_grams[j])
                         for i in range(len(opening_grams)) for j in range(i + 1, len(opening_grams))]
        opening_template = mean(_jaccard(a, b) for a, b in opening_pairs) if opening_pairs else 0.0
        label_ratio = mean(1.0 if r["opening_has_scene_label"] else 0.0 for r in episode_recs)
        time_establishing_ratio = mean(1.0 if r["opening_has_time_establishing"] else 0.0 for r in episode_recs)
        location_ratio = mean(1.0 if r["opening_has_location_word"] else 0.0 for r in episode_recs)

        # 语言层
        mean_gap = mean(r["punct_gap_avg"] for r in episode_recs)
        mean_short = mean(r["short_clause_ratio"] for r in episode_recs)
        mean_cross = mean(r["cross_ep_4gram_rep"] for r in episode_recs)

        # ── 滑动窗口最大局部相似度 ──
        # 对所有集对 (i<j)，计算 max 窗口 4-gram Jaccard，检测"最像的段落有多像"
        texts = [r["script"] for r in rows]
        window_sims: list[float] = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                wj = _max_window_jaccard(texts[i], texts[j])
                window_sims.append(wj)
        mean_window_repeat = mean(window_sims) if window_sims else 0.0
        max_window_repeat = max(window_sims) if window_sims else 0.0

        # 跨集短语复用计数（哪些 4 字短语出现在最多集里）
        per_ep_grams: list[set[str]] = [set(_ngrams(_keep_cjk(r["script"]))) for r in rows]
        phrase_ep_count: dict[str, int] = defaultdict(int)
        for gs in per_ep_grams:
            for gram in gs:
                phrase_ep_count[gram] += 1
        top_phrases = [
            {"phrase": p, "episode_count": c, "episode_share": round(c / len(rows), 3)}
            for p, c in Counter(phrase_ep_count).most_common(15) if c >= 2
        ]

        # ② 弥漫短语占比：出现在 ≥半数集 中的 4-gram 占全部集的 4-gram 集合比例
        half_ep = max(1, len(rows) // 2)
        pervasive_count = sum(1 for _p, c in phrase_ep_count.items() if c >= half_ep)
        pervasive_phrase_share = pervasive_count / len(phrase_ep_count) if phrase_ep_count else 0.0

        # ── formulaic_density：弥漫式 AI 写作指纹 ──
        # 测"模型不受情节推动，不由自主反复用同一种措辞"的程度
        scripts = [r["script"] for r in rows]
        char_names = _extract_character_names(scripts)
        story_id = rows[0].get("story_id") or _trajectory_id(rows[0]["job_id"]).split("__")[0]
        proper_nouns = _load_or_detect_proper_nouns(
            run_dir, story_id, scripts, detect_proper_nouns, client,
        )
        proper_nouns |= _title_exclusions(_load_story_titles().get(story_id, ""))
        fd_density, fd_top_phrases = _compute_formulaic_density(
            per_ep_grams, len(rows), char_names, proper_nouns,
        )

        # ── 邻集边界场景重叠：检测"前一集结尾 = 后一集开头"的即视感重复 ──
        # 三层加权：地点匹配(0.35) + 角色重叠(0.30) + 对话内容相似度(0.35)
        boundaries = compute_all_boundaries(scripts)
        boundary_scores = [b["boundary_overlap_score"] for b in boundaries]
        boundary_mean = mean(boundary_scores) if boundary_scores else 0.0
        boundary_max = max(boundary_scores) if boundary_scores else 0.0
        boundary_high = sum(1 for s in boundary_scores if s > 0.7)
        boundary_mean_dialog = mean(b["dialogue_similarity"] for b in boundaries) if boundaries else 0.0

        # 前后半段趋势（回答"越往后是否更重复"）
        half = max(1, len(episode_recs) // 2)
        early = episode_recs[:half]
        late = episode_recs[half:]
        trend = {
            "early_cross_ep_4gram_rep": round(mean(r["cross_ep_4gram_rep"] for r in early), 4) if early else None,
            "late_cross_ep_4gram_rep": round(mean(r["cross_ep_4gram_rep"] for r in late), 4) if late else None,
            "early_punct_gap_avg": round(mean(r["punct_gap_avg"] for r in early), 3) if early else None,
            "late_punct_gap_avg": round(mean(r["punct_gap_avg"] for r in late), 3) if late else None,
            "early_opening_label_ratio": round(mean(1.0 if r["opening_has_scene_label"] else 0.0 for r in early), 3) if early else None,
            "late_opening_label_ratio": round(mean(1.0 if r["opening_has_scene_label"] else 0.0 for r in late), 3) if late else None,
        }

        # ── 综合 SRI_v2：全局表面相似度，等权四分量 ──
        # 分量说明：
        #   ① global_pairwise_jaccard    全篇两两 4-gram Jaccard（全文本，替代旧版开篇偏好）
        #   ② pervasive_phrase_share    跨 ≥半数集重复的 4-gram 占全部 4-gram 的比例
        #   ③ clause_shortness          短句归一化（保留，为全局标点特征）
        #   ④ mean_cross                跨集 4-gram 复用（保留，累积复用）
        clause_shortness = _clamp((16 - mean_gap) / 16)
        sri_v2 = mean([global_pairwise_jaccard, pervasive_phrase_share, clause_shortness, mean_cross])
        # 旧版 SRI（保留兼容）
        sri_legacy = mean([opening_template, time_establishing_ratio, clause_shortness, mean_cross])
        # ── SRI_v3：用 formulaic_density 替换 pervasive_phrase_share（第②分量） ──
        # formulaic_density 归一化到 [0,1]：参考 GPT-4.1 最大值约 0.032，Gemini 约 0.007
        fd_normalized = _clamp(fd_density * 30)  # ×30 使 3.2%→0.96，0.7%→0.21
        sri_v3 = mean([global_pairwise_jaccard, fd_normalized, clause_shortness, mean_cross])

        trajectories.append({
            "trajectory_id": traj_id,
            "n_episodes": len(rows),
            # v2 (全局) 表面重复指标
            "global_pairwise_4gram_jaccard": round(global_pairwise_jaccard, 4),
            "pervasive_phrase_share": round(pervasive_phrase_share, 4),
            "surface_repetition_index": round(sri_v2, 4),          # SRI_v2
            "surface_repetition_index_legacy": round(sri_legacy, 4),
            # v3 指标
            "formulaic_density": round(fd_density, 6),
            "formulaic_density_pct": round(fd_density * 100, 4),
            "surface_repetition_index_v3": round(sri_v3, 4),       # SRI_v3
            "top_formulaic_phrases": fd_top_phrases,
            # 邻集边界场景重叠
            "boundary_mean_score": round(boundary_mean, 4),
            "boundary_max_score": round(boundary_max, 4),
            "boundary_high_count": boundary_high,
            "boundary_mean_dialog_sim": round(boundary_mean_dialog, 4),
            # 旧版开篇指标（保留输出但不计入新 SRI）
            "opening_label_ratio": round(label_ratio, 4),
            "opening_time_establishing_ratio": round(time_establishing_ratio, 4),
            "opening_location_ratio": round(location_ratio, 4),
            "opening_template_score": round(opening_template, 4),
            # 语言层
            "mean_punct_gap_avg": round(mean_gap, 3),
            "mean_short_clause_ratio": round(mean_short, 4),
            "mean_cross_ep_4gram_rep": round(mean_cross, 4),
            # 滑动窗口局部重复
            "mean_window_repeat": round(mean_window_repeat, 4),
            "max_window_repeat": round(max_window_repeat, 4),
            "mean_self_window_repeat": round(mean(r["self_window_repeat"] for r in episode_recs), 4),
            "trend_early_vs_late": trend,
            "top_repeated_phrases": top_phrases,
        })

    summary = {
        "created_at": utc_now(),
        "metric_definitions": {
            # v2 全局指标
            "global_pairwise_4gram_jaccard": "全篇两两 4-gram Jaccard 均值（所有集对的全文本，非仅开头）；0–1，越低越不雷同",
            "pervasive_phrase_share": "跨 ≥半数集重复的 4-gram 占全部集 4-gram 集合的比例；越高越说明短语充斥各集",
            "cross_ep_4gram_rep": "本集 4-gram 落在前面已生成集的比例（生成时视角），衡量随集数累积复用",
            "surface_repetition_index": "SRI_v2=均值(全局两两Jaccard, 弥漫短语占比, 短句归一, 跨集复用)；全部使用全文本，不再偏好开头",
            "surface_repetition_index_legacy": "旧版 SRI=均值(开头模板, 时辰定场比, 短句归一, 跨集复用)，开头 120 字偏重",
            # v3 指标
            "formulaic_density": "弥漫式 AI 写作指纹：覆盖 ≥15%集且单集 ≤2次、不含角色名/场景词/专名的 4-gram 占全部 4-gram 比例。测模型不由自主重复同种措辞的程度",
            "surface_repetition_index_v3": "SRI_v3=均值(全局两两Jaccard, formulaic_density归一化×30, 短句归一, 跨集复用)；用 formulaic_density 替换 pervasive_phrase_share",
            "top_formulaic_phrases": "弥漫式 AI 指纹短语及其覆盖集数",
            "boundary_mean_score": "邻集边界场景重叠均值：检测'前一集结尾=后一集开头'的即视感重复（地点匹配0.35+角色重叠0.30+对话相似0.35）",
            "boundary_max_score": "所有邻集边界重叠分数的最大值（最高的一对边界重复程度）",
            "boundary_high_count": "边界重叠分数>0.7 的邻集对数（高即视感重复的边界数量）",
            "boundary_mean_dialog_sim": "邻集边界对话内容相似度均值（'说的话都差不多'的即视感）",
            # 旧版开篇指标（保留但不计入新 SRI）
            "opening_label_ratio": '开头含\u201c场景X\u201d标题的集比例',
            "opening_time_establishing_ratio": "开头 120 字内含时辰定场词的集比例",
            "opening_location_ratio": "开头含故事固有地点/物件词的集比例",
            "opening_template_score": "各集开头片段两两 4-gram Jaccard 均值",
            # 语言层
            "punct_gap_avg": "标点间平均字数，越短越显 AI 味",
            "short_clause_ratio": "超短分句（<8 字）占全部分句比例",
            "top_repeated_phrases": "跨≥2集重复最多的 4 字短语及覆盖集比例",
        },
        "trajectories": trajectories,
    }
    write_json(run_dir / "surface_repetition_summary.json", summary)
    with (run_dir / "surface_repetition.jsonl").open("w", encoding="utf-8") as fh:
        for rec in per_episode:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "nmf_v3_smoke_s03")
    parser.add_argument("--detect-proper-nouns", action="store_true", help="调用 API 自动检测故事专有名词（默认关闭，保持离线）")
    args = parser.parse_args()
    summary = analyze(args.run_dir, detect_proper_nouns=args.detect_proper_nouns)
    for t in summary["trajectories"]:
        print(f"轨迹 {t['trajectory_id']} (n={t['n_episodes']})")
        print(f"  SRI_v2={t['surface_repetition_index']}  SRI_v3={t.get('surface_repetition_index_v3','N/A')}  formulaic_density={t.get('formulaic_density_pct','N/A')}%")
        print(f"  边界重复 mean={t.get('boundary_mean_score','N/A')}  max={t.get('boundary_max_score','N/A')}  high(>0.7)={t.get('boundary_high_count','N/A')}  dialog_sim={t.get('boundary_mean_dialog_sim','N/A')}")
        print(f"  开场模板={t['opening_template_score']}  时辰定场比={t['opening_time_establishing_ratio']}  地点词比={t['opening_location_ratio']}")
        print(f"  标点均字={t['mean_punct_gap_avg']}  短句比={t['mean_short_clause_ratio']}  跨集复用={t['mean_cross_ep_4gram_rep']}")
        tr = t["trend_early_vs_late"]
        print(f"  趋势 跨集复用 early={tr['early_cross_ep_4gram_rep']} late={tr['late_cross_ep_4gram_rep']} | 标点均字 early={tr['early_punct_gap_avg']} late={tr['late_punct_gap_avg']} | 标签比 early={tr['early_opening_label_ratio']} late={tr['late_opening_label_ratio']}")
        if t["top_repeated_phrases"]:
            top = ", ".join(f"{p['phrase']}({p['episode_share']})" for p in t["top_repeated_phrases"][:8])
            print(f"  高频重复短语: {top}")
        fd = t.get("top_formulaic_phrases", [])
        if fd:
            ftop = ", ".join(f"{p['phrase']}({p['episode_share']})" for p in fd[:8])
            print(f"  弥漫AI指纹: {ftop}")
    print(f"episode_rows={len(summary['trajectories'])} summary_written=surface_repetition_summary.json")


if __name__ == "__main__":
    main()
