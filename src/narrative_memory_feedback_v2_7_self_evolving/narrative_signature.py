from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable

SIGNATURE_SCHEMA_VERSION = "1.1"

CONTROLLED_VALUES = {
    "mechanism_operations": {
        "INFORMATION_COMPARISON", "CLAIM_TEST", "DISCOVERY", "DISCLOSURE",
        "NEGOTIATION", "CONFRONTATION", "DECISION_OR_COMMITMENT", "ATTEMPT_AND_FAILURE",
        "PLAN_REVISION", "PAST_EVENT_REFRAME", "RESOURCE_REALLOCATION", "DECEPTION_OR_WITHHOLDING",
        "DEADLINE_RESPONSE", "RELATIONSHIP_REPAIR", "OTHER",
    },
    "trigger_type": {
        "NEW_INFORMATION", "PHYSICAL_EVENT", "CHARACTER_ACTION", "PRIOR_CONTRADICTION",
        "ATTEMPT_FAILURE", "DISCLOSURE", "DEADLINE_OR_RESOURCE", "RELATIONSHIP_EVENT", "OTHER",
    },
    "obstacle_type": {
        "NONE", "MISSING_INFORMATION", "RULE_OR_INSTITUTION", "CONFLICTING_INFORMATION",
        "CHARACTER_RESISTANCE", "RESOURCE_OR_TIME", "PHYSICAL_LIMIT", "INTERNAL_CONFLICT", "OTHER",
    },
    "state_change_target": {
        "FACT", "BELIEF", "PLAN", "GOAL", "RELATIONSHIP", "RESOURCE",
        "STATUS_OR_POWER", "THREAT_OR_STAKES", "OTHER",
    },
    "information_delta": {
        "UNKNOWN_TO_KNOWN", "BELIEF_REVISED", "OPTIONS_NARROWED", "OPTIONS_EXPANDED",
        "PLAN_FORMED", "PLAN_FAILED", "RELATIONSHIP_SHIFT", "RESOURCE_CHANGED",
        "STATUS_CHANGED", "STAKES_ESCALATED", "NO_DURABLE_CHANGE", "OTHER",
    },
    "causal_relation": {
        "CONFIRMATION", "CONTRADICTION", "ABSENCE_OR_GAP", "REINTERPRETATION",
        "CAUSE_REVEALED", "TRADEOFF", "ACTION_CONSEQUENCE", "OTHER",
    },
    "turn_type": {
        "FACT_CONFIRMED", "ASSUMPTION_REVISED", "PAST_RECONTEXTUALIZED",
        "NEW_CAUSAL_LINK", "PLAN_FAILS", "CHARACTER_CHOICE", "SCOPE_EXPANDS",
        "POWER_SHIFT", "RELATIONSHIP_REDEFINED", "OTHER",
    },
    "resolution_type": {
        "NEXT_ACTION_TARGET", "ACTION_COMMITMENT", "PARTIAL_ANSWER", "RELATIONSHIP_SHIFT",
        "FAILED_ATTEMPT", "IMMEDIATE_ACTION", "TEMPORARY_SETTLEMENT", "NO_CLEAR_RESOLUTION", "OTHER",
    },
    "hook_type": {
        "PENDING_INFORMATION", "UNRESOLVED_IDENTITY", "NEW_EVENT", "IMMINENT_ACTION",
        "CHARACTER_DECISION", "CONSEQUENCE", "OPEN_CAUSAL_QUESTION", "RELATIONSHIP_TENSION", "OTHER",
    },
    "interaction_pattern": {
        "SOLO_ACTION", "COOPERATIVE_ACTION", "METHOD_DISAGREEMENT", "INFORMATION_WITHHOLDING",
        "NEGOTIATION", "CONFRONTATION", "SUPPORT_WITH_RESERVATION", "ROLE_REVERSAL",
        "AVOIDANCE", "OTHER",
    },
    "relationship_change": {
        "NONE", "TRUST_INCREASE", "TRUST_DECREASE", "CONDITIONAL_COOPERATION",
        "CONFLICT_ESCALATION", "DEPENDENCE_INCREASE", "BOUNDARY_RENEGOTIATED",
        "RECONCILIATION", "SEPARATION", "OTHER",
    },
    "scene_functions": {
        "DISCOVERY", "DISCLOSURE", "VERIFICATION", "PLANNING", "ACCESS_ATTEMPT", "CONVERSATION", "INTERVIEW",
        "CONFRONTATION", "EXPERIMENT", "ACTION", "CONSEQUENCE", "REFLECTION",
        "NEGOTIATION", "COMMITMENT", "OTHER",
    },
}

SIGNATURE_FIELDS = (
    "macro_move", "mechanism_operations", "mechanism_summary", "trigger_type",
    "obstacle_type", "state_change_target", "information_delta", "causal_relation",
    "turn_type", "resolution_type", "hook_type", "interaction_pattern",
    "relationship_change", "scene_functions", "expressive_motifs", "evidence",
)

CORE_DIMENSION_WEIGHTS = {
    "mechanism_operations": 0.18,
    "state_change_target": 0.12,
    "information_delta": 0.20,
    "causal_relation": 0.12,
    "turn_type": 0.18,
    "resolution_type": 0.12,
    "obstacle_type": 0.05,
    "interaction_pattern": 0.03,
}

CONTEXT_DIMENSIONS = ("trigger_type", "hook_type", "relationship_change")
NON_INFORMATIVE_VALUES = {"NONE", "OTHER", "NO_DURABLE_CHANGE", "NO_CLEAR_RESOLUTION"}


def build_signature_prompt(*, script: str, macro_move: str, move_evidence: str) -> str:
    allowed = "\n".join(f"- {field}: {', '.join(sorted(values))}" for field, values in CONTROLLED_VALUES.items())
    return f"""你是盲态叙事结构标注器。只分析当前剧本实际采用的推进机制，不评价质量，不依据题材猜测。一级 macro_move 已由另一标注器给出；你的任务是区分同一大类内部的具体结构。只输出完整合法 JSON 对象，不要 Markdown。

已标注 macro_move：{macro_move}
已标注依据：{move_evidence}

受控值：
{allowed}

输出格式：
{{
  "macro_move": "原样复制已标注 macro_move",
  "mechanism_operations": ["选择1至2个受控值，按主次排序"],
  "mechanism_summary": "用不含人物专名的单句描述：什么触发、人物采取什么操作、哪个状态如何改变，不超过60字",
  "trigger_type": "一个受控值",
  "obstacle_type": "一个受控值",
  "state_change_target": "被改变的核心状态对象，一个受控值",
  "information_delta": "本集前后最重要的状态变化，一个受控值",
  "causal_relation": "促成变化的核心关系，一个受控值",
  "turn_type": "一个受控值",
  "resolution_type": "一个受控值",
  "hook_type": "一个受控值",
  "interaction_pattern": "一个受控值",
  "relationship_change": "一个受控值",
  "scene_functions": ["按剧本实际顺序选择2至6个受控值，可重复"],
  "expressive_motifs": ["最多3个短语，只记录明显反复承担叙事功能的动作或台词模式；没有则空数组"],
  "evidence": "不超过80字，说明结构判断依据"
}}

规则：
- 不要根据悬疑、家庭、职场、爱情等题材惯例自动选择标签，只依据当前剧本真正改变状态的行动。
- mechanism_operations 描述改变剧情状态的操作，不是道具、职业流程或场景名称。
- state_change_target 指向被改变的核心对象；information_delta 描述该对象前后发生了什么变化。
- turn_type 与 resolution_type 优先标注观众实际感受到的主要变化，不把普通题材流程误当成核心转折。
- scene_functions 保留顺序，用于区分场景组织。
- expressive_motifs 使用可跨集比较的抽象短语，不写具体台词全文。

当前剧本：
{script}"""


def validate_signature(value: dict[str, Any], macro_move: str) -> None:
    missing = set(SIGNATURE_FIELDS) - set(value)
    if missing:
        raise ValueError(f"签名缺少字段：{sorted(missing)}")
    if value["macro_move"] != macro_move:
        raise ValueError("macro_move 与既有标注不一致")
    operations = value["mechanism_operations"]
    if not isinstance(operations, list) or not 1 <= len(operations) <= 2:
        raise ValueError("mechanism_operations 必须包含 1-2 项")
    for field, allowed in CONTROLLED_VALUES.items():
        current = value[field]
        values = current if isinstance(current, list) else [current]
        if field == "scene_functions" and not 2 <= len(values) <= 6:
            raise ValueError("scene_functions 必须包含 2-6 项")
        if any(item not in allowed for item in values):
            raise ValueError(f"{field} 包含非受控值")
    for field, maximum in (("mechanism_summary", 120), ("evidence", 160)):
        if not isinstance(value[field], str) or not value[field].strip() or len(value[field]) > maximum:
            raise ValueError(f"{field} 为空或过长")
    motifs = value["expressive_motifs"]
    if not isinstance(motifs, list) or len(motifs) > 3 or any(not isinstance(item, str) or len(item) > 40 for item in motifs):
        raise ValueError("expressive_motifs 格式错误")


def entropy(values: Iterable[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def effective_count(values: Iterable[str]) -> float:
    return 2 ** entropy(values)


def _jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _lcs_similarity(left: list[str], right: list[str]) -> float:
    if not left and not right:
        return 1.0
    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, a in enumerate(left, start=1):
        for j, b in enumerate(right, start=1):
            table[i][j] = table[i - 1][j - 1] + 1 if a == b else max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1] / max(len(left), len(right))


def _informative_match(field: str, left: Any, right: Any) -> tuple[float, bool]:
    if field == "mechanism_operations":
        informative_left = [item for item in left if item not in NON_INFORMATIVE_VALUES]
        informative_right = [item for item in right if item not in NON_INFORMATIVE_VALUES]
        return _jaccard(informative_left, informative_right), bool(informative_left or informative_right)
    if left in NON_INFORMATIVE_VALUES and right in NON_INFORMATIVE_VALUES:
        return 0.0, False
    return float(left == right), True


def structured_similarity(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    parts: dict[str, Any] = {}
    weighted_sum = 0.0
    active_weight = 0.0
    for field, weight in CORE_DIMENSION_WEIGHTS.items():
        score, informative = _informative_match(field, left[field], right[field])
        parts[field] = score
        if informative:
            weighted_sum += score * weight
            active_weight += weight
    core = weighted_sum / active_weight if active_weight else 0.0
    parts["scene_functions"] = _lcs_similarity(left["scene_functions"], right["scene_functions"])
    parts["expressive_motifs"] = _jaccard(left["expressive_motifs"], right["expressive_motifs"])
    for field in CONTEXT_DIMENSIONS:
        score, informative = _informative_match(field, left[field], right[field])
        parts[field] = score if informative else None
    decisive_match = all(left[field] == right[field] for field in ("information_delta", "turn_type", "resolution_type"))
    mechanism_overlap = parts["mechanism_operations"] >= 0.5
    high_repetition = core >= 0.75 and decisive_match and mechanism_overlap
    partial_skeleton_reuse = core >= 0.55 and not high_repetition
    parts["core_structural_similarity"] = round(core, 6)
    parts["scene_sequence_similarity"] = round(parts["scene_functions"], 6)
    parts["expressive_motif_similarity"] = round(parts["expressive_motifs"], 6)
    parts["macro_move_match"] = float(left["macro_move"] == right["macro_move"])
    parts["decisive_change_match"] = decisive_match
    parts["repetition_class"] = "HIGH_REPETITION" if high_repetition else "PARTIAL_SKELETON_REUSE" if partial_skeleton_reuse else "DISTINCT"
    return parts


# ── 邻集边界场景重叠检测（无需 API，轻量级）──────────────────────────
# 检测"上一集结尾场景 = 下一集开头场景"的边界重复现象。
# 这在长篇连载中常见：模型忘了已写过某场景，在新集开头又重写了一遍。

def _extract_scene_meta(script: str) -> list[dict[str, Any]]:
    """从剧本中提取每个场景的元信息（地点、角色、时间）。

    支持三种格式：
      drama:    \"N-M. Location Time 人物：A、B\"
      custom:   \"**N-M. Location Time**\" / \"人物：A、B\"
      D_diverse: \"场景N  Location/Time\"
    """
    scenes: list[dict[str, Any]] = []

    # 先匹配场景头行，捕获整行内容，再细分地点/时间/角色
    header_pattern = re.compile(
        r'(?:^|\n)(?:\*{0,4}#{0,4}\s*)?'
        r'(\d+-\d+\.|场景[一二三四五六七八九十\d]+)\s*'
        r'([^\n]+)',   # 整行内容：地点 + 可能的时间 + 可能的人物
        re.MULTILINE,
    )
    for m in header_pattern.finditer(script):
        line = m.group(2).strip()
        # 解析角色：从行末或下一行的 "人物：..." 提取
        characters: set[str] = set()
        char_match = re.search(r'人物[：:]\s*([^\n]+)', line)
        if char_match:
            characters.update(re.findall(r'[\u4e00-\u9fff]{2,3}', char_match.group(1)))
            # 去掉人物部分，剩余是地点+时间
            location_part = line[:char_match.start()].strip()
        else:
            # 检查下一行
            rest = script[m.end():m.end() + 150]
            next_char = re.search(r'人物[：:]\s*([^\n]+)', rest)
            if next_char and next_char.start() < 60:  # 下一行但足够近
                characters.update(re.findall(r'[\u4e00-\u9fff]{2,3}', next_char.group(1)))
            location_part = line

        # 解析地点：去掉末尾的时间词
        location = re.sub(r'\s+[日夜晨午傍晚]{1,2}$', '', location_part).strip()
        # 去掉末尾的 "内" "外" "面" 等方位词
        location = re.sub(r'[内外面]$', '', location).strip()
        # D_diverse 格式：地点/时间 → 取 / 之前
        if '/' in location and not any(c in location for c in '人物场景'):
            location = location.split('/')[0].strip()

        if location:
            scenes.append({
                "location": location,
                "characters": characters,
                "pos": m.start(),
            })
    return scenes


def _extract_names_from_snippet(text: str) -> set[str]:
    """从短文段中提取角色名（用于边界区域的补充提取）。"""
    names: set[str] = set()
    # 人物：标签
    for m in re.finditer(r'人物[：:]\s*([^\n]+)', text):
        for name in re.findall(r'[\u4e00-\u9fff]{2,3}', m.group(1)):
            names.add(name)
    # 对话前缀
    for pat in [
        r'(?:^|\n)([\u4e00-\u9fff]{2,3})[（(]',  # 角色名(
        r'\*\*([\u4e00-\u9fff]{2,3})\*\*',       # **角色名**
        r'(?:^|\n)([\u4e00-\u9fff]{2,3})\n——',    # 角色名\n——
    ]:
        for m in re.finditer(pat, text):
            names.add(m.group(1))
    return names


def _extract_boundary_dialogue(script: str, is_tail: bool, window: int = 800) -> list[str]:
    """从剧本边界区域提取纯对话内容（去掉角色名和动作描述）。"""
    region = script[-window:] if is_tail else script[:window]
    lines = []
    for line in region.split("\n"):
        line = line.strip()
        if not line or len(line) < 4:
            continue
        # 跳过非对话行
        if line.startswith(("△", "（", "---", "场景", "人物：", "**", "####", "#")):
            continue
        if re.match(r"\d+-\d+\.", line):
            continue
        # 提取对话内容：去掉角色名前缀和动作描述
        # 模式：角色名（动作）：台词 / 角色名——台词 / 角色名：台词
        content = re.sub(
            r"^[\u4e00-\u9fff]{2,3}[（(][^)]*[)）][：:]?\s*", "", line,
        )
        content = re.sub(r"^[\u4e00-\u9fff]{2,3}[：:]\s*", "", content)
        content = re.sub(r"^——\s*", "", content)
        if content and len(content) >= 4:
            lines.append(content)
    return lines


def _dialogue_similarity(lines_a: list[str], lines_b: list[str]) -> float:
    """计算两组对话行的内容相似度（CJK 2-gram Jaccard）。"""
    if not lines_a or not lines_b:
        return 0.0
    text_a = "".join(lines_a)
    text_b = "".join(lines_b)
    ngrams_a = {text_a[i:i + 2] for i in range(len(text_a) - 1)}
    ngrams_b = {text_b[i:i + 2] for i in range(len(text_b) - 1)}
    if not ngrams_a or not ngrams_b:
        return 0.0
    return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)


def boundary_overlap(ep_i: str, ep_j: str) -> dict[str, Any]:
    """检测两集边界是否在描述同一个场景。

    三层信号加权：
      1. 地点匹配 (0.35)
      2. 角色重叠 (0.30)
      3. 对话内容相似度 (0.35) — 捕获"说的话都差不多"的即视感
    """
    # ── 地点 & 角色 ──
    scenes_i = _extract_scene_meta(ep_i)
    scenes_j = _extract_scene_meta(ep_j)

    last_scene = scenes_i[-1] if scenes_i else {}
    first_scene = scenes_j[0] if scenes_j else {}

    tail_loc = last_scene.get("location", "")
    head_loc = first_scene.get("location", "")
    tail_chars = last_scene.get("characters", set())
    head_chars = first_scene.get("characters", set())

    # 从边界片段补充角色名
    tail = ep_i[-600:] if len(ep_i) > 600 else ep_i
    head = ep_j[:600] if len(ep_j) > 600 else ep_j
    tail_chars |= _extract_names_from_snippet(tail)
    head_chars |= _extract_names_from_snippet(head)

    # 地点匹配
    if tail_loc and head_loc:
        tail_clean = re.sub(r"[的]", "", tail_loc)
        head_clean = re.sub(r"[的]", "", head_loc)
        if tail_clean == head_clean:
            loc_match = 1.0
        elif tail_clean in head_clean or head_clean in tail_clean:
            loc_match = 0.8
        else:
            t2 = {tail_clean[i:i + 2] for i in range(len(tail_clean) - 1)}
            h2 = {head_clean[i:i + 2] for i in range(len(head_clean) - 1)}
            loc_match = len(t2 & h2) / max(len(t2 | h2), 1) if t2 and h2 else 0.0
    else:
        loc_match = 0.0

    # 角色重叠
    if tail_chars and head_chars:
        char_jaccard = len(tail_chars & head_chars) / len(tail_chars | head_chars)
    else:
        char_jaccard = 0.0

    # ── 对话内容相似度 ──
    tail_dialog = _extract_boundary_dialogue(ep_i, is_tail=True)
    head_dialog = _extract_boundary_dialogue(ep_j, is_tail=False)
    dialog_sim = _dialogue_similarity(tail_dialog, head_dialog)
    # 归一化：典型 drama 对话 Jaccard 在 0.02-0.15 范围，映射到 [0,1]
    dialog_norm = min(1.0, dialog_sim / 0.10)

    # ── 综合分数 ──
    score = 0.35 * loc_match + 0.30 * char_jaccard + 0.35 * dialog_norm

    return {
        "character_overlap": round(char_jaccard, 4),
        "location_overlap": round(loc_match, 4),
        "dialogue_similarity": round(dialog_sim, 4),
        "dialogue_similarity_norm": round(dialog_norm, 4),
        "boundary_overlap_score": round(score, 4),
        "shared_characters": sorted(tail_chars & head_chars),
        "tail_location": tail_loc,
        "head_location": head_loc,
        "tail_n_scenes": len(scenes_i),
        "head_n_scenes": len(scenes_j),
        "tail_dialog_lines": len(tail_dialog),
        "head_dialog_lines": len(head_dialog),
    }


def compute_all_boundaries(scripts: list[str]) -> list[dict[str, Any]]:
    """对一条轨迹的所有集计算逐邻集边界的重叠分数。"""
    results = []
    for i in range(len(scripts) - 1):
        overlap = boundary_overlap(scripts[i], scripts[i + 1])
        overlap["episode_pair"] = f"E{i+1:02d}→E{i+2:02d}"
        results.append(overlap)
    return results
