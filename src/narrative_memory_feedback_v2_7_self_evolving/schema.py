from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_json

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STATE_FIELDS = (
    "character_state", "relationship_state", "known_information",
    "unknown_information", "confirmed_facts", "unresolved_threads",
    "resources_and_evidence", "current_goals", "timeline",
)
POOL_TAG_BY_CONDITION = {
    "B_high_diverse_frozen": "high_diverse",
    "C_high_homogeneous_frozen": "high_homogeneous",
    "D_high_diverse_append": "high_diverse",
    "E_high_homogeneous_append": "high_homogeneous",
    "F_low_append": "low",
    # 检索干预消融条件：复用同一高多样冻结池，只改变检索打分/重排逻辑
    "R0_current": "high_diverse",
    "R1_no_exposure": "high_diverse",
    "R3_controlled": "high_diverse",
    # 双记忆实验新条件：状态-策略解耦 + 多样性感知检索
    "D_diverse": "high_diverse",
    "D_random": "high_diverse",
    "D_no_card": "high_diverse",
}


class ProtocolError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def validate_state(state: dict[str, Any], label: str = "state") -> None:
    require(isinstance(state, dict), f"{label} 必须是对象")
    require(set(state) == set(STATE_FIELDS), f"{label} 字段必须严格匹配状态 schema")
    for field in STATE_FIELDS:
        require(isinstance(state[field], (list, dict)), f"{label}.{field} 必须是数组或对象")


def validate_config(config: dict[str, Any]) -> None:
    required = {"schema_version", "protocol_id", "prompt_version", "seed", "episode_ids", "runs_per_condition", "conditions", "retrieval", "writeback", "experience_lifecycle", "branch_sampling", "generation", "extraction", "evaluation", "output_characters", "runtime"}
    require(required <= set(config), f"config 缺少字段：{sorted(required - set(config))}")
    require(config["runs_per_condition"] >= 1, "runs_per_condition 必须大于 0")
    require(config["retrieval"]["top_k"] >= 1, "retrieval.top_k 必须大于 0")
    expected = {"A_state_only", *POOL_TAG_BY_CONDITION}
    require(expected <= set(config["conditions"]), "核心实验必须包含 A-F 六组（允许额外对照条件）")
    require(config["conditions"]["A_state_only"]["retrieval_enabled"] is False, "A 组不能检索创作经验")


def validate_assets(config: dict[str, Any], stories: list[dict[str, Any]], plans: list[dict[str, Any]], memories: list[dict[str, Any]], taxonomy: list[dict[str, Any]]) -> None:
    validate_config(config)
    story_ids = {item["story_id"] for item in stories}
    require(len(story_ids) == len(stories), "story_id 必须唯一")
    for story in stories:
        validate_state(story["initial_state"], f"{story['story_id']}.initial_state")
    plan_keys = {(item["story_id"], item["episode_id"]) for item in plans}
    expected_plans = {(story_id, episode_id) for story_id in story_ids for episode_id in config["episode_ids"]}
    require(plan_keys == expected_plans, "每个故事必须覆盖全部配置集数")
    move_ids = {item["move_id"] for item in taxonomy}
    memory_ids: set[str] = set()
    tags_by_story = {story_id: set() for story_id in story_ids}
    for memory in memories:
        require(memory["memory_id"] not in memory_ids, "memory_id 必须唯一")
        memory_ids.add(memory["memory_id"])
        require(memory["story_id"] in story_ids, "经验引用未知故事")
        require(memory["move_id"] in move_ids and memory["cluster_id"] in move_ids, "经验引用未知 Move")
        require(0 <= memory["quality_score"] <= 1, "quality_score 必须在 [0,1]")
        require(bool(memory["pool_tags"]), "经验必须至少属于一个池")
        tags_by_story[memory["story_id"]].update(memory["pool_tags"])
    for story_id, tags in tags_by_story.items():
        require({"high_diverse", "high_homogeneous", "low"} <= tags, f"{story_id} 缺少实验经验池")


def load_protocol(root: Path = ROOT) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config = read_json(root / "config.json")
    stories = read_json(root / "data" / "stories.json")
    plans = read_json(root / "data" / "episode_plans.json")
    memories = read_json(root / "data" / "memory_pools.json")
    taxonomy = read_json(root / "data" / "narrative_move_taxonomy.json")
    validate_assets(config, stories, plans, memories, taxonomy)
    return config, stories, plans, memories, taxonomy
