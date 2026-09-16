from __future__ import annotations

import copy
import math
import random
import re
from collections import Counter
from typing import Any

from .io_utils import stable_hash
from .schema import POOL_TAG_BY_CONDITION


def _features(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    return {compact[index:index + 2] for index in range(max(0, len(compact) - 1))}


def _similarity(left: str, right: str) -> float:
    a, b = _features(left), _features(right)
    return len(a & b) / len(a | b) if a and b else 0.0


class MemoryPool:
    def __init__(self, cards: list[dict[str, Any]], *, condition: str, story_id: str) -> None:
        tag = POOL_TAG_BY_CONDITION.get(condition)
        self.condition = condition
        self.story_id = story_id
        self.cards = {
            card["memory_id"]: copy.deepcopy(card)
            for card in cards
            if card["story_id"] == story_id and tag in card["pool_tags"]
        } if tag else {}
        self.exposures: Counter[str] = Counter()
        # 逐轨迹近期使用轨迹（cluster_id 序列）；因 pool 按轨迹重建而天然逐轨迹隔离。
        # 用于 RecentReuse（近因惩罚）与 Novelty(H)（相对整条故事历史簇频率）。
        self.used_clusters: list[str] = []

    def _recency_weighted_use(self, cluster_id: str, *, decay: float = 0.6) -> float:
        """近期使用强度：越近权重越高（指数衰减）。"""
        total = 0.0
        for index, used in enumerate(reversed(self.used_clusters)):
            if used == cluster_id:
                total += decay ** index
        return total

    def _cluster_freq(self, cluster_id: str) -> float:
        if not self.used_clusters:
            return 0.0
        return self.used_clusters.count(cluster_id) / len(self.used_clusters)

    def retrieve(self, query: str, *, settings: dict[str, float], rng_seed: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        similarity_weight = float(settings.get("similarity_weight", 1.0))
        quality_weight = float(settings.get("quality_weight", 0.15))
        exposure_weight = float(settings.get("exposure_weight", 0.05))
        top_k = int(settings.get("top_k", 3))
        min_similarity = float(settings.get("min_similarity", 0.0))
        cluster_diversity_cap = int(settings.get("cluster_diversity_cap", top_k))
        selection_mode = str(settings.get("selection_mode", "top_k"))
        recency_weight = float(settings.get("recency_weight", 0.0))
        redundancy_weight = float(settings.get("redundancy_weight", 0.0))
        novelty_weight = float(settings.get("novelty_weight", 0.0))
        min_relevance = float(settings.get("min_relevance", 0.0))
        reject_if_no_novelty = bool(settings.get("reject_if_no_novelty", False))
        exclude_writeback = bool(settings.get("exclude_writeback", False))
        recency_decay = float(settings.get("recency_decay", 0.6))

        rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

        # 候选预筛选：排除写回卡（D_no_card）、相关性门槛、可检索源类型
        candidates: list[dict[str, Any]] = []
        for memory_id, card in self.cards.items():
            if exclude_writeback and card.get("source_type") == "writeback":
                continue
            searchable = " ".join(str(card.get(key, "")) for key in ("applicable_situation", "narrative_strategy", "expected_effect"))
            similarity = _similarity(query, searchable)
            quality = float(card["quality_score"])
            exposure_boost = exposure_weight * math.log1p(self.exposures[card["cluster_id"]])
            recent_use = self._recency_weighted_use(card["cluster_id"], decay=recency_decay)
            novelty = 1.0 - self._cluster_freq(card["cluster_id"])
            below_relevance = similarity < min_relevance or similarity < min_similarity
            candidates.append({
                "memory_id": memory_id,
                "cluster_id": card["cluster_id"],
                "move_id": card["move_id"],
                "similarity_raw": round(similarity, 6),
                "quality_raw": round(quality, 6),
                "exposure_raw": round(exposure_boost, 6),
                "recent_use_raw": round(recent_use, 6),
                "novelty_raw": round(novelty, 6),
                "cluster_exposure_before": self.exposures[card["cluster_id"]],
                "base_score": round(similarity_weight * similarity + quality_weight * quality + exposure_boost, 6),
                "recent_use": recent_use,
                "novelty": novelty,
                "below_min_similarity": below_relevance,
            })

        def _field_text(c: dict[str, Any]) -> str:
            return " ".join(str(c.get(k, "")) for k in ("applicable_situation", "narrative_strategy", "expected_effect"))

        # 选中 / 丢弃：先收集，落账在 reject 决策之后（避免误增曝光）
        selected_list: list[dict[str, Any]] = []
        dropped_list: list[tuple[dict[str, Any], str]] = []

        def _drop(cand: dict[str, Any], reason: str) -> None:
            dropped_list.append((cand, reason))

        if selection_mode == "random_relevance":
            eligible = [c for c in candidates if not c["below_min_similarity"]]
            if not eligible:
                for cand in candidates:
                    _drop(cand, "no_eligible_candidates")
            else:
                chosen = rng.sample(eligible, min(top_k, len(eligible)))
                for cand in chosen:
                    cand["rank"] = len(selected_list) + 1
                    selected_list.append(cand)
                for cand in sorted(eligible, key=lambda item: -item["similarity_raw"]):
                    if cand not in selected_list:
                        _drop(cand, "below_top_k")
                for cand in candidates:
                    if cand["below_min_similarity"]:
                        _drop(cand, "below_min_similarity")
        else:
            scored = []
            for cand in candidates:
                score = cand["base_score"] - recency_weight * cand["recent_use"] + novelty_weight * cand["novelty"]
                if selection_mode == "diverse" and selected_list:
                    max_sim = max((_similarity(_field_text(cand), _field_text(s)) for s in selected_list), default=0.0)
                    score -= redundancy_weight * max_sim
                    cand["redundancy_raw"] = round(max_sim, 6)
                cand["score"] = round(score, 6)
                scored.append(cand)
            scored.sort(key=lambda item: (-item["score"], item["memory_id"]))
            seen_per_cluster: dict[str, int] = {}
            for cand in scored:
                if cand["below_min_similarity"]:
                    _drop(cand, "below_min_similarity")
                    continue
                if seen_per_cluster.get(cand["cluster_id"], 0) >= cluster_diversity_cap:
                    _drop(cand, "cluster_diversity_cap")
                    continue
                if len(selected_list) >= top_k:
                    _drop(cand, "below_top_k")
                    continue
                seen_per_cluster[cand["cluster_id"]] = seen_per_cluster.get(cand["cluster_id"], 0) + 1
                cand["rank"] = len(selected_list) + 1
                selected_list.append(cand)

        # reject_if_no_novelty：所选卡既无相关性又无新方向（相对 H）则不塞策略卡
        if reject_if_no_novelty and selected_list:
            has_rel = any(not c["below_min_similarity"] for c in selected_list)
            has_novel = any(c["novelty"] >= 0.5 and not c["below_min_similarity"] for c in selected_list)
            if not (has_rel and has_novel):
                for cand in selected_list:
                    _drop(cand, "rejected_no_novelty")
                selected_list = []

        # 落账：选中者增加曝光并计入近期使用历史
        events: list[dict[str, Any]] = []
        for cand in selected_list:
            self.exposures[cand["cluster_id"]] += 1
            self.used_clusters.append(cand["cluster_id"])
            events.append({
                "memory_id": cand["memory_id"],
                "cluster_id": cand["cluster_id"],
                "move_id": cand["move_id"],
                "rank": cand.get("rank"),
                "selected": True,
                "drop_reason": None,
                "score": cand.get("score", cand["base_score"]),
                "similarity_raw": cand["similarity_raw"],
                "quality_raw": cand["quality_raw"],
                "exposure_raw": cand["exposure_raw"],
                "recent_use_raw": cand["recent_use_raw"],
                "novelty_raw": cand["novelty_raw"],
                "redundancy_raw": cand.get("redundancy_raw", 0.0),
                "cluster_exposure_before": cand["cluster_exposure_before"],
                "cluster_exposure_after": self.exposures[cand["cluster_id"]],
                "selection_mode": selection_mode,
            })
        for cand, reason in dropped_list:
            events.append({
                "memory_id": cand["memory_id"],
                "cluster_id": cand["cluster_id"],
                "move_id": cand["move_id"],
                "rank": None,
                "selected": False,
                "drop_reason": reason,
                "score": cand.get("score", cand["base_score"]),
                "similarity_raw": cand["similarity_raw"],
                "quality_raw": cand["quality_raw"],
                "exposure_raw": cand["exposure_raw"],
                "recent_use_raw": cand["recent_use_raw"],
                "novelty_raw": cand["novelty_raw"],
                "redundancy_raw": cand.get("redundancy_raw", 0.0),
                "cluster_exposure_before": cand["cluster_exposure_before"],
                "cluster_exposure_after": self.exposures[cand["cluster_id"]],
                "selection_mode": selection_mode,
            })
        cards = [copy.deepcopy(self.cards[cand["memory_id"]]) for cand in selected_list]
        return cards, events

    def append_writeback(self, *, job_id: str, episode_id: str, candidate: dict[str, Any], parent_memory_ids: list[str]) -> dict[str, Any]:
        memory_id = f"WB_{job_id}_{stable_hash(candidate)}"
        card = {
            "memory_id": memory_id,
            "story_id": self.story_id,
            "pool_tags": ["runtime_writeback"],
            "quality_score": float(candidate["quality_score"]),
            "cluster_id": candidate["move_id"],
            "move_id": candidate["move_id"],
            "applicable_situation": candidate["applicable_situation"],
            "narrative_strategy": candidate["narrative_strategy"],
            "expected_effect": candidate["expected_effect"],
            "risks": list(candidate.get("risks", [])),
            "source_type": "writeback",
            "source_episode": episode_id,
            "source_job_id": job_id,
            "parent_memory_ids": list(parent_memory_ids),
        }
        self.cards[memory_id] = card
        return copy.deepcopy(card)

    def snapshot(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(self.cards[key]) for key in sorted(self.cards)]
