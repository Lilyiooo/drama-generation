from __future__ import annotations

import copy
import random
import re
from typing import Any

from .io_utils import stable_hash


VALID_FUNCTIONS = {
    "INITIATE", "PROGRESS", "TEST", "CONSEQUENCE",
    "PRESERVE", "CONSOLIDATE", "RECOVER", "CLOSE",
}
VALID_PHASES = {"setup", "expansion", "escalation", "closure"}
VALID_HARM_FLAGS = {
    "STATE_CONFLICT", "ANCHOR_VIOLATION", "CAUSAL_BREAK",
    "MECHANISM_REPETITION", "CARD_ECHO", "UNSUPPORTED_INVENTION",
}
ACTIVE_CANDIDATE_STATUSES = {"CANDIDATE", "TRIAL"}


def _features(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    return {compact[index:index + 2] for index in range(max(0, len(compact) - 1))}


def _similarity(left: str, right: str) -> float:
    a, b = _features(left), _features(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def _card_text(card: dict[str, Any]) -> str:
    return " ".join([
        str(card.get("applicable_situation", "")),
        *[str(item) for item in card.get("operator_steps", [])],
        str(card.get("expected_state_change", "")),
    ])


class SelfEvolvingV4Pool:
    """逐轨迹隔离的 V4 经验卡生命周期。

    专家卡始终可检索；自生成卡先进入候选池，只能以单卡随机试用方式曝光，
    达到跨上下文成功阈值后才进入正式检索池。所有状态都可序列化恢复。
    """

    def __init__(self, expert_cards: list[dict[str, Any]], settings: dict[str, Any]) -> None:
        self.expert_cards = {
            card["memory_id"]: copy.deepcopy(card)
            for card in expert_cards
        }
        self.candidate_cards: dict[str, dict[str, Any]] = {}
        self.promoted_cards: dict[str, dict[str, Any]] = {}
        self.card_stats: dict[str, dict[str, Any]] = {}
        self.settings = copy.deepcopy(settings)

    def searchable_cards(self) -> list[dict[str, Any]]:
        cards = [*self.expert_cards.values(), *self.promoted_cards.values()]
        return [copy.deepcopy(card) for card in cards if card.get("lifecycle_status") not in {"SUSPENDED", "RETIRED"}]

    def trial_cards(self) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(card)
            for memory_id, card in self.candidate_cards.items()
            if self.card_stats.get(memory_id, {}).get("lifecycle_status") in ACTIVE_CANDIDATE_STATUSES
        ]

    def get_card(self, memory_id: str) -> dict[str, Any] | None:
        card = self.candidate_cards.get(memory_id) or self.promoted_cards.get(memory_id) or self.expert_cards.get(memory_id)
        return copy.deepcopy(card) if card else None

    def _duplicate(self, card: dict[str, Any]) -> tuple[str | None, float]:
        threshold = float(self.settings.get("duplicate_similarity", 0.72))
        best_id: str | None = None
        best_score = 0.0
        pools = (self.expert_cards, self.candidate_cards, self.promoted_cards)
        for pool in pools:
            for memory_id, existing in pool.items():
                if existing.get("primary_move_id") != card.get("primary_move_id"):
                    continue
                score = _similarity(_card_text(card), _card_text(existing))
                if score > best_score:
                    best_id, best_score = memory_id, score
        return (best_id, best_score) if best_score >= threshold else (None, best_score)

    def add_candidate(
        self,
        *,
        candidate: dict[str, Any],
        job_id: str,
        episode_id: str,
        context_signature: str,
        state_tags: set[str],
        episode_function: str,
        phase: str,
        protected_invariants: list[str],
        parent_memory_ids: list[str],
        story_terms: set[str],
        mode: str,
    ) -> dict[str, Any]:
        confidence = candidate.get("confidence")
        quality = candidate.get("quality_score")
        minimum_confidence = float(self.settings.get("minimum_confidence", 0.70))
        minimum_quality = float(self.settings.get("minimum_quality", 0.70))
        if not isinstance(confidence, (int, float)) or confidence < minimum_confidence:
            return {"action": "rejected", "reason": "low_confidence", "card": None}
        if not isinstance(quality, (int, float)) or quality < minimum_quality:
            return {"action": "rejected", "reason": "low_quality", "card": None}
        if candidate.get("card_type", "STRATEGY") != "STRATEGY":
            return {"action": "rejected", "reason": "unsupported_card_type", "card": None}

        move_id = str(candidate.get("primary_move_id") or candidate.get("move_id") or "")
        situation = str(candidate.get("applicable_situation", "")).strip()
        operator_steps = candidate.get("operator_steps")
        if not isinstance(operator_steps, list):
            legacy_operator = str(candidate.get("narrative_strategy", "")).strip()
            operator_steps = [legacy_operator] if legacy_operator else []
        operator_steps = [str(item).strip() for item in operator_steps if str(item).strip()][:3]
        expected_change = str(candidate.get("expected_state_change") or candidate.get("expected_effect") or "").strip()
        if not move_id or not situation or not operator_steps or not expected_change:
            return {"action": "rejected", "reason": "incomplete_operator", "card": None}

        transferable_text = " ".join([situation, *operator_steps, expected_change])
        leaked_terms = sorted(term for term in story_terms if len(term) >= 2 and term in transferable_text)
        if leaked_terms:
            return {"action": "rejected", "reason": "story_specific_terms", "details": leaked_terms, "card": None}

        requested_tags = candidate.get("applicable_state_features", [])
        if not isinstance(requested_tags, list):
            requested_tags = []
        matched_tags = [str(tag) for tag in requested_tags if str(tag) in state_tags]
        matched_tags = list(dict.fromkeys(matched_tags))[:3]
        if len(matched_tags) < 2:
            return {"action": "rejected", "reason": "insufficient_state_signature", "card": None}

        functions = candidate.get("episode_functions", [])
        if not isinstance(functions, list):
            functions = []
        functions = [str(item).upper() for item in functions if str(item).upper() in VALID_FUNCTIONS]
        if episode_function in VALID_FUNCTIONS and episode_function not in functions:
            functions.insert(0, episode_function)
        functions = list(dict.fromkeys(functions))[:3]
        phases = candidate.get("applicable_phases", [])
        if not isinstance(phases, list):
            phases = []
        phases = [str(item).lower() for item in phases if str(item).lower() in VALID_PHASES]
        if phase in VALID_PHASES and phase not in phases:
            phases.insert(0, phase)
        phases = list(dict.fromkeys(phases))[:3]

        mechanism = re.sub(r"[^A-Z0-9_]+", "_", str(candidate.get("mechanism_family", "")).upper()).strip("_")
        if not mechanism:
            mechanism = f"SELF_{move_id}_{stable_hash(transferable_text)[:8].upper()}"
        memory_id = f"WBV4_{job_id}_{stable_hash(candidate)}"
        card = {
            "memory_id": memory_id,
            "version": "4.1.0-self-evolving",
            "layer": "strategy",
            "status": "promoted" if mode == "naive" else "candidate",
            "lifecycle_status": "PROMOTED" if mode == "naive" else "CANDIDATE",
            "primary_move_id": move_id,
            "secondary_move_ids": [],
            "mechanism_family": mechanism,
            "applicable_phases": phases or [phase],
            "requires_all": [],
            "requires_any": matched_tags,
            "forbids_any": [str(item) for item in candidate.get("inapplicable_state_features", []) if isinstance(item, str)][:3],
            "repair_triggers": [],
            "cooldown_episodes": int(candidate.get("cooldown_episodes", 2)),
            "applicable_situation": situation,
            "operator_steps": operator_steps,
            "expected_state_change": expected_change,
            "observable_evidence": [str(item) for item in candidate.get("observable_evidence", []) if isinstance(item, str)][:3],
            "risks": [str(item) for item in candidate.get("risks", []) if isinstance(item, str)][:2],
            "abort_conditions": [str(item) for item in candidate.get("abort_conditions", []) if isinstance(item, str)][:3],
            "prohibited_inventions": ["new_key_actor", "new_decisive_resource", "unestablished_secret"],
            "prohibited_outcomes": ["hard_anchor_violation", "cost_free_success"],
            "anti_copy_rules": [
                "use_only_current_context_entities",
                "instantiate_roles_not_examples",
                "do_not_repeat_card_wording_as_dialogue",
            ],
            "source_type": "self_generated_v4_naive" if mode == "naive" else "self_generated_v4",
            "source_episode": episode_id,
            "source_job_id": job_id,
            "source_context_signature": context_signature,
            "parent_memory_ids": list(parent_memory_ids),
            "curator_prior": None,
            "utility": {},
            "card_type": "STRATEGY",
            "retrieval_mode": "direct",
            "episode_functions": functions or [episode_function],
            "invariant_guards": list(protected_invariants),
            "conflicts_with_invariants": [
                str(item) for item in candidate.get("conflicts_with_invariants", [])
                if isinstance(item, str)
            ][:3],
            "risk_outcomes": [],
            "blocks_outcomes": [],
            "abstraction_level": "cross_story",
            "quality_score": float(quality),
        }

        duplicate_id, duplicate_score = self._duplicate(card)
        if duplicate_id:
            stats = self.card_stats.get(duplicate_id)
            if stats and duplicate_id in self.candidate_cards:
                contexts = stats.setdefault("source_contexts", [])
                if context_signature not in contexts:
                    contexts.append(context_signature)
                stats["observation_count"] = int(stats.get("observation_count", 1)) + 1
                return {
                    "action": "merged_observation",
                    "reason": "near_duplicate_candidate",
                    "duplicate_of": duplicate_id,
                    "similarity": round(duplicate_score, 6),
                    "card": copy.deepcopy(self.candidate_cards[duplicate_id]),
                }
            return {
                "action": "rejected",
                "reason": "covered_by_searchable_card",
                "duplicate_of": duplicate_id,
                "similarity": round(duplicate_score, 6),
                "card": None,
            }

        stats = {
            "memory_id": memory_id,
            "lifecycle_status": card["lifecycle_status"],
            "observation_count": 1,
            "exposure_count": 0,
            "adoption_count": 0,
            "success_count": 0,
            "harm_count": 0,
            "source_contexts": [context_signature],
            "exposure_contexts": [],
            "adoption_contexts": [],
            "success_contexts": [],
            "successful_functions": [],
            "successful_phases": [],
            "episodes_exposed": [],
            "episodes_adopted": [],
            "last_feedback": None,
        }
        self.card_stats[memory_id] = stats
        if mode == "naive":
            self.promoted_cards[memory_id] = card
            return {"action": "naive_promoted", "reason": "unconditional_append", "card": copy.deepcopy(card)}
        self.candidate_cards[memory_id] = card
        return {"action": "candidate_created", "reason": "passed_candidate_gate", "card": copy.deepcopy(card)}

    def should_trial(self, *, rng_seed: int) -> bool:
        probability = float(self.settings.get("trial_probability", 0.35))
        return random.Random(rng_seed).random() < probability

    def record_exposure(self, memory_id: str, *, episode_id: str, context_signature: str) -> dict[str, Any] | None:
        stats = self.card_stats.get(memory_id)
        card = self.candidate_cards.get(memory_id)
        if not stats or not card or stats.get("lifecycle_status") not in ACTIVE_CANDIDATE_STATUSES:
            return None
        stats["exposure_count"] += 1
        if context_signature not in stats["exposure_contexts"]:
            stats["exposure_contexts"].append(context_signature)
        stats["episodes_exposed"].append(episode_id)
        stats["lifecycle_status"] = "TRIAL"
        card["lifecycle_status"] = "TRIAL"
        card["status"] = "trial"
        return {"action": "candidate_exposed", "memory_id": memory_id, "stats": copy.deepcopy(stats)}

    def record_feedback(
        self,
        memory_id: str,
        *,
        feedback: dict[str, Any],
        episode_id: str,
        context_signature: str,
        experience_context: dict[str, Any],
        evidence_valid: bool,
    ) -> dict[str, Any] | None:
        stats = self.card_stats.get(memory_id)
        if not stats or memory_id not in self.candidate_cards:
            return None
        adopted = bool(feedback.get("adopted")) and evidence_valid
        harms = [str(item).upper() for item in feedback.get("harm_flags", []) if str(item).upper() in VALID_HARM_FLAGS]
        if adopted:
            stats["adoption_count"] += 1
            stats["episodes_adopted"].append(episode_id)
            if context_signature not in stats["adoption_contexts"]:
                stats["adoption_contexts"].append(context_signature)
        success = bool(
            adopted
            and feedback.get("expected_effect_achieved")
            and any(bool(feedback.get(field)) for field in ("state_supported", "anchor_supported", "causal_supported"))
            and not harms
        )
        if success:
            stats["success_count"] += 1
            if context_signature not in stats["success_contexts"]:
                stats["success_contexts"].append(context_signature)
            successful_functions = stats.setdefault("successful_functions", [])
            successful_phases = stats.setdefault("successful_phases", [])
            episode_function = str(experience_context.get("episode_function", ""))
            phase = str(experience_context.get("phase", ""))
            if episode_function in VALID_FUNCTIONS and episode_function not in successful_functions:
                successful_functions.append(episode_function)
            if phase in VALID_PHASES and phase not in successful_phases:
                successful_phases.append(phase)
        if harms:
            stats["harm_count"] += 1
            stats["lifecycle_status"] = "SUSPENDED"
            self.candidate_cards[memory_id]["lifecycle_status"] = "SUSPENDED"
            self.candidate_cards[memory_id]["status"] = "suspended"
        max_unadopted = int(self.settings.get("retire_after_unadopted_exposures", 5))
        if stats["exposure_count"] >= max_unadopted and stats["adoption_count"] == 0:
            stats["lifecycle_status"] = "RETIRED"
            self.candidate_cards[memory_id]["lifecycle_status"] = "RETIRED"
            self.candidate_cards[memory_id]["status"] = "retired"
        stats["last_feedback"] = {
            "episode_id": episode_id,
            "adopted": adopted,
            "success": success,
            "harm_flags": harms,
            "evidence_valid": evidence_valid,
        }
        return {
            "action": "candidate_feedback",
            "memory_id": memory_id,
            "adopted": adopted,
            "success": success,
            "harm_flags": harms,
            "stats": copy.deepcopy(stats),
        }

    def promote_ready(self) -> list[dict[str, Any]]:
        promoted: list[dict[str, Any]] = []
        min_exposure = int(self.settings.get("promotion_min_exposure", 3))
        min_adoption = int(self.settings.get("promotion_min_adoption", 2))
        min_success_rate = float(self.settings.get("promotion_min_success_rate", 0.65))
        min_contexts = int(self.settings.get("promotion_min_contexts", 2))
        for memory_id, card in list(self.candidate_cards.items()):
            stats = self.card_stats[memory_id]
            if stats.get("lifecycle_status") not in ACTIVE_CANDIDATE_STATUSES:
                continue
            adoption_count = int(stats["adoption_count"])
            success_rate = stats["success_count"] / adoption_count if adoption_count else 0.0
            if (
                stats["exposure_count"] < min_exposure
                or adoption_count < min_adoption
                or success_rate < min_success_rate
                or stats["harm_count"] != 0
                or len(stats["success_contexts"]) < min_contexts
            ):
                continue
            promoted_card = copy.deepcopy(card)
            promoted_card["lifecycle_status"] = "PROMOTED"
            promoted_card["status"] = "promoted"
            promoted_card["episode_functions"] = list(dict.fromkeys([
                *promoted_card.get("episode_functions", []),
                *stats.get("successful_functions", []),
            ]))
            promoted_card["applicable_phases"] = list(dict.fromkeys([
                *promoted_card.get("applicable_phases", []),
                *stats.get("successful_phases", []),
            ]))
            promoted_card["utility"] = {
                "exposure_count": stats["exposure_count"],
                "adoption_count": adoption_count,
                "adoption_rate": round(adoption_count / stats["exposure_count"], 6),
                "success_count": stats["success_count"],
                "success_rate": round(success_rate, 6),
                "harm_count": stats["harm_count"],
                "context_coverage": len(stats["success_contexts"]),
            }
            self.promoted_cards[memory_id] = promoted_card
            stats["lifecycle_status"] = "PROMOTED"
            del self.candidate_cards[memory_id]
            promoted.append(copy.deepcopy(promoted_card))
        return promoted

    def snapshot(self) -> dict[str, Any]:
        return {
            "candidate_cards": copy.deepcopy(self.candidate_cards),
            "promoted_cards": copy.deepcopy(self.promoted_cards),
            "card_stats": copy.deepcopy(self.card_stats),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.candidate_cards = copy.deepcopy(snapshot.get("candidate_cards", {}))
        self.promoted_cards = copy.deepcopy(snapshot.get("promoted_cards", {}))
        self.card_stats = copy.deepcopy(snapshot.get("card_stats", {}))
