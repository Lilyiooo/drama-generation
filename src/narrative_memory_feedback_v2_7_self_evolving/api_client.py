from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Protocol


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]], settings: dict[str, Any], *, seed: int | None, purpose: str) -> str:
        ...


class OpenAICompatibleClient:
    def __init__(self, *, api_key: str, base_url: str | None, model: str, disable_thinking: bool = False) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.disable_thinking = disable_thinking

    @classmethod
    def from_environment(cls, purpose: str) -> "OpenAICompatibleClient":
        prefixes = {"generation": "NMF_GENERATION", "extraction": "NMF_EXTRACTION", "evaluation": "NMF_EVALUATION"}
        prefix = prefixes.get(purpose, "NMF_EVALUATION")
        api_key = os.environ.get(f"{prefix}_API_KEY", "")
        base_url = os.environ.get(f"{prefix}_BASE_URL", "")
        model = os.environ.get(f"{prefix}_MODEL", "")
        if not api_key or not base_url:
            try:
                from agentdriver.llm_core.api_keys import OPENAI_API_KEY, OPENAI_BASE_URL
                api_key = api_key or OPENAI_API_KEY
                base_url = base_url or OPENAI_BASE_URL
            except ImportError:
                pass
        if not model:
            try:
                from experience_following import settings as _settings
                model = {
                    "generation": getattr(_settings, "GENERATION_MODEL", ""),
                    "extraction": getattr(_settings, "EXTRACTION_MODEL", ""),
                    "evaluation": getattr(_settings, "EVALUATOR_MODEL", ""),
                }.get(purpose, "")
            except ImportError:
                pass
        if not api_key or not model:
            raise RuntimeError(f"真实运行需要配置 {prefix}_API_KEY 和 {prefix}_MODEL")
        # kimi 系列默认开启思考链（按输出价收费、极易撑爆 max_tokens 导致空内容），
        # 生成/抽取场景一律关闭思考，直接出结果。
        disable_thinking = os.environ.get(f"{prefix}_DISABLE_THINKING", "true" if model.lower().startswith("kimi") else "false").lower() in {"1", "true", "yes"}
        return cls(api_key=api_key, base_url=base_url or None, model=model, disable_thinking=disable_thinking)

    def complete(self, messages: list[dict[str, str]], settings: dict[str, Any], *, seed: int | None, purpose: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("未安装 openai 依赖") from exc
        timeout_seconds = float(os.environ.get("NMF_API_TIMEOUT_SECONDS", "120"))
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout_seconds, max_retries=0)
        request: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": settings["temperature"], "max_tokens": settings["max_output_tokens"]}
        for key in ("top_p", "presence_penalty", "frequency_penalty"):
            if key in settings:
                request[key] = settings[key]
        if self.disable_thinking:
            request["extra_body"] = ({"thinking_enabled": False} if self.model.lower().startswith("kimi") else {"chat_template_kwargs": {"enable_thinking": False}})
        if seed is not None:
            request["seed"] = seed
        max_attempts = max(1, int(os.environ.get("NMF_API_MAX_ATTEMPTS", "5")))
        for attempt in range(max_attempts):
            try:
                response = client.chat.completions.create(**request)
            except Exception as exc:
                message = str(exc).lower()
                if "seed" in message and "seed" in request:
                    request.pop("seed", None)
                    continue
                retryable = any(marker in message for marker in ("状态错误", "'code': '4001'", '"code":"4001"', "timeout", "timed out", "rate limit", "ratelimit", "429", "限流", "connection", "internalservererror", "error code: 500", "502", "503", "504"))
                if not retryable or attempt + 1 >= max_attempts:
                    raise
                time.sleep(min(2 ** attempt, 30))
                continue
            if response.choices[0].finish_reason == "length":
                raise RuntimeError(f"{purpose} 输出达到 token 上限，请增大 max_output_tokens 或关闭思考模式；本次结果未保存")
            content = response.choices[0].message.content
            if content:
                return content
            if attempt + 1 >= max_attempts:
                raise RuntimeError("模型返回空内容")
            time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("API 重试结束但没有响应")


class FakeLLMClient:
    @staticmethod
    def _prompt_json(prompt: str, start: str, end: str) -> Any:
        match = re.search(re.escape(start) + r"(.*?)\n" + re.escape(end), prompt, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None

    def complete(self, messages: list[dict[str, str]], settings: dict[str, Any], *, seed: int | None, purpose: str) -> str:
        if purpose == "generation":
            return "内景。测试场景。人物依据现有状态完成一次可核查行动，并留下下一步选择。"
        if purpose == "extraction" or purpose.startswith("extraction_retry_"):
            prompt = messages[-1]["content"]
            context = self._prompt_json(prompt, "本集经验形成上下文：", "本集受控试用卡：") or {}
            trial_cards = self._prompt_json(prompt, "本集受控试用卡：", "允许的 move_id：") or []
            feedback = [
                {
                    "memory_id": card["memory_id"], "adopted": True, "evidence": "可核查行动",
                    "expected_effect_achieved": True, "state_supported": False,
                    "anchor_supported": False, "causal_supported": True, "harm_flags": [],
                }
                for card in trial_cards
            ]
            state_tags = list(context.get("state_tags", []))
            candidate = None
            if context.get("candidate_generation_allowed") and len(state_tags) >= 2:
                candidate = {
                    "quality_score": 0.82, "confidence": 0.86, "card_type": "STRATEGY",
                    "primary_move_id": "INFO_GAIN", "mechanism_family": "EVIDENCE_LED_PROGRESS",
                    "episode_functions": [context.get("episode_function", "PROGRESS")],
                    "applicable_phases": [context.get("phase", "expansion")],
                    "applicable_state_features": state_tags[:2], "inapplicable_state_features": [],
                    "applicable_situation": "已有证据但推进路径仍不确定时",
                    "operator_steps": ["先核查已有证据的可验证部分", "再据结果收窄下一步选择"],
                    "expected_state_change": "新增可靠信息并保留后续选择",
                    "observable_evidence": ["人物完成核查", "后续选择范围缩小"],
                    "risks": ["核查过程可能过于程序化"], "abort_conditions": ["缺少可核查证据"],
                    "conflicts_with_invariants": [],
                }
            return json.dumps({
                "state_delta": {"character_state": [], "relationship_state": [],
                    "known_information": ["完成一次测试核查"], "unknown_information": [],
                    "confirmed_facts": [], "unresolved_threads": ["下一步仍待选择"],
                    "resources_and_evidence": [], "current_goals": [], "timeline": []},
                "resolved_goals": [], "resolved_unknown": [], "retracted_facts": [],
                "new_obligations": [], "resolved_obligations": [],
                "primary_move_id": "INFO_GAIN", "secondary_move_ids": [],
                "move_evidence": "完成一次可核查行动", "state_constraint_count": 1,
                "state_consistency_issues": [], "card_feedback": feedback,
                "writeback_candidate": candidate,
            }, ensure_ascii=False)
        if purpose.startswith("character_trace"):
            return json.dumps({
                "new_emotional_traces": [], "emotional_updates": [],
                "new_agency_traces": [], "agency_updates": [],
            }, ensure_ascii=False)
        if purpose.startswith("relationship_extraction"):
            return json.dumps({
                "new_relationship_memories": [], "relationship_updates": [],
            }, ensure_ascii=False)
        if purpose == "evaluation":
            return '{"dialogue_quality":78,"scene_structure":80,"visual_adaptability":75,"state_consistency":82,"anchor_satisfaction":80,"causal_continuity":79,"mechanism_novelty":72,"rule_echo_avoidance":85,"details":{"dialogue_quality":[{"type":"问题","item":"句式机械","evidence":"多处台句长相近","points":-22}],"scene_structure":[{"type":"问题","item":"个别场次增量弱","evidence":"中段过渡场","points":-20}],"visual_adaptability":[{"type":"问题","item":"动作指示偏少","evidence":"关键场景仅对话","points":-25}],"state_consistency":[{"type":"问题","item":"轻微漂移","evidence":"人物口吻略变","points":-18}],"anchor_satisfaction":[{"type":"问题","item":"目标部分达成","evidence":"集末留口","points":-20}],"causal_continuity":[{"type":"问题","item":"局部断裂","evidence":"一场衔接弱","points":-21}],"mechanism_novelty":[{"type":"问题","item":"机制常见","evidence":"常规核查推进","points":-28}],"rule_echo_avoidance":[{"type":"问题","item":"轻微边界回声","evidence":"一句谨慎表态","points":-15}]},"justifications":{"dialogue_quality":"对白基本来自当下行动但句式偏同质","scene_structure":"场次可用但中段偏弱","visual_adaptability":"可拍摄但动作指示不足","state_consistency":"与上一状态基本一致","anchor_satisfaction":"完成本集主要任务","causal_continuity":"行动大体触发后续","mechanism_novelty":"机制为常见推进","rule_echo_avoidance":"基本无规则复述"},"rule_echo_spans":[],"violations":[]}'
        if purpose == "branching":
            return '[{"branch_id":"B1","move_id":"INFO_GAIN","outline":"核查旧证据","constraint_check":[]},{"branch_id":"B2","move_id":"BOND_LOSS","outline":"通过对峙选择路径","constraint_check":[]},{"branch_id":"B3","move_id":"INFO_REFRAME","outline":"保留竞争解释","constraint_check":[]},{"branch_id":"B4","move_id":"CAPACITY_LOSS","outline":"让现有方案局部失败","constraint_check":[]},{"branch_id":"B5","move_id":"OPTION_LOSS","outline":"以权限或期限收窄路径","constraint_check":[]}]'
        raise ValueError(f"未知 purpose: {purpose}")
