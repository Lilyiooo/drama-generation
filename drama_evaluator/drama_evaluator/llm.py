from __future__ import annotations

import os
import random
import threading
import time
import uuid
from pathlib import Path

import requests
import yaml


class VenusClient:
    """与 drama_operator_new 相同的 Venus 调用方式。

    从 drama_operator_new/trpc_python.yaml 读取 venus 的 url 与 token，
    使用 requests.post 直连；但针对 GPT-5.6 只传其支持的
    ``max_completion_tokens``，不再传旧模型专用的 temperature/top_p/top_k/
    do_sample/max_tokens 等参数。
    """

    def __init__(self) -> None:
        rel_paths = [
            "drama_operator-master/drama_operator_new/trpc_python.yaml",
            "drama_operator/trpc_python.yaml",
        ]
        trpc_path = None
        for parent in Path(__file__).resolve().parents:
            for rel in rel_paths:
                candidate = parent / rel
                if candidate.is_file():
                    trpc_path = candidate
                    break
            if trpc_path is not None:
                break
        if trpc_path is None:
            raise RuntimeError(f"未找到 trpc 配置，已尝试相对路径：{rel_paths}")
        with open(trpc_path, "r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp)
        self.url = config["API_URLS"]["venus"]
        token = os.getenv("VENUS_API_KEY") or config["API_KEYS"]["venus"]
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int,
        timeout: float,
    ) -> str:
        payload = {
            "task_id": "eval_" + str(uuid.uuid4()),
            "model": model,
            "messages": messages,
            "stream": False,
            "max_completion_tokens": max_output_tokens,
        }
        rsp = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            stream=False,
            timeout=(5, timeout),
        )
        if rsp.status_code != 200:
            raise RuntimeError(f"Error: {rsp.status_code}, {rsp.text}")
        data = rsp.json()
        content = data["choices"][0]["message"]["content"]
        if not content:
            raise RuntimeError("模型返回了空内容")
        return content


class LocalVLLMClient:
    def __init__(self, base_url: str) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ.get("DRAMA_EVAL_API_KEY", "EMPTY"),
        }

    def chat(self, model: str, messages: list[dict[str, str]], *, max_output_tokens: int, timeout: float) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_output_tokens,
            "temperature": float(os.environ.get("DRAMA_EVAL_TEMPERATURE", "0")),
            "chat_template_kwargs": {
                "enable_thinking": os.environ.get("DRAMA_EVAL_ENABLE_THINKING", "false").lower() in {"1", "true", "yes"},
            },
        }
        response = requests.post(self.url, headers=self.headers, json=payload, timeout=(10, timeout))
        if response.status_code != 200:
            raise RuntimeError(f"Error: {response.status_code}, {response.text}")
        choice = response.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError("评测输出被截断，请增大 --max-output-tokens 或关闭思考模式")
        content = choice["message"].get("content")
        if not content or not content.strip():
            raise RuntimeError("模型返回了空内容")
        return content


class OpenAIChatClient:
    def __init__(
        self,
        *,
        model: str,
        max_output_tokens: int = 16000,
        timeout: float = 600.0,
        retries: int = 3,
    ) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.retries = retries
        self._counter = 0
        self._counter_lock = threading.Lock()
        base_url = os.environ.get("DRAMA_EVAL_BASE_URL", "")
        self._client = LocalVLLMClient(base_url) if base_url else VenusClient()

    def _next_call_number(self) -> int:
        with self._counter_lock:
            self._counter += 1
            return self._counter

    def text(self, system: str, user: str, *, label: str) -> str:
        last_error: Exception | None = None
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for attempt in range(1, self.retries + 1):
            call_number = self._next_call_number()
            print(
                f"[LLM #{call_number}] {label}，第 {attempt}/{self.retries} 次，"
                f"model={self.model}",
                flush=True,
            )
            try:
                content = self._client.chat(
                    self.model,
                    messages,
                    max_output_tokens=self.max_output_tokens,
                    timeout=self.timeout,
                )
                print(f"[LLM #{call_number}] {label} 完成", flush=True)
                return content.strip()
            except Exception as error:
                last_error = error
                print(
                    f"[LLM #{call_number}] {label} 失败：{type(error).__name__}: {error}",
                    flush=True,
                )
                # 参数类错误（400）重试无意义，直接抛出。
                if "unsupported_parameter" in str(error) or "invalid_request_error" in str(error):
                    break
                if attempt < self.retries:
                    time.sleep(min(30.0, 2 ** (attempt - 1)) + random.random())
        raise RuntimeError(f"{label} 请求失败：{last_error}") from last_error
