"""Local Qwen/Ollama provider for constrained candidate adjudication."""
from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from typing import Dict
from usage_tracker import tracker


class ProviderUnavailable(RuntimeError):
    pass


def _post(url: str, payload: dict, headers: dict, timeout: int = 90) -> dict:
    req = Request(url, data=json.dumps(payload).encode(), method="POST", headers={"Content-Type": "application/json", **headers})
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ProviderUnavailable(str(exc)) from exc


def _json_text(text: str) -> str:
    return text.strip().removeprefix("```json").removesuffix("```").strip()


def generate_qwen_decisions(prompts: Dict[str, str], system_prompt: str) -> Dict[str, str]:
    model = "qwen2.5:7b"
    base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    output = {}
    for key, prompt in prompts.items():
        response = _post(f"{base}/api/chat", {"model": model, "stream": False, "format": "json", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]}, {})
        usage = response.get("prompt_eval_count", 0) or 0
        output_tokens = response.get("eval_count", 0) or 0
        tracker.record(model, "candidate_adjudication", str(key), int(usage), int(output_tokens))
        text = response.get("message", {}).get("content", "")
        if text:
            output[key] = _json_text(text)
    return output
