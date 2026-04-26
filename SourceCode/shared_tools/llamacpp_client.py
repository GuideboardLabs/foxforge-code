import json
import re
import time
import urllib.error
import urllib.request
from typing import Any


class LlamaCppClient:
    """Client for llama.cpp server's OpenAI-compatible API.

    Drop-in replacement for OllamaClient.chat() when routing models
    through a TurboQuant-enabled llama.cpp server.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8180") -> None:
        self.base_url = base_url.rstrip("/")

    def _post_json(self, path: str, payload: dict[str, Any], timeout: int | float | None = 300) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            if timeout is None:
                resp_ctx = urllib.request.urlopen(req)
            else:
                effective = float(timeout) if float(timeout) > 0 else 300.0
                resp_ctx = urllib.request.urlopen(req, timeout=effective)
            with resp_ctx as resp:
                data = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"llama.cpp HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not connect to llama.cpp server at {self.base_url}") from exc

        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError("llama.cpp server returned non-JSON response") from exc

    _THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

    def chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        *,
        prior_messages: list[dict[str, str]] | None = None,
        user_images: list[str] | None = None,
        temperature: float = 0.3,
        num_ctx: int = 8192,
        think: bool | None = None,
        num_predict: int | None = None,
        timeout: int = 300,
        retry_attempts: int = 1,
        retry_backoff_sec: float = 1.25,
        fallback_models: list[str] | None = None,
    ) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if prior_messages:
            for item in prior_messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "")).strip().lower()
                content = str(item.get("content", "")).strip()
                if role not in {"user", "assistant"} or not content:
                    continue
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})

        try:
            predict = int(num_predict) if num_predict is not None else -1
        except (TypeError, ValueError):
            predict = -1
        if predict <= 0:
            # OpenAI-compatible llama.cpp endpoints often default to low max_tokens if omitted.
            # Force a high generation ceiling to avoid clipped replies.
            predict = max(2048, min(int(num_ctx or 8192), 8192))

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": predict,
            "stream": False,
        }

        attempts = max(1, int(retry_attempts))
        backoff = max(0.0, float(retry_backoff_sec))
        errors: list[str] = []

        for attempt in range(1, attempts + 1):
            try:
                response = self._post_json("/v1/chat/completions", payload, timeout=timeout)
                choices = response.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise RuntimeError("llama.cpp response missing choices")
                message = choices[0].get("message") or {}
                content = message.get("content")
                if not isinstance(content, str):
                    raise RuntimeError("llama.cpp response missing message content")
                clean = content.strip()
                if not clean:
                    raise RuntimeError("llama.cpp returned empty message content")
                if not think:
                    clean = self._THINK_RE.sub("", clean).strip()
                if not clean:
                    raise RuntimeError("llama.cpp returned only think tags with no content")
                return clean
            except Exception as exc:
                errors.append(f"attempt {attempt}/{attempts}: {exc}")
                if attempt < attempts and backoff > 0:
                    sleep_sec = backoff * (1.0 + (attempt - 1) * 0.5)
                    time.sleep(sleep_sec)
                continue

        tail = " | ".join(errors[-6:]) if errors else "unknown failure"
        raise RuntimeError(f"llama.cpp chat failed after retries: {tail}")

    def embed(self, model: str, text: str, *, timeout: int = 60) -> list[float]:
        raise NotImplementedError("LlamaCppClient does not support embeddings; use OllamaClient")

    def list_local_models(self) -> list[str]:
        """Query /v1/models for served model names."""
        try:
            return self.list_local_models_strict()
        except Exception:
            return []

    def list_local_models_strict(self) -> list[str]:
        """Query /v1/models for served model names and raise on failure."""
        url = f"{self.base_url}/v1/models"
        req = urllib.request.Request(url=url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"llama.cpp HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not connect to llama.cpp server at {self.base_url}") from exc
        except Exception as exc:
            raise RuntimeError(f"Could not read llama.cpp models from {self.base_url}") from exc
        models = data.get("data") or []
        return [m["id"] for m in models if isinstance(m, dict) and "id" in m]
