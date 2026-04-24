"""Inference router that dispatches model calls to Ollama or llama.cpp servers.

Drop-in replacement for OllamaClient — same .chat() / .embed() / .list_local_models() API.
When ``llama_cpp_servers`` is configured in config.ini ([inference]), matching models are
routed to a llama.cpp server; everything else goes through Ollama as usual.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from shared_tools.llamacpp_client import LlamaCppClient
from shared_tools.model_routing import load_model_routing
from shared_tools.ollama_client import OllamaClient

LOGGER = logging.getLogger(__name__)


class InferenceRouter:
    _SERVER_BACKOFF_SEC = 180.0
    _SERVER_MODELS_TTL_SEC = 300.0
    _shared_lock = threading.Lock()
    _shared_backoff_until: dict[str, float] = {}
    _shared_models_cache: dict[str, dict[str, Any]] = {}

    def __init__(self, repo_root: Path | None = None) -> None:
        self._ollama = OllamaClient()
        self._llama_clients: dict[str, LlamaCppClient] = {}
        self._model_map: dict[str, str] = {}  # model_name -> server_key
        self._fallback_flags: dict[str, bool] = {}  # server_key -> fallback_to_ollama
        self.last_wait_error = ""
        self.last_wait_candidates: list[str] = []
        self.last_wait_polls = 0
        self.last_wait_elapsed_sec = 0

        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[2]

        routing = load_model_routing(repo_root)
        servers = routing.get("llama_cpp_servers")
        if not isinstance(servers, dict):
            return

        for key, cfg in servers.items():
            if not isinstance(cfg, dict):
                continue
            base_url = str(cfg.get("base_url", "")).strip()
            if not base_url:
                continue
            models = cfg.get("models", [])
            if not isinstance(models, list):
                continue
            self._llama_clients[key] = LlamaCppClient(base_url)
            self._fallback_flags[key] = bool(cfg.get("fallback_to_ollama", True))
            for model_name in models:
                name = str(model_name).strip()
                if name:
                    self._model_map[name] = key

        if self._model_map:
            LOGGER.info("InferenceRouter: llama.cpp routing active for %s", list(self._model_map.keys()))

    def _client_for_model(self, model: str) -> OllamaClient | LlamaCppClient:
        server_key = self._model_map.get(model)
        if server_key and server_key in self._llama_clients:
            return self._llama_clients[server_key]
        return self._ollama

    @classmethod
    def _server_in_backoff(cls, server_key: str) -> bool:
        now = time.monotonic()
        with cls._shared_lock:
            until = float(cls._shared_backoff_until.get(server_key, 0.0) or 0.0)
        return until > now

    @classmethod
    def _mark_server_backoff(cls, server_key: str, seconds: float | None = None) -> None:
        duration = float(seconds if seconds is not None else cls._SERVER_BACKOFF_SEC)
        with cls._shared_lock:
            cls._shared_backoff_until[server_key] = time.monotonic() + max(5.0, duration)

    @classmethod
    def _clear_server_backoff(cls, server_key: str) -> None:
        with cls._shared_lock:
            cls._shared_backoff_until.pop(server_key, None)

    def _server_declares_model(self, model: str) -> bool:
        server_key = self._model_map.get(model, "")
        if not server_key or server_key not in self._llama_clients:
            return False
        if self._server_in_backoff(server_key):
            return False

        now = time.monotonic()
        with self._shared_lock:
            cached = dict(self._shared_models_cache.get(server_key) or {})
        expires_at = float(cached.get("expires_at", 0.0) or 0.0)
        models = cached.get("models")
        if expires_at > now and isinstance(models, set):
            return model in models

        client = self._llama_clients[server_key]
        try:
            declared = set(client.list_local_models_strict())
        except Exception as exc:
            LOGGER.warning("InferenceRouter: llama.cpp model discovery failed for %s: %s", server_key, exc)
            self._mark_server_backoff(server_key)
            return False

        with self._shared_lock:
            self._shared_models_cache[server_key] = {
                "expires_at": now + self._SERVER_MODELS_TTL_SEC,
                "models": declared,
            }
        self._clear_server_backoff(server_key)
        return model in declared

    def _should_fallback(self, model: str) -> bool:
        server_key = self._model_map.get(model, "")
        return self._fallback_flags.get(server_key, False)

    @staticmethod
    def _candidate_models(model: str, fallback_models: list[str] | None = None) -> list[str]:
        out: list[str] = []
        for name in [model, *(fallback_models or [])]:
            key = str(name or "").strip()
            if not key or key in out:
                continue
            out.append(key)
        return out

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
        num_predict: int | None = -1,
        timeout: int = 300,
        retry_attempts: int = 1,
        retry_backoff_sec: float = 1.25,
        fallback_models: list[str] | None = None,
        keep_alive: str = "10m",
        task_class: str | None = None,
        artifact_importance: str | None = None,
        tier: str | None = None,
    ) -> str:
        _task_class = str(task_class or "").strip() or "-"
        _artifact_importance = str(artifact_importance or "").strip() or "-"
        _tier = str(tier or "").strip() or "-"
        kwargs: dict[str, Any] = dict(
            prior_messages=prior_messages,
            user_images=user_images,
            temperature=temperature,
            num_ctx=num_ctx,
            think=think,
            num_predict=num_predict,
            timeout=timeout,
            retry_attempts=retry_attempts,
            retry_backoff_sec=retry_backoff_sec,
        )
        errors: list[str] = []
        _call_start = time.perf_counter()
        for _cand_idx, candidate in enumerate(self._candidate_models(model, fallback_models)):
            routed_to_llama = self._server_declares_model(candidate)
            client = self._llama_clients[self._model_map[candidate]] if routed_to_llama else self._ollama
            _route = "llama.cpp" if routed_to_llama else "ollama"
            _attempt_start = time.perf_counter()
            # keep_alive is Ollama-specific — only include it when routing to Ollama
            call_kwargs = dict(kwargs, fallback_models=[])
            if not routed_to_llama:
                call_kwargs["keep_alive"] = keep_alive
            try:
                result = client.chat(
                    candidate,
                    system_prompt,
                    user_prompt,
                    **call_kwargs,
                )
                LOGGER.info(
                    "inference_call model=%s route=%s tier=%s task_class=%s artifact_importance=%s elapsed=%.3fs total=%.3fs attempts=%d fallback=%s status=ok",
                    candidate, _route,
                    _tier, _task_class, _artifact_importance,
                    round(time.perf_counter() - _attempt_start, 3),
                    round(time.perf_counter() - _call_start, 3),
                    _cand_idx + 1, _cand_idx > 0,
                )
                return result
            except RuntimeError as exc:
                _attempt_elapsed = round(time.perf_counter() - _attempt_start, 3)
                errors.append(f"{candidate} via {_route}: {exc}")
                LOGGER.warning(
                    "inference_call model=%s route=%s tier=%s task_class=%s artifact_importance=%s elapsed=%.3fs attempt=%d status=fail error=%s",
                    candidate, _route, _tier, _task_class, _artifact_importance,
                    _attempt_elapsed, _cand_idx + 1, str(exc)[:120],
                )
                if routed_to_llama:
                    server_key = self._model_map.get(candidate, "")
                    self._mark_server_backoff(server_key)
                    if self._should_fallback(candidate):
                        _fb_start = time.perf_counter()
                        try:
                            LOGGER.warning("llama.cpp server failed for %s, falling back to Ollama", candidate)
                            result = self._ollama.chat(
                                candidate,
                                system_prompt,
                                user_prompt,
                                **dict(kwargs, fallback_models=[], keep_alive=keep_alive),
                            )
                            LOGGER.info(
                                "inference_call model=%s route=ollama_fallback tier=%s task_class=%s artifact_importance=%s elapsed=%.3fs total=%.3fs attempts=%d fallback=true status=ok",
                                candidate,
                                _tier, _task_class, _artifact_importance,
                                round(time.perf_counter() - _fb_start, 3),
                                round(time.perf_counter() - _call_start, 3),
                                _cand_idx + 1,
                            )
                            return result
                        except RuntimeError as ollama_exc:
                            errors.append(f"{candidate} via ollama fallback: {ollama_exc}")
                            LOGGER.warning(
                                "inference_call model=%s route=ollama_fallback tier=%s task_class=%s artifact_importance=%s elapsed=%.3fs attempt=%d status=fail",
                                candidate, _tier, _task_class, _artifact_importance,
                                round(time.perf_counter() - _fb_start, 3), _cand_idx + 1,
                            )
                continue

        tail = " | ".join(errors[-8:]) if errors else "No model candidates were available."
        raise RuntimeError(f"InferenceRouter chat failed after routed retries/fallbacks: {tail}")

    def wait_for_available(
        self,
        model: str,
        *,
        fallback_models: list[str] | None = None,
        max_wait_sec: int = 300,
        poll_interval_sec: int = 15,
    ) -> bool:
        """Reachability check for any of {model, *fallback_models}.

        Preflight must not trigger model loads. This checks llama.cpp declarations
        and then delegates to Ollama metadata probes when needed.
        """
        candidates: list[str] = []
        for item in [model, *(fallback_models or [])]:
            entry = str(item or "").strip()
            if entry and entry not in candidates:
                candidates.append(entry)
        if not candidates:
            return False
        self.last_wait_error = ""
        self.last_wait_candidates = list(candidates)
        self.last_wait_polls = 0
        self.last_wait_elapsed_sec = 0
        started = time.monotonic()
        deadline = started + float(max_wait_sec)
        polls = 0
        last_error = ""
        while time.monotonic() < deadline:
            polls += 1
            # Fast path for mapped llama.cpp servers.
            for cand in candidates:
                try:
                    if self._server_declares_model(cand):
                        self.last_wait_error = ""
                        self.last_wait_polls = polls
                        self.last_wait_elapsed_sec = int(max(0.0, time.monotonic() - started))
                        return True
                except Exception as exc:
                    last_error = f"llama.cpp probe {cand}: {str(exc).strip()[:240]}"

            remaining_sec = max(1, int(deadline - time.monotonic()))
            if remaining_sec <= 0:
                break
            try:
                inner_wait = min(remaining_sec, int(max(1.0, float(poll_interval_sec))) + 5)
                inner_poll = min(
                    max(1.0, float(poll_interval_sec)),
                    max(1.0, inner_wait / 2.0),
                )
                if self._ollama.wait_for_available(
                    candidates[0],
                    fallback_models=candidates[1:],
                    max_wait_sec=inner_wait,
                    poll_interval_sec=int(inner_poll),
                ):
                    self.last_wait_error = ""
                    self.last_wait_polls = polls + int(self._ollama.last_wait_polls or 0)
                    self.last_wait_elapsed_sec = int(max(0.0, time.monotonic() - started))
                    return True
                if str(self._ollama.last_wait_error or "").strip():
                    last_error = str(self._ollama.last_wait_error).strip()
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc).strip()[:240]}"

            elapsed = int(max(0.0, time.monotonic() - started))
            LOGGER.debug(
                "wait_for_available: candidates=%s not ready, elapsed=%ds, error=%s",
                candidates,
                elapsed,
                last_error or "unknown",
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(float(poll_interval_sec), max(0.0, remaining)))
        elapsed = int(max(0.0, time.monotonic() - started))
        self.last_wait_error = last_error or "unknown"
        self.last_wait_polls = polls
        self.last_wait_elapsed_sec = elapsed
        LOGGER.warning(
            "wait_for_available timeout: candidates=%s elapsed=%ds polls=%d last_error=%s",
            candidates,
            elapsed,
            polls,
            last_error or "unknown",
        )
        return False

    def release_models(self, models: list[str]) -> None:
        """Release Ollama-hosted models from VRAM after a pool run completes.

        Only releases models NOT in the llama.cpp model map — those are managed
        by the external server process, not Ollama's VRAM scheduler.
        """
        for model in models:
            if not model:
                continue
            if model not in self._model_map:
                self._ollama.release_model(model)

    def embed(self, model: str, text: str, *, timeout: int = 60) -> list[float]:
        return self._ollama.embed(model, text, timeout=timeout)

    def list_local_models(self) -> list[str]:
        models = self._ollama.list_local_models()
        for model_name in self._model_map:
            if model_name not in models:
                models.append(model_name)
        return models
