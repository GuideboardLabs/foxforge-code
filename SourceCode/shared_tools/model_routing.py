from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from shared_tools.config_ini import load_config


def _lane(role: str, *, fallback_models: list[str] | None = None, temperature: float = 0.2, num_ctx: int = 8192, think: bool | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "provider": "ollama",
        "model": role,
        "temperature": temperature,
        "num_ctx": num_ctx,
        "timeout_sec": 240,
        "retry_attempts": 2,
        "retry_backoff_sec": 1.2,
        "fallback_models": list(fallback_models or []),
    }
    if think is not None:
        row["think"] = bool(think)
    return row


def load_model_routing(repo_root: Path) -> dict[str, Any]:
    cfg = load_config(repo_root)

    try:
        llama_cpp_raw = cfg.get("inference", "llama_cpp_servers", fallback="")
        llama_cpp = json.loads(llama_cpp_raw) if llama_cpp_raw.strip() else {}
    except Exception:
        llama_cpp = {}
    if not isinstance(llama_cpp, dict):
        llama_cpp = {}

    models = cfg.section("models")
    get = lambda key, default="": str(models.get(key, default)).strip()
    ack_http = str(cfg.get("inference", "mcp.acknowledge_unsafe_http", fallback="false")).strip().lower() in {"1", "true", "yes", "on"}

    routing: dict[str, Any] = {
        "llama_cpp_servers": llama_cpp,
        "mcp.acknowledge_unsafe_http": ack_http,
        "orchestrator_reasoning": _lane(get("orchestrator_reasoning", "deepseek-r1:8b"), fallback_models=[get("research_execution_planner", "qwen3:8b")], think=True, num_ctx=12288),
        "reynard_layer": _lane(get("reynard_layer", "dolphin3:8b"), fallback_models=[get("orchestrator_reasoning", "deepseek-r1:8b")], temperature=0.55),
        "conversation_layer": _lane(get("reynard_layer", "dolphin3:8b"), fallback_models=[get("orchestrator_reasoning", "deepseek-r1:8b")], temperature=0.55),
        "chat_routing_gate": _lane(get("chat_routing_gate", "qwen3:4b"), fallback_models=[get("intent_confirmer", "gemma3:4b")], temperature=0.1),
        "embeddings": _lane(get("embeddings", "qwen3-embedding:4b"), fallback_models=[], temperature=0.0, num_ctx=4096),
        "research_pool": {
            "model": get("research_market_analyst", "qwen3:8b"),
            "parallel_agents": 2,
            "temperature": 0.25,
            "num_ctx": 12288,
            "timeout_sec": 420,
            "retry_attempts": 3,
            "retry_backoff_sec": 2.0,
            "fallback_models": [
                get("research_technical", "deepseek-r1:8b"),
                get("research_execution_planner", "qwen3:8b"),
            ],
        },
        "synthesis": {
            "model": get("synthesis_default", "qwen3:8b"),
            "temperature": 0.2,
            "num_ctx": 12288,
            "timeout_sec": 480,
            "fallback_models": [get("synthesis_premium", "qwen3:14b")],
            "tier_default": {
                "model": get("synthesis_default", "qwen3:8b"),
                "temperature": 0.2,
                "num_ctx": 12288,
                "timeout_sec": 480,
                "retry_attempts": 3,
                "retry_backoff_sec": 2.0,
                "fallback_models": [get("synthesis_premium", "qwen3:14b")],
            },
            "tier_premium": {
                "model": get("synthesis_premium", "qwen3:14b"),
                "temperature": 0.2,
                "num_ctx": 16384,
                "timeout_sec": 900,
                "retry_attempts": 2,
                "retry_backoff_sec": 2.0,
                "fallback_models": [get("synthesis_default", "qwen3:8b")],
            },
        },
        "make_tool": _lane(get("make_tool_implementer", "qwen2.5-coder:7b"), fallback_models=[get("make_tool_architect", "qwen2.5-coder:7b")], num_ctx=12288),
        "make_app": _lane(get("make_webapp_implementer", "qwen2.5-coder:14b"), fallback_models=[get("make_webapp_fallback", "qwen2.5-coder:7b")], num_ctx=12288),
        "make_desktop_app": _lane(get("make_desktop_implementer", "qwen2.5-coder:14b"), fallback_models=[get("make_desktop_architect", "qwen2.5-coder:14b")], num_ctx=12288),
        "intent_confirmer": _lane(get("intent_confirmer", "gemma3:4b"), fallback_models=[get("chat_routing_gate", "qwen3:4b")], temperature=0.0, num_ctx=4096),
        "web_query_refiner": _lane(get("web_query_refiner", "qwen3:8b"), fallback_models=[get("web_query_refiner_fallback", "deepseek-r1:8b")], temperature=0.2),
    }

    return routing


def lane_model_config(repo_root: Path, lane_key: str) -> dict[str, Any]:
    routing = load_model_routing(repo_root)
    value = routing.get(lane_key, {})
    if isinstance(value, dict):
        return value
    return {}


def llama_cpp_server_config(repo_root: Path) -> dict[str, Any]:
    routing = load_model_routing(repo_root)
    servers = routing.get("llama_cpp_servers", {})
    if isinstance(servers, dict):
        return servers
    return {}


def resolved_tier_config(lane_cfg: dict, tier: Literal["default", "premium"]) -> dict[str, Any] | None:
    cfg = lane_cfg if isinstance(lane_cfg, dict) else {}
    if tier == "premium":
        premium = cfg.get("tier_premium")
        if isinstance(premium, dict):
            return dict(premium)
        return None

    default_cfg = cfg.get("tier_default")
    if isinstance(default_cfg, dict):
        return dict(default_cfg)

    legacy_keys = (
        "model",
        "num_ctx",
        "temperature",
        "fallback_models",
        "timeout_sec",
        "retry_attempts",
        "retry_backoff_sec",
        "think",
        "num_predict",
        "num_gpu",
        "keep_alive",
    )
    synthesized: dict[str, Any] = {}
    for key in legacy_keys:
        if key in cfg:
            synthesized[key] = cfg.get(key)
    return synthesized
