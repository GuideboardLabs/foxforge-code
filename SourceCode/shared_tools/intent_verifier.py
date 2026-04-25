"""Lightweight command intent verifier.

Runs the intent_confirmer model (gemma3:4b by default) before heavy
dispatch. Skipped entirely for trivial read-only commands so there is
zero added latency for those.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "SourceCode"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from shared_tools.config_ini import load_config
from shared_tools.model_routing import load_model_routing
from shared_tools.ollama_client import OllamaClient

# Commands that need no verification — fast, read-only, unambiguous.
_SKIP_PREFIXES = (
    "/msg",          # always goes to Reynard, never needs intent check
    "/forage",       # search — intent is always clear
    "/view",         # file viewer — self-explanatory flags
    "/projects",
    "/system",
    "/models",
    "/quit",
    "/help",
    "/open",
    "/stack show",
    "/stack change",
    "/git status",
    "/git init",
    "/git commit",
    "/git push",
    "/plan",
    "/build",
    "/execute",
    "/clear",
)

_SYSTEM_PROMPT = """\
You are a command intent verifier for a local coding assistant TUI.
Given a user command and current project context, decide whether the
command intent is clear and actionable.

Reply with ONLY valid JSON in this exact shape:
{"ok": true, "clarification": null}
or
{"ok": false, "clarification": "<one short sentence asking what you need>"}

Do not include any other text, markdown, or explanation.\
"""

_lock = threading.Lock()
_client: OllamaClient | None = None
_lane_cfg: dict[str, Any] = {}
_repo_root: Path | None = None


def _get_client() -> tuple[OllamaClient, dict[str, Any]]:
    global _client, _lane_cfg
    with _lock:
        if _client is None:
            cfg = load_config(_repo_root or ROOT)
            base_url = cfg.get("app", "ollama_base_url", fallback="http://127.0.0.1:11434")
            _client = OllamaClient(base_url=base_url)
            routing = load_model_routing(_repo_root or ROOT)
            _lane_cfg = routing.get("intent_confirmer", {})
        return _client, dict(_lane_cfg)


def init(repo_root: Path) -> None:
    global _repo_root
    _repo_root = repo_root


def prime() -> None:
    """Warm the model in a background thread so the first real command is fast."""
    def _warm() -> None:
        try:
            client, lane = _get_client()
            model = lane.get("model", "gemma3:4b")
            client.chat(
                model=model,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt='Command: /help\nProject: none\nContext: startup warm-up',
                temperature=0.0,
                num_ctx=2048,
                num_predict=32,
                timeout=60,
                keep_alive="15m",
            )
        except Exception:
            pass
    threading.Thread(target=_warm, daemon=True).start()


def verify(raw: str, project_slug: str = "", stack: str = "") -> tuple[bool, str | None]:
    """Return (ok, clarification). ok=True means proceed; False means show clarification."""
    cmd = raw.strip()
    if not cmd:
        return True, None

    # Plain text (no slash) always goes directly to Reynard chat — never gate it.
    if not cmd.startswith("/"):
        return True, None

    # Skip commands that need no verification.
    cmd_lower = cmd.lower()
    for prefix in _SKIP_PREFIXES:
        if cmd_lower.startswith(prefix):
            return True, None

    try:
        client, lane = _get_client()
        model = lane.get("model", "gemma3:4b")
        context_parts = []
        if project_slug:
            context_parts.append(f"Active project: {project_slug}")
        if stack:
            context_parts.append(f"Stack: {stack}")
        if not context_parts:
            context_parts.append("No active project.")
        context = "\n".join(context_parts)

        user_prompt = f"Command: {cmd}\n{context}"
        raw_reply = client.chat(
            model=model,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            num_ctx=2048,
            num_predict=64,
            timeout=30,
            keep_alive="15m",
        )

        # Strip any accidental markdown fences
        cleaned = raw_reply.strip().strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

        parsed = json.loads(cleaned)
        ok = bool(parsed.get("ok", True))
        clarification = parsed.get("clarification") or None
        return ok, clarification

    except Exception:
        # On any failure (timeout, model not loaded, parse error), let the
        # command through rather than blocking the user.
        return True, None


__all__ = ["init", "prime", "verify"]
