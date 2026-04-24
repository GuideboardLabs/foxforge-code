"""Intent Confirmer — cheap LLM gate between prompt digestion and Make lane routing.

Prevents casual phrases ("make me some tea") from firing expensive multi-agent
Make pools. Uses gemma3:4b for fast (<2s) inference.

Rules:
- If UI mode == "make" AND make_type is explicitly set → skip (user was deliberate).
- If UI mode == "make" but no type → confirms and suggests type.
- If UI mode == "talk" but build-intent regex fired → gates the upgrade.
  Defaults to "chat" on confidence < 0.7.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from shared_tools.activity_bus import telemetry_emit
from shared_tools.config_ini import load_config
from shared_tools.model_routing import load_model_routing

from .make_type_classifier import classify as classify_make_type
from .make_types import MAKE_TYPES, make_types_prompt_fragment, normalize_make_type

LOGGER = logging.getLogger(__name__)

_MODEL = "gemma3:4b"
_TEMPERATURE = 0.1
_NUM_CTX = 4096
_TIMEOUT = 30


def _runtime_model_settings(repo_root: Path) -> tuple[str, float, int, int]:
    try:
        cfg = load_config(repo_root)
        return cfg.get_model("intent_confirmer"), _TEMPERATURE, _NUM_CTX, _TIMEOUT
    except Exception:
        return _MODEL, _TEMPERATURE, _NUM_CTX, _TIMEOUT

_BUILD_INTENT_TERMS = frozenset({
    "build", "create", "make", "generate", "draft", "design", "redesign",
    "implement", "code", "develop", "scaffold", "spec", "prototype",
    "produce", "assemble", "ship", "write the", "launch",
})

_AMBIGUOUS_MAKE_PHRASES = frozenset({
    "make me", "make it", "make sure", "make sense", "make do",
    "make up", "make out", "make time", "make room", "make way",
    "make believe", "make peace", "make friends", "make money",
    "make dinner", "make lunch", "make breakfast", "make food",
    "make tea", "make coffee", "make a move", "make a deal",
    "create an account", "create a profile", "create an event",
})


def _input_hash(text: str) -> str:
    norm = " ".join(str(text or "").strip().lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _classifier_settings(repo_root: Path) -> tuple[bool, float]:
    routing = load_model_routing(repo_root)
    if not isinstance(routing, dict):
        return False, 0.70
    nested = routing.get("make_type_classifier", {}) if isinstance(routing.get("make_type_classifier", {}), dict) else {}
    enabled_raw = routing.get("make_type_classifier.enabled", nested.get("enabled", False))
    threshold_raw = routing.get(
        "make_type_classifier.confidence_threshold",
        nested.get("confidence_threshold", 0.70),
    )
    enabled = bool(enabled_raw)
    try:
        threshold = float(threshold_raw)
    except (TypeError, ValueError):
        threshold = 0.70
    return enabled, max(0.0, min(1.0, threshold))


def _emit_decision(
    repo_root: Path,
    *,
    text: str,
    route: str,
    confidence: float,
    skipped: bool,
    fast_path_hit: bool,
    latency_ms: float,
) -> None:
    telemetry_emit(
        repo_root,
        "gate_decisions.jsonl",
        {
            "gate": "intent_confirmer",
            "input_hash": _input_hash(text),
            "fast_path_hit": bool(fast_path_hit),
            "route": str(route),
            "confidence": round(float(confidence), 4),
            "skipped": bool(skipped),
            "latency_ms": round(float(latency_ms), 2),
        },
        retention_days=14,
    )


def _emit_full_decision(
    repo_root: Path,
    *,
    text: str,
    ui_mode: str,
    make_type: str,
    result: dict[str, Any],
) -> None:
    payload = {
        "gate": "intent_confirmer_full",
        "text": str(text or "")[:2000],
        "input_hash": _input_hash(text),
        "ui_mode": str(ui_mode or ""),
        "preselected_make_type": str(make_type or ""),
        "intent": str(result.get("intent", "chat")),
        "confidence": float(result.get("confidence", 0.0) or 0.0),
        "suggested_type": str(result.get("suggested_type", "")),
        "llm_type": str(result.get("llm_type", "")),
        "classifier_type": str(result.get("classifier_type", "")),
        "classifier_confidence": float(result.get("classifier_confidence", 0.0) or 0.0),
        "used": str(result.get("used", "")),
        "reason": str(result.get("reason", ""))[:240],
        "skipped": bool(result.get("skipped", False)),
    }
    telemetry_emit(
        repo_root,
        "intent_confirmer_full.jsonl",
        payload,
        retention_days=30,
    )


def _emit_make_type_decision(
    repo_root: Path,
    *,
    text: str,
    classifier_type: str,
    classifier_confidence: float,
    llm_type: str,
    llm_confidence: float,
    used: str,
) -> None:
    telemetry_emit(
        repo_root,
        "make_type_decisions.jsonl",
        {
            "text": str(text or "")[:2000],
            "input_hash": _input_hash(text),
            "classifier_type": str(classifier_type or ""),
            "classifier_confidence": round(float(classifier_confidence), 4),
            "llm_type": str(llm_type or ""),
            "llm_confidence": round(float(llm_confidence), 4),
            "used": str(used or ""),
        },
        retention_days=30,
    )


def has_build_intent(text: str) -> bool:
    """Quick regex-based build intent check (same logic as orchestrator/main.py)."""
    low = text.lower()
    for term in _BUILD_INTENT_TERMS:
        if " " in term or "-" in term:
            if term in low:
                return True
        elif re.search(rf"\b{re.escape(term)}\b", low):
            return True
    return False


def is_obviously_ambiguous(text: str) -> bool:
    """Return True if the text matches a known ambiguous make-phrase."""
    low = text.lower().strip()
    for phrase in _AMBIGUOUS_MAKE_PHRASES:
        if low.startswith(phrase) or f" {phrase} " in f" {low} ":
            return True
    return False


def _resolve_suggested_type(
    *,
    text: str,
    repo_root: Path,
    llm_type: str,
    llm_confidence: float,
) -> tuple[str, str, float, str]:
    llm_normalized = normalize_make_type(llm_type)
    classifier_label = ""
    classifier_conf = 0.0
    used = "llm"

    classifier_enabled, threshold = _classifier_settings(repo_root)
    if classifier_enabled:
        try:
            classifier_label, classifier_conf = classify_make_type(text, repo_root=repo_root)
            classifier_label = normalize_make_type(classifier_label)
        except Exception as exc:
            LOGGER.debug("make_type_classifier failed: %s", exc)
            classifier_label, classifier_conf = "", 0.0

    chosen = llm_normalized
    if classifier_label and classifier_conf >= threshold:
        chosen = classifier_label
        used = "classifier"
    elif llm_normalized:
        used = "llm"
    elif classifier_label:
        chosen = classifier_label
        used = "classifier_low_confidence"

    _emit_make_type_decision(
        repo_root,
        text=text,
        classifier_type=classifier_label,
        classifier_confidence=classifier_conf,
        llm_type=llm_normalized,
        llm_confidence=llm_confidence,
        used=used,
    )
    return chosen, classifier_label, classifier_conf, used


def confirm_make_intent(
    text: str,
    repo_root: Path,
    *,
    ui_mode: str = "talk",
    make_type: str = "",
) -> dict[str, Any]:
    """Confirm whether the user prompt is a genuine Make-lane build request.

    Returns dict with:
      - intent: "make" | "chat" | "forage"
      - confidence: float 0..1
      - suggested_type: str (best Make type_id guess, if intent=="make")
      - reason: str
      - skipped: bool (True when confirmation was skipped — fast path)
    """
    started = time.perf_counter()
    ui_mode = str(ui_mode or "talk").strip().lower()
    make_type = normalize_make_type(make_type)

    # Fast path: user explicitly chose mode=make AND selected a type from the modal
    if ui_mode == "make" and make_type:
        result = {
            "intent": "make",
            "confidence": 1.0,
            "suggested_type": make_type,
            "reason": "User explicitly selected Make mode and type from the UI.",
            "skipped": True,
            "used": "ui",
            "llm_type": make_type,
            "classifier_type": make_type,
            "classifier_confidence": 1.0,
        }
        _emit_decision(
            repo_root,
            text=text,
            route=str(result.get("intent", "chat")),
            confidence=float(result.get("confidence", 0.0)),
            skipped=True,
            fast_path_hit=True,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        _emit_full_decision(repo_root, text=text, ui_mode=ui_mode, make_type=make_type, result=result)
        return result

    # If obviously ambiguous — skip LLM call, return chat
    if is_obviously_ambiguous(text):
        result = {
            "intent": "chat",
            "confidence": 0.95,
            "suggested_type": "",
            "reason": "Phrase matches known ambiguous non-build expression.",
            "skipped": True,
            "used": "fast_path",
        }
        _emit_decision(
            repo_root,
            text=text,
            route=str(result.get("intent", "chat")),
            confidence=float(result.get("confidence", 0.0)),
            skipped=True,
            fast_path_hit=True,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        _emit_full_decision(repo_root, text=text, ui_mode=ui_mode, make_type=make_type, result=result)
        return result

    # If mode is "talk" and no build intent regex — skip LLM call
    if ui_mode == "talk" and not has_build_intent(text):
        result = {
            "intent": "chat",
            "confidence": 0.99,
            "suggested_type": "",
            "reason": "No build-intent keywords found in talk mode.",
            "skipped": True,
            "used": "fast_path",
        }
        _emit_decision(
            repo_root,
            text=text,
            route=str(result.get("intent", "chat")),
            confidence=float(result.get("confidence", 0.0)),
            skipped=True,
            fast_path_hit=True,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        _emit_full_decision(repo_root, text=text, ui_mode=ui_mode, make_type=make_type, result=result)
        return result

    # LLM call for ambiguous cases
    try:
        from shared_tools.ollama_client import OllamaClient

        system_prompt = (
            "You are an intent classifier. Determine whether the user's message is a "
            "genuine request to BUILD or CREATE a deliverable artifact (code, document, script, "
            "video script, essay, app, etc.) using an AI Make pipeline — or whether it is "
            "casual conversation, a question, or a non-build request.\n\n"
            "Respond with ONLY valid JSON in this exact format:\n"
            '{"intent": "make"|"chat"|"forage", "confidence": 0.0-1.0, '
            '"suggested_type": "<type_id or empty string>", "reason": "<one sentence>"}\n\n'
            f"Valid type_ids: {make_types_prompt_fragment()}\n\n"
            "Rules:\n"
            "- intent='make' only if the user wants an artifact PRODUCED (a file, a document, "
            "  code, a script). A question about how to do something is 'chat'.\n"
            "- intent='forage' only if the user wants research/investigation without building.\n"
            "- intent='chat' for everything else.\n"
            "- confidence < 0.7 means you're not sure — default to 'chat'.\n"
            "- Return ONLY the JSON object. No markdown, no explanation."
        )
        user_prompt = (
            f"UI mode declared by user: {ui_mode}\n"
            f"Make type pre-selected: {make_type or '(none)'}\n\n"
            f"User message:\n{text[:800]}"
        )

        model_name, temperature, num_ctx, timeout = _runtime_model_settings(repo_root)
        client = OllamaClient()
        raw = client.chat(
            model=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            num_ctx=num_ctx,
            think=False,
            timeout=timeout,
            retry_attempts=2,
            retry_backoff_sec=1.0,
        )
        raw = str(raw or "").strip()

        # Extract JSON from response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            intent = str(data.get("intent", "chat")).strip().lower()
            if intent not in ("make", "chat", "forage"):
                intent = "chat"
            confidence = float(data.get("confidence", 0.0))
            llm_type = normalize_make_type(str(data.get("suggested_type", "")).strip().lower())

            # Enforce the confidence floor: < 0.7 → chat
            if intent == "make" and confidence < 0.7:
                intent = "chat"

            suggested = llm_type
            classifier_type = ""
            classifier_conf = 0.0
            used = "llm"
            if intent == "make":
                suggested, classifier_type, classifier_conf, used = _resolve_suggested_type(
                    text=text,
                    repo_root=repo_root,
                    llm_type=llm_type,
                    llm_confidence=confidence,
                )

            result = {
                "intent": intent,
                "confidence": confidence,
                "suggested_type": suggested,
                "reason": str(data.get("reason", "")).strip()[:200],
                "skipped": False,
                "llm_type": llm_type,
                "classifier_type": classifier_type,
                "classifier_confidence": classifier_conf,
                "used": used,
            }
            _emit_decision(
                repo_root,
                text=text,
                route=str(result.get("intent", "chat")),
                confidence=float(result.get("confidence", 0.0)),
                skipped=False,
                fast_path_hit=False,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
            _emit_full_decision(repo_root, text=text, ui_mode=ui_mode, make_type=make_type, result=result)
            return result
    except Exception as exc:
        LOGGER.warning("IntentConfirmer LLM call failed: %s — defaulting to declared mode", exc)

    # Fallback: trust declared UI mode
    fallback_intent = ui_mode if ui_mode in ("make", "forage") else "chat"
    suggested_type = make_type
    used = "ui_mode_fallback"
    if fallback_intent == "make" and not suggested_type:
        try:
            suggested_type, _classifier_type, _classifier_conf, used = _resolve_suggested_type(
                text=text,
                repo_root=repo_root,
                llm_type="",
                llm_confidence=0.0,
            )
        except Exception:
            suggested_type = ""
    result = {
        "intent": fallback_intent,
        "confidence": 0.5,
        "suggested_type": normalize_make_type(suggested_type),
        "reason": "LLM confirmation unavailable; falling back to declared UI mode.",
        "skipped": False,
        "used": used,
    }
    _emit_decision(
        repo_root,
        text=text,
        route=str(result.get("intent", "chat")),
        confidence=float(result.get("confidence", 0.0)),
        skipped=False,
        fast_path_hit=False,
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )
    _emit_full_decision(repo_root, text=text, ui_mode=ui_mode, make_type=make_type, result=result)
    return result


__all__ = [
    "MAKE_TYPES",
    "confirm_make_intent",
    "has_build_intent",
    "is_obviously_ambiguous",
]
