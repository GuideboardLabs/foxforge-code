from __future__ import annotations

import json
import re
from typing import Any, Literal


def score_artifact(
    *,
    artifact_kind: Literal["research_summary", "domain_summary", "project_summary", "implementation_brief"],
    artifact_md: str,
    primitives: dict | None,
    project_context: str,
    workspace_knowledge: str,
    client: Any,
    model_cfg: dict,
) -> dict:
    body = str(artifact_md or "").strip()
    if not body:
        return {
            "genericity_score": 5,
            "usefulness_score": 0,
            "generic_phrases": ["empty artifact"],
            "missing_specifics": ["project-specific recommendations"],
            "reasoning": "Artifact is empty.",
        }

    # Cheap lexical fallback so scoring still works without LLM.
    generic_hits: list[str] = []
    for phrase in ("smart goals", "follow agile", "iterate quickly", "best practice", "industry standard"):
        if phrase in body.lower():
            generic_hits.append(phrase)
    concrete_tokens = 0
    for token in (str(project_context or "") + "\n" + str(workspace_knowledge or "")).split():
        clean = token.strip().lower()
        if len(clean) >= 4 and clean in body.lower():
            concrete_tokens += 1
    base_genericity = 4 if generic_hits else 2
    base_usefulness = 1 if generic_hits else 3
    if concrete_tokens >= 8:
        base_genericity = max(0, base_genericity - 1)
        base_usefulness = min(5, base_usefulness + 1)

    model = str(model_cfg.get("model", "qwen3:8b")).strip() or "qwen3:8b"
    fallback = model_cfg.get("fallback_models", []) if isinstance(model_cfg.get("fallback_models", []), list) else []
    system_prompt = (
        "Score artifact quality for genericity/usefulness. Return ONLY JSON with keys: "
        "genericity_score, usefulness_score, generic_phrases, missing_specifics, reasoning."
    )
    user_prompt = (
        f"Artifact kind: {artifact_kind}\n\n"
        f"Artifact:\n{body[:7000]}\n\n"
        f"Primitives:\n{json.dumps(primitives or {}, ensure_ascii=True)[:2500]}\n\n"
        f"Project context:\n{project_context[:1200]}\n\n"
        f"Workspace knowledge:\n{workspace_knowledge[:1200]}\n"
    )

    parsed: dict[str, Any] = {}
    try:
        raw = str(
            client.chat(
                model=model,
                fallback_models=fallback,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                num_ctx=int(model_cfg.get("num_ctx", 12288) or 12288),
                think=False,
                timeout=int(model_cfg.get("timeout_sec", 420) or 420),
                retry_attempts=1,
                retry_backoff_sec=1.0,
                task_class="genericity_gate",
                tier="default",
            )
            or ""
        ).strip()
        payload = raw
        if not payload.startswith("{"):
            m = re.search(r"\{[\s\S]*\}", payload)
            if m:
                payload = m.group(0)
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            parsed = {}
    except Exception:
        parsed = {}

    def _score(key: str, default: int) -> int:
        try:
            value = int(parsed.get(key, default))
        except Exception:
            value = default
        return max(0, min(5, value))

    out = {
        "genericity_score": _score("genericity_score", base_genericity),
        "usefulness_score": _score("usefulness_score", base_usefulness),
        "generic_phrases": [str(x).strip() for x in parsed.get("generic_phrases", generic_hits) if str(x).strip()] if isinstance(parsed.get("generic_phrases", generic_hits), list) else list(generic_hits),
        "missing_specifics": [str(x).strip() for x in parsed.get("missing_specifics", []) if str(x).strip()] if isinstance(parsed.get("missing_specifics", []), list) else [],
        "reasoning": str(parsed.get("reasoning", "")).strip() or "heuristic score",
    }
    return out


__all__ = ["score_artifact"]
