from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared_tools.file_store import ProjectStore


PRIMITIVES_KEYS = (
    "milestones",
    "success_criteria",
    "failure_modes",
    "measurement_dimensions",
    "capabilities",
)


def _default_primitives() -> dict[str, Any]:
    return {
        "milestones": [],
        "success_criteria": [],
        "failure_modes": [],
        "measurement_dimensions": [],
        "capabilities": [],
        "notes": "",
    }


def extract_primitives(
    *,
    question: str,
    synthesis_md: str,
    claims: list[dict[str, Any]] | None,
    client: Any,
    model_cfg: dict[str, Any],
    research_intent: str,
) -> dict[str, Any]:
    intent = str(research_intent or "").strip().lower()
    if intent not in {"domain_foraging", "technical_planning", "final_synthesis"}:
        return {"enabled": False, **_default_primitives()}

    model = str(model_cfg.get("model", "qwen3:8b")).strip() or "qwen3:8b"
    fallback = model_cfg.get("fallback_models", []) if isinstance(model_cfg.get("fallback_models", []), list) else []
    system_prompt = (
        "Extract domain primitives from research synthesis. Return ONLY JSON object with keys: "
        "milestones, success_criteria, failure_modes, measurement_dimensions, capabilities, notes."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Synthesis:\n{synthesis_md[:7000]}\n\n"
        f"Claims:\n{json.dumps(claims or [], ensure_ascii=True)[:3000]}\n"
    )
    try:
        raw = str(
            client.chat(
                model=model,
                fallback_models=fallback,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                num_ctx=int(model_cfg.get("num_ctx", 12288) or 12288),
                think=False,
                timeout=int(model_cfg.get("timeout_sec", 420) or 420),
                retry_attempts=1,
                retry_backoff_sec=1.0,
                task_class="domain_primitives",
                tier="default",
            )
            or ""
        ).strip()
        payload = raw
        if not payload.startswith("{"):
            import re
            match = re.search(r"\{[\s\S]*\}", payload)
            if match:
                payload = match.group(0)
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            parsed = {}
    except Exception:
        parsed = {}

    out = _default_primitives()
    out["enabled"] = True
    for key in PRIMITIVES_KEYS:
        value = parsed.get(key)
        out[key] = value if isinstance(value, list) else []
    out["notes"] = str(parsed.get("notes", "")).strip()
    return out


def persist_primitives(
    *,
    repo_root: Path,
    project_slug: str,
    summary_path: str,
    primitives: dict[str, Any],
) -> str:
    store = ProjectStore(repo_root)
    summary_name = Path(summary_path).name if summary_path else store.timestamped_name("research_summary")
    file_name = f"{summary_name}.primitives.json"
    path = store.write_project_file(
        project_slug,
        "research_summaries",
        file_name,
        json.dumps(primitives, indent=2, ensure_ascii=True),
    )
    return str(path)


__all__ = ["extract_primitives", "persist_primitives", "PRIMITIVES_KEYS"]
