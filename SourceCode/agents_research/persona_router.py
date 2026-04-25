from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from shared_tools.model_routing import lane_model_config
from shared_tools.project_engine import ProjectEngine


PERSONAS: list[dict[str, str]] = [
    {
        "id": "pm",
        "label": "Product Manager",
        "query_lens": "user problem framing, use cases, adoption friction, product-market fit",
    },
    {
        "id": "market",
        "label": "Market Analyst",
        "query_lens": "market signals, competitor approaches, category demand, differentiation",
    },
    {
        "id": "proj_mgr",
        "label": "Project Manager",
        "query_lens": "adoption risk, implementation complexity, timeline patterns, real-world delivery signals",
    },
    {
        "id": "tech_lead",
        "label": "Tech Lead",
        "query_lens": "architecture patterns, implementation tradeoffs, stack constraints, integration risks",
    },
]


def build_project_context_block(repo_root: Path, project_slug: str) -> str:
    slug = str(project_slug or "").strip() or "general"
    try:
        engine = ProjectEngine(repo_root)
        project = engine.get_by_slug(slug)
    except Exception:
        project = None
    if not isinstance(project, dict):
        return ""

    lines = [f"Active coding project: {project.get('name', slug)}"]
    desc = str(project.get("description") or "").strip()
    if desc:
        lines.append(f"Description: {desc}")
    stack = project.get("stack")
    if isinstance(stack, dict) and stack:
        lines.append(f"Stack: {json.dumps(stack, ensure_ascii=True)}")
    workspace = str(project.get("workspace_path") or "").strip()
    if workspace:
        lines.append(f"Workspace: {workspace}")
    return "\n".join(lines)


def _fallback_queries(question: str) -> list[dict[str, str]]:
    base = " ".join(str(question or "").strip().split())
    out: list[dict[str, str]] = []
    for row in PERSONAS:
        lens = str(row.get("query_lens", "")).strip()
        out.append(
            {
                "id": str(row["id"]),
                "label": str(row["label"]),
                "query": f"{base} {lens}".strip(),
            }
        )
    return out


def _parse_query_payload(raw: str) -> list[dict[str, str]]:
    body = str(raw or "").strip()
    if not body:
        return []
    # Strip fenced wrappers when models ignore constraints.
    body = body.strip("`")
    if body.lower().startswith("json"):
        body = body[4:].strip()

    payload: dict[str, Any] | None = None
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        match = re.search(r"\{[\s\S]*\}", body)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    payload = parsed
            except Exception:
                payload = None

    if not isinstance(payload, dict):
        return []
    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, list):
        return []

    by_id = {str(row["id"]): row for row in PERSONAS}
    out: list[dict[str, str]] = []
    for row in raw_queries:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id", "")).strip()
        query = " ".join(str(row.get("query", "")).split()).strip()
        if pid not in by_id or not query:
            continue
        out.append(
            {
                "id": pid,
                "label": str(by_id[pid]["label"]),
                "query": query,
            }
        )

    # Ensure stable persona order and fill missing rows.
    out_by_id = {str(row["id"]): row for row in out}
    merged: list[dict[str, str]] = []
    for persona in PERSONAS:
        pid = str(persona["id"])
        if pid in out_by_id:
            merged.append(out_by_id[pid])
    return merged


def generate_persona_queries(
    *,
    question: str,
    project_context: str,
    client: Any,
    repo_root: Path,
) -> list[dict[str, str]]:
    cfg = lane_model_config(repo_root, "intent_confirmer") or {}
    model = str(cfg.get("model", "gemma3:4b")).strip() or "gemma3:4b"
    fallback = cfg.get("fallback_models", ["qwen3:4b"]) if isinstance(cfg.get("fallback_models", []), list) else ["qwen3:4b"]

    persona_lines = "\n".join(
        f"- {row['id']}: {row['label']} | lens={row['query_lens']}" for row in PERSONAS
    )
    system_prompt = (
        "Generate exactly one high-quality web search query for each persona. "
        "Return ONLY JSON with shape: {\"queries\":[{\"id\":\"pm\",\"query\":\"...\"}, ...]}."
    )
    user_prompt = (
        f"Research request:\n{str(question or '').strip()}\n\n"
        f"Project context:\n{str(project_context or '').strip() or 'none'}\n\n"
        f"Personas:\n{persona_lines}\n\n"
        "Rules:\n"
        "1) One query per persona id.\n"
        "2) Each query must be materially different from the others.\n"
        "3) Keep each query under 16 words and focused on that persona lens.\n"
        "4) Queries must discover information ABOUT the topic — not generate plans, deliverables, or process outputs.\n"
        "5) The research topic must be the subject of the query, not a modifier for a process term.\n"
    )

    try:
        raw = client.chat(
            model=model,
            fallback_models=fallback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            num_ctx=4096,
            think=False,
            timeout=30,
            retry_attempts=1,
            retry_backoff_sec=0.5,
            task_class="persona_query_planner",
            tier="default",
        )
    except Exception:
        return _fallback_queries(question)

    parsed = _parse_query_payload(str(raw or ""))
    if len(parsed) == len(PERSONAS):
        return parsed

    # Merge whatever parsed with deterministic fallbacks.
    fallback_rows = {row["id"]: row for row in _fallback_queries(question)}
    parsed_rows = {row["id"]: row for row in parsed}
    merged: list[dict[str, str]] = []
    for persona in PERSONAS:
        pid = str(persona["id"])
        merged.append(parsed_rows.get(pid) or fallback_rows[pid])
    return merged


__all__ = ["PERSONAS", "build_project_context_block", "generate_persona_queries"]
