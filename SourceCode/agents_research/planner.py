from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any


def _now_iso_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    body = _clean_text(raw)
    if not body:
        return None
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", body, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _slugify_token(text: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", _clean_text(text).lower()).strip("_")
    return token[:48]


def _normalize_persona_affinity(raw: Any, personas: list[str]) -> list[str]:
    known = {_clean_text(p).lower(): _clean_text(p) for p in personas if _clean_text(p)}
    out: list[str] = []
    rows = raw if isinstance(raw, list) else []
    for item in rows:
        key = _clean_text(item).lower()
        if not key:
            continue
        if key in known and known[key] not in out:
            out.append(known[key])
    return out


def _fallback_tree(question: str, personas: list[str], max_leaves: int = 5) -> dict[str, Any]:
    root_id = "root"
    branch_templates = [
        "What are the most critical confirmed facts and current state related to this question?",
        "What does the standard or consensus approach miss? When does it fail in practice for real users?",
        "What are the highest-risk failure modes, behavioral edge cases, and silent failure loops?",
        "What non-obvious design or implementation patterns differentiate good solutions from mediocre ones?",
        "What evidence is missing or weak and needs external validation?",
    ]

    persona_defaults = personas[:]
    if not persona_defaults:
        persona_defaults = ["research_agent"]

    nodes: list[dict[str, Any]] = [
        {
            "id": root_id,
            "question": _clean_text(question),
            "depth": 0,
            "persona_affinity": persona_defaults[:2],
        }
    ]
    edges: list[dict[str, str]] = []

    leaves = max(1, min(int(max_leaves), len(branch_templates)))
    for idx in range(leaves):
        q = branch_templates[idx]
        node_id = f"b{idx + 1}"
        affinity = [persona_defaults[idx % len(persona_defaults)]]
        nodes.append(
            {
                "id": node_id,
                "question": q,
                "depth": 1,
                "persona_affinity": affinity,
            }
        )
        edges.append({"from": root_id, "to": node_id})

    return {
        "root_id": root_id,
        "nodes": nodes,
        "edges": edges,
        "planner": "fallback",
    }


def _normalize_tree(raw_tree: dict[str, Any], *, question: str, personas: list[str], max_leaves: int) -> dict[str, Any]:
    root_id = _clean_text(raw_tree.get("root_id")) or "root"
    raw_nodes = raw_tree.get("nodes") if isinstance(raw_tree.get("nodes"), list) else []
    raw_edges = raw_tree.get("edges") if isinstance(raw_tree.get("edges"), list) else []

    nodes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in raw_nodes:
        if not isinstance(item, dict):
            continue
        node_id = _clean_text(item.get("id")) or _slugify_token(item.get("question", ""))
        if not node_id or node_id in seen_ids:
            continue
        node_q = _clean_text(item.get("question"))
        if not node_q:
            continue
        seen_ids.add(node_id)
        try:
            depth = int(item.get("depth", 1))
        except Exception:
            depth = 1
        nodes.append(
            {
                "id": node_id,
                "question": node_q,
                "depth": max(0, depth),
                "persona_affinity": _normalize_persona_affinity(item.get("persona_affinity"), personas),
            }
        )

    if not nodes:
        return _fallback_tree(question, personas, max_leaves=max_leaves)

    node_ids = {n["id"] for n in nodes}
    if root_id not in node_ids:
        root = {"id": root_id, "question": _clean_text(question), "depth": 0, "persona_affinity": personas[:2]}
        nodes.insert(0, root)
        node_ids.add(root_id)

    edges: list[dict[str, str]] = []
    for item in raw_edges:
        if not isinstance(item, dict):
            continue
        frm = _clean_text(item.get("from"))
        to = _clean_text(item.get("to"))
        if not frm or not to or frm not in node_ids or to not in node_ids or frm == to:
            continue
        row = {"from": frm, "to": to}
        if row not in edges:
            edges.append(row)

    children = {e["to"] for e in edges}
    leaves = [n for n in nodes if n["id"] != root_id and n["id"] not in {e["from"] for e in edges}]
    if not leaves:
        leaves = [n for n in nodes if n["id"] != root_id and n["id"] not in children]
    if not leaves:
        leaves = [n for n in nodes if n["id"] != root_id]

    leaves = leaves[: max(1, int(max_leaves))]
    leaf_ids = {n["id"] for n in leaves}
    if leaf_ids:
        edges = [e for e in edges if e["to"] in node_ids and (e["to"] in leaf_ids or e["from"] == root_id)]

    return {
        "root_id": root_id,
        "nodes": nodes,
        "edges": edges,
        "planner": _clean_text(raw_tree.get("planner")) or "llm",
    }


def plan_research_tree(
    question: str,
    *,
    client: Any,
    model_cfg: dict[str, Any],
    personas: list[str],
    max_leaves: int = 5,
    max_depth: int = 2,
) -> dict[str, Any]:
    """Build a research tree (root -> breadth questions -> optional depth leaves)."""
    clean_question = _clean_text(question)
    if not clean_question:
        return _fallback_tree(question, personas, max_leaves=max_leaves)

    model = _clean_text(model_cfg.get("model"))
    if not model or client is None:
        return _fallback_tree(question, personas, max_leaves=max_leaves)

    persona_list = [p for p in personas if _clean_text(p)]
    persona_hint = ", ".join(persona_list[:16]) or "research_agent"
    today = _now_iso_date()
    prompt = (
        f"Date: {today}. Build a compact research tree for this question.\\n"
        "Return JSON only with shape: {root_id, nodes:[{id,question,depth,persona_affinity:[...] }], edges:[{from,to}], planner}.\\n"
        "Rules:\\n"
        f"- Max leaves: {max(1, int(max_leaves))}.\\n"
        f"- Max depth: {max(1, int(max_depth))}.\\n"
        "- Include a single root node at depth 0.\\n"
        "- Leaves must be concrete answerable sub-questions.\\n"
        "- persona_affinity must reference only allowed personas.\\n"
        f"Allowed personas: {persona_hint}.\\n"
        "- Keep wording concise and evidence-seeking."
    )
    user_prompt = f"Research question:\n{clean_question}"

    try:
        raw = client.chat(
            model=model,
            system_prompt=prompt,
            user_prompt=user_prompt,
            temperature=float(model_cfg.get("temperature", 0.15)),
            num_ctx=min(int(model_cfg.get("num_ctx", 12288)), 12288),
            think=bool(model_cfg.get("think", False)),
            timeout=int(model_cfg.get("timeout_sec", 120) or 120),
            retry_attempts=max(1, int(model_cfg.get("retry_attempts", 2))),
            retry_backoff_sec=float(model_cfg.get("retry_backoff_sec", 1.5)),
            fallback_models=model_cfg.get("fallback_models", []) if isinstance(model_cfg.get("fallback_models", []), list) else [],
        )
        parsed = _extract_json_object(raw)
        if not isinstance(parsed, dict):
            return _fallback_tree(question, personas, max_leaves=max_leaves)
        out = _normalize_tree(parsed, question=question, personas=persona_list, max_leaves=max_leaves)
        out["planner"] = f"{out.get('planner', 'llm')}:{model}"
        return out
    except Exception:
        return _fallback_tree(question, personas, max_leaves=max_leaves)


def _leaf_nodes(tree: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = [dict(n) for n in (tree.get("nodes") or []) if isinstance(n, dict)]
    edges = [dict(e) for e in (tree.get("edges") or []) if isinstance(e, dict)]
    parents = {str(e.get("from", "")).strip() for e in edges}
    leaves = [n for n in nodes if str(n.get("id", "")).strip() and str(n.get("id", "")).strip() not in parents]
    root_id = str(tree.get("root_id", "")).strip()
    leaves = [n for n in leaves if str(n.get("id", "")).strip() != root_id]
    return leaves or [n for n in nodes if str(n.get("id", "")).strip() != root_id]


def assign_tree_leaves(tree: dict[str, Any], personas: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Assign tree leaves to personas using affinity-first greedy matching."""
    clean_personas = [_clean_text(p) for p in personas if _clean_text(p)]
    if not clean_personas:
        return {}
    leaves = _leaf_nodes(tree)
    if not leaves:
        return {p: [] for p in clean_personas}

    assignments: dict[str, list[dict[str, Any]]] = {p: [] for p in clean_personas}

    for idx, leaf in enumerate(leaves):
        affinity = [
            _clean_text(a)
            for a in (leaf.get("persona_affinity") if isinstance(leaf.get("persona_affinity"), list) else [])
            if _clean_text(a)
        ]
        chosen = ""
        for candidate in affinity:
            if candidate in assignments:
                chosen = candidate
                break
        if not chosen:
            chosen = clean_personas[idx % len(clean_personas)]
        assignments[chosen].append(
            {
                "id": _clean_text(leaf.get("id")) or f"leaf_{idx + 1}",
                "question": _clean_text(leaf.get("question")),
                "depth": int(leaf.get("depth", 1) or 1),
                "persona_affinity": affinity,
            }
        )

    # Ensure each persona has at least one branch by borrowing the nearest leaf.
    all_leaves = [item for rows in assignments.values() for item in rows]
    if all_leaves:
        for idx, persona in enumerate(clean_personas):
            if assignments[persona]:
                continue
            assignments[persona] = [dict(all_leaves[idx % len(all_leaves)])]

    return assignments


def init_visited_agents_per_leaf(tree: dict[str, Any], personas: list[str]) -> dict[str, list[str]]:
    leaves = _leaf_nodes(tree)
    if not leaves:
        return {"root": [str(p).strip() for p in personas if str(p).strip()][:1]}
    out: dict[str, list[str]] = {}
    for leaf in leaves:
        leaf_id = _clean_text(leaf.get("id")) or f"leaf_{len(out) + 1}"
        out[leaf_id] = []
    return out
