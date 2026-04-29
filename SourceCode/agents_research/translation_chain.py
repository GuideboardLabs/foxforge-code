from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from shared_tools.model_routing import lane_model_config


CHAIN_STAGES: list[dict[str, str]] = [
    {
        "id": "pm",
        "label": "Product Manager",
        "focus": "Translate into concrete user and product implications for this exact project.",
    },
    {
        "id": "market",
        "label": "Market Analyst",
        "focus": "Translate into market, category, and competitor implications without repeating PM output.",
    },
    {
        "id": "proj_mgr",
        "label": "Project Manager",
        "focus": (
            "Translate into execution structure and answer explicitly whether this project appears "
            "new, in-development, in-theory, or already-evolved."
        ),
    },
    {
        "id": "tech_lead",
        "label": "Tech Lead",
        "focus": "Translate into technical implementation guidance tied to the project stack/workspace context.",
    },
]

RECOMMENDATION_STRENGTH_LABELS = {
    "implement_now",
    "prototype",
    "design_option",
    "future_experiment",
    "weak_do_not_prioritize",
    "reject",
}


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, _plain(a), _plain(b)).ratio())


def _extract_bullets(text: str, *, limit: int = 8) -> list[str]:
    rows: list[str] = []
    for line in str(text or "").splitlines():
        m = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.+)$", line.strip())
        if not m:
            continue
        value = str(m.group(1) or "").strip()
        if not value:
            continue
        rows.append(value)
        if len(rows) >= max(1, int(limit)):
            break
    return rows


def _render_stage(label: str, body: str) -> str:
    cleaned = str(body or "").strip()
    if not cleaned:
        cleaned = "- No additional implications generated for this stage."
    if not cleaned.lstrip().startswith(("-", "*", "1.")):
        cleaned = "- " + cleaned
    return f"### {label}\n{cleaned}"


def _model_for_stage(
    *,
    stage_id: str,
    synthesis_cfg: dict[str, Any],
    use_premium_tech_lead: bool,
) -> str:
    default_model = str(synthesis_cfg.get("model", "qwen3:8b")).strip() or "qwen3:8b"
    if stage_id != "tech_lead" or not use_premium_tech_lead:
        return default_model
    premium = synthesis_cfg.get("tier_premium", {}) if isinstance(synthesis_cfg.get("tier_premium", {}), dict) else {}
    premium_model = str(premium.get("model", "")).strip()
    return premium_model or default_model


def run_translation_chain(
    *,
    question: str,
    synthesis_summary: str,
    project_context: str,
    client: Any,
    repo_root: Path,
    synthesis_cfg: dict[str, Any] | None = None,
    use_premium_tech_lead: bool = False,
    primitives: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run PM -> Market -> Project Manager -> Tech Lead translation stages."""
    cfg = dict(synthesis_cfg or lane_model_config(repo_root, "synthesis") or {})
    fallback = cfg.get("fallback_models", []) if isinstance(cfg.get("fallback_models", []), list) else []
    stage_outputs: list[dict[str, str]] = []
    prior_blocks: list[str] = []

    for stage in CHAIN_STAGES:
        label = str(stage["label"])
        stage_id = str(stage["id"])
        model = _model_for_stage(
            stage_id=stage_id,
            synthesis_cfg=cfg,
            use_premium_tech_lead=bool(use_premium_tech_lead),
        )
        prior_text = "\n\n".join(prior_blocks)
        system_prompt = (
            "You are a translation stage in a research pipeline. "
            "Return concise markdown bullets only (3-6 bullets). "
            "Do not parrot prior stages. Add new implications from your role lens."
        )
        user_prompt = (
            f"Research question:\n{str(question or '').strip()}\n\n"
            f"Synthesis summary:\n{str(synthesis_summary or '').strip()[:6000]}\n\n"
            f"Project context:\n{str(project_context or '').strip() or 'none'}\n\n"
            f"Domain Primitives:\n{json.dumps(primitives or {}, ensure_ascii=True)[:3000] or 'none'}\n\n"
            f"Prior stage outputs:\n{prior_text or 'none'}\n\n"
            f"Your role: {label}\n"
            f"Role focus: {stage['focus']}\n\n"
            "Rules:\n"
            "1) No preamble.\n"
            "2) No section heading.\n"
            "3) Mention project-specific details when possible (name, stack, workspace, maturity).\n"
            "4) Project Manager stage must include one bullet that explicitly states: new / in-development / in-theory / already-evolved.\n"
        )

        def _call(extra_instruction: str = "") -> str:
            prompt = user_prompt
            if extra_instruction:
                prompt = f"{prompt}\n\n{extra_instruction}"
            return str(
                client.chat(
                    model=model,
                    fallback_models=fallback,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    temperature=0.2,
                    num_ctx=int(cfg.get("num_ctx", 12288) or 12288),
                    think=False,
                    timeout=int(cfg.get("timeout_sec", 420) or 420),
                    retry_attempts=2,
                    retry_backoff_sec=1.5,
                    task_class="translation_chain",
                    tier="premium" if (stage_id == "tech_lead" and use_premium_tech_lead) else "default",
                )
                or ""
            ).strip()

        output = _call()
        similarity = max((_similarity(output, prior) for prior in prior_blocks), default=0.0)
        reran = False
        if similarity > 0.85:
            output = _call("Your prior output was too similar. Rewrite with distinct implications and different claims.")
            reran = True

        cleaned = str(output or "").strip()
        if not cleaned:
            cleaned = "- No additional implications generated for this stage."

        stage_outputs.append(
            {
                "id": stage_id,
                "label": label,
                "content": cleaned,
                "reran_for_parrot": bool(reran),
            }
        )
        prior_blocks.append(f"{label}:\n{cleaned}")

    chain_sections = ["## Project Implications"]
    for row in stage_outputs:
        chain_sections.append(_render_stage(str(row["label"]), str(row["content"])))

    recommendations = _classify_recommendation_strength(
        question=question,
        project_context=project_context,
        synthesis_summary=synthesis_summary,
        stage_outputs=stage_outputs,
        primitives=primitives,
        client=client,
        cfg=cfg,
    )
    if recommendations:
        chain_sections.append("## Recommended Actions (with strength labels)")
        for row in recommendations:
            finding = str(row.get("finding", "")).strip()
            implication = str(row.get("implication", "")).strip()
            strength = str(row.get("strength", "")).strip().lower()
            rationale = str(row.get("rationale", "")).strip()
            if strength not in RECOMMENDATION_STRENGTH_LABELS:
                strength = "design_option"
            text = finding or implication or "Action candidate"
            if implication and implication != finding:
                text = f"{text} -> {implication}"
            if rationale:
                text = f"{text} ({rationale})"
            chain_sections.append(f"- {text} - strength: {strength}")

    chain_block = "\n\n".join(chain_sections)

    base = str(synthesis_summary or "").strip()
    # Replace existing section if present, else append.
    if re.search(r"^##\s+Project\s+Implications\b", base, flags=re.IGNORECASE | re.MULTILINE):
        base = re.sub(
            r"(?is)^##\s+Project\s+Implications\b[\s\S]*$",
            chain_block,
            base,
        ).strip()
    else:
        base = f"{base}\n\n{chain_block}".strip()

    return {
        "summary": base,
        "stages": stage_outputs,
        "recommendations": recommendations,
    }


def _classify_recommendation_strength(
    *,
    question: str,
    project_context: str,
    synthesis_summary: str,
    stage_outputs: list[dict[str, str]],
    primitives: dict[str, Any] | None,
    client: Any,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    model = str(cfg.get("model", "qwen3:8b")).strip() or "qwen3:8b"
    fallback = cfg.get("fallback_models", []) if isinstance(cfg.get("fallback_models", []), list) else []
    stage_blob = "\n\n".join(
        f"{str(row.get('label', 'stage'))}:\n{str(row.get('content', '')).strip()}"
        for row in stage_outputs
    ).strip()
    system_prompt = (
        "You classify recommendation strength for project implications. "
        "Return ONLY JSON list where each row has: finding, implication, strength, rationale, evidence_ref, primitive_ref. "
        "Allowed strength labels: implement_now, prototype, design_option, future_experiment, weak_do_not_prioritize, reject."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Project context:\n{project_context or 'none'}\n\n"
        f"Synthesis summary:\n{synthesis_summary[:5000]}\n\n"
        f"Domain primitives:\n{json.dumps(primitives or {}, ensure_ascii=True)[:2800] or 'none'}\n\n"
        f"Translation stages:\n{stage_blob[:6000]}\n"
    )
    try:
        raw = str(
            client.chat(
                model=model,
                fallback_models=fallback,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                num_ctx=int(cfg.get("num_ctx", 12288) or 12288),
                think=False,
                timeout=int(cfg.get("timeout_sec", 420) or 420),
                retry_attempts=1,
                retry_backoff_sec=1.0,
                task_class="recommendation_strength",
                tier="default",
            )
            or ""
        ).strip()
        candidate = raw
        if not candidate.startswith("["):
            m = re.search(r"\[[\s\S]*\]", candidate)
            if m:
                candidate = m.group(0)
        parsed = json.loads(candidate)
        if not isinstance(parsed, list):
            return []
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for row in parsed:
        if not isinstance(row, dict):
            continue
        strength = str(row.get("strength", "")).strip().lower()
        if strength not in RECOMMENDATION_STRENGTH_LABELS:
            continue
        out.append(
            {
                "finding": str(row.get("finding", "")).strip(),
                "implication": str(row.get("implication", "")).strip(),
                "strength": strength,
                "rationale": str(row.get("rationale", "")).strip(),
                "evidence_ref": list(row.get("evidence_ref", [])) if isinstance(row.get("evidence_ref", []), list) else [],
                "primitive_ref": list(row.get("primitive_ref", [])) if isinstance(row.get("primitive_ref", []), list) else [],
            }
        )
    return out[:18]


def ensure_project_specificity(summary_md: str, project_context: str) -> str:
    summary = str(summary_md or "").strip()
    context = str(project_context or "").strip()
    if not summary or not context:
        return summary

    low = summary.lower()
    tokens: list[str] = []

    proj_match = re.search(r"Active coding project:\s*(.+)", context)
    if proj_match:
        tokens.append(str(proj_match.group(1)).strip().lower())

    stack_match = re.search(r"Stack:\s*(\{.+\})", context)
    if stack_match:
        raw = str(stack_match.group(1)).strip()
        try:
            stack = json.loads(raw)
        except Exception:
            stack = {}
        if isinstance(stack, dict):
            for value in stack.values():
                text = str(value or "").strip().lower()
                if text and text not in {"none", "null"}:
                    tokens.append(text)

    workspace_match = re.search(r"Workspace:\s*(.+)", context)
    if workspace_match:
        workspace = str(workspace_match.group(1)).strip().lower()
        if workspace:
            tokens.append(workspace.split("/")[-1])

    tokens = [t for t in tokens if t]
    if tokens and any(t in low for t in tokens):
        return summary

    anchor_lines = ["## Project Context Anchors"]
    for line in context.splitlines():
        line = str(line).strip()
        if line:
            anchor_lines.append(f"- {line}")
    return f"{summary}\n\n" + "\n".join(anchor_lines)


def enforce_final_summary_contract(
    *,
    summary_md: str,
    findings: list[dict[str, Any]] | None,
    source_quality_warning: str = "",
    analogy_sources: list[str] | None = None,
) -> str:
    summary = str(summary_md or "").strip()
    if not summary:
        return summary

    sections: list[str] = []

    # Evidence/Inference/Speculation/Analogy buckets.
    evidence_bullets = _extract_bullets("\n".join([ln for ln in summary.splitlines() if "[E]" in ln]))
    inference_bullets = _extract_bullets("\n".join([ln for ln in summary.splitlines() if "[I]" in ln]))
    speculation_bullets = _extract_bullets("\n".join([ln for ln in summary.splitlines() if "[S]" in ln]))
    analogy_bullets = _extract_bullets("\n".join([ln for ln in summary.splitlines() if "[A]" in ln]))

    if not re.search(r"^##\s+Evidence\b", summary, flags=re.IGNORECASE | re.MULTILINE):
        # Only emit sub-sections that have real content; skip Inference [I] (redundant with inline tags).
        bucket_parts: list[str] = []
        if evidence_bullets:
            bucket_parts.append("### Evidence [E]")
            bucket_parts.extend(f"- {row}" for row in evidence_bullets)
        if speculation_bullets:
            bucket_parts.append("### Speculation [S]")
            bucket_parts.extend(f"- {row}" for row in speculation_bullets)
        if analogy_bullets:
            bucket_parts.append("### Analogies [A]")
            bucket_parts.extend(f"- {row}" for row in analogy_bullets)
        if bucket_parts:
            sections.append("## Evidence / Inference / Speculation Buckets")
            sections.extend(bucket_parts)

    if not re.search(r"^##\s+Recommended\s+Actions\b", summary, flags=re.IGNORECASE | re.MULTILINE):
        next_steps = _extract_bullets(
            "\n".join([ln for ln in summary.splitlines() if re.search(r"next steps", ln, re.IGNORECASE)])
        )
        sections.append("## Recommended Actions")
        if next_steps:
            sections.extend(f"- {row}" for row in next_steps)
        else:
            sections.append("- No explicit action list was generated in this pass.")

    if not re.search(r"^##\s+Insights?\b", summary, flags=re.IGNORECASE | re.MULTILINE):
        sections.extend(
            [
                "## Insights & Design Implications",
                "- Coverage gap: no non-obvious insights were surfaced in this run.",
                "- Consider re-running with a more specific forage question targeting edge cases and failure modes.",
            ]
        )

    if not re.search(r"^##\s+Failure\s+Modes?\b", summary, flags=re.IGNORECASE | re.MULTILINE):
        sections.extend(
            [
                "## Failure Modes & Risks",
                "- No specific failure modes were identified in this run.",
                "- Treat consensus-only findings as untested until at least one failure scenario is documented.",
            ]
        )

    if not re.search(r"^##\s+Differentiation\b", summary, flags=re.IGNORECASE | re.MULTILINE):
        sections.extend(
            [
                "## Differentiation Opportunity",
                "- No differentiation opportunities were identified in this run.",
            ]
        )

    if not re.search(r"^##\s+Source\s+Quality\s+Notes\b", summary, flags=re.IGNORECASE | re.MULTILINE):
        warning = str(source_quality_warning or "").strip()
        if warning:
            sections.append("## Source Quality Notes")
            sections.append(f"- {warning}")

    if not re.search(r"^##\s+Rejected\s*/?\s*Weak\s+Sources\b", summary, flags=re.IGNORECASE | re.MULTILINE):
        weak_agents: list[str] = []
        for row in findings or []:
            if not isinstance(row, dict):
                continue
            confidence = row.get("confidence")
            try:
                score = int(confidence)
            except Exception:
                score = 0
            if score > 0 and score <= 2:
                weak_agents.append(str(row.get("agent", "agent")))
        if weak_agents:
            sections.append("## Rejected/Weak Sources")
            for name in sorted(set(weak_agents)):
                sections.append(f"- Weak support from {name}; excluded from final recommendations.")

    if analogy_sources and not re.search(r"^##\s+Design\s+Inspiration\b", summary, flags=re.IGNORECASE | re.MULTILINE):
        sections.append("## Design Inspiration (analogies - non-evidence)")
        for url in analogy_sources[:8]:
            sections.append(f"- [A] {url}")

    if not sections:
        return summary
    return f"{summary}\n\n" + "\n".join(sections)


__all__ = [
    "CHAIN_STAGES",
    "RECOMMENDATION_STRENGTH_LABELS",
    "enforce_final_summary_contract",
    "ensure_project_specificity",
    "run_translation_chain",
]
