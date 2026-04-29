from __future__ import annotations

from typing import Any


def _no_data_stub(title: str) -> str:
    return f"# {title}\n\n_No structured data was available for this artifact._\n"


def render_research_summary(
    *,
    synthesis_md: str,
    claims: list[dict[str, Any]] | None = None,
    primitives: dict[str, Any] | None = None,
    recommendations: list[dict[str, Any]] | None = None,
    project_metadata: dict[str, Any] | None = None,
) -> str:
    body = str(synthesis_md or "").strip()
    if body:
        return body
    return _no_data_stub("Research Summary")


def render_domain_summary(
    *,
    synthesis_md: str,
    claims: list[dict[str, Any]] | None,
    primitives: dict[str, Any] | None,
    recommendations: list[dict[str, Any]] | None,
    project_metadata: dict[str, Any] | None = None,
) -> str:
    if not primitives:
        return _no_data_stub("Domain Summary")
    lines = ["# Domain Summary", "", "## Domain Primitives"]
    for key in ("capabilities", "milestones", "success_criteria", "failure_modes", "measurement_dimensions"):
        lines.append(f"### {key.replace('_', ' ').title()}")
        values = primitives.get(key) if isinstance(primitives, dict) else []
        if isinstance(values, list) and values:
            lines.extend(f"- {str(v).strip()}" for v in values if str(v).strip())
        else:
            lines.append("- none")
    return "\n".join(lines).strip() + "\n"


def render_project_summary(
    *,
    synthesis_md: str,
    claims: list[dict[str, Any]] | None,
    primitives: dict[str, Any] | None,
    recommendations: list[dict[str, Any]] | None,
    project_metadata: dict[str, Any] | None = None,
) -> str:
    lines = ["# Project Summary", "", "## Project Context"]
    meta = project_metadata or {}
    name = str(meta.get("name", meta.get("slug", "project"))).strip()
    stack = meta.get("stack") if isinstance(meta.get("stack"), dict) else {}
    lines.append(f"- Project: {name or 'project'}")
    if stack:
        lines.append(f"- Stack: {stack}")
    lines.extend(["", "## Recommended Actions"])
    if recommendations:
        for row in recommendations:
            finding = str(row.get("finding", "")).strip() or str(row.get("implication", "")).strip() or "Action"
            strength = str(row.get("strength", "design_option")).strip().lower() or "design_option"
            lines.append(f"- {finding} - strength: {strength}")
    else:
        lines.append("- none")
    if primitives:
        lines.extend(["", "## Primitive Anchors"])
        for key in ("milestones", "success_criteria", "failure_modes"):
            values = primitives.get(key) if isinstance(primitives, dict) else []
            if isinstance(values, list) and values:
                lines.append(f"### {key.replace('_', ' ').title()}")
                lines.extend(f"- {str(v).strip()}" for v in values if str(v).strip())
    return "\n".join(lines).strip() + "\n"


def render_implementation_brief(
    *,
    synthesis_md: str,
    claims: list[dict[str, Any]] | None,
    primitives: dict[str, Any] | None,
    recommendations: list[dict[str, Any]] | None,
    project_metadata: dict[str, Any] | None = None,
) -> str:
    lines = ["# Implementation Brief", "", "## Build Plan"]
    if recommendations:
        for row in recommendations[:12]:
            finding = str(row.get("finding", "")).strip() or str(row.get("implication", "")).strip() or "Action"
            strength = str(row.get("strength", "design_option")).strip().lower() or "design_option"
            lines.append(f"- {finding} - strength: {strength}")
    else:
        lines.append("- No prioritized implementation actions were generated.")

    if primitives:
        lines.extend(["", "## Constraints & Failure Modes"])
        failures = primitives.get("failure_modes") if isinstance(primitives, dict) else []
        if isinstance(failures, list) and failures:
            lines.extend(f"- {str(v).strip()}" for v in failures if str(v).strip())
        else:
            lines.append("- none")
    return "\n".join(lines).strip() + "\n"


__all__ = [
    "render_domain_summary",
    "render_implementation_brief",
    "render_project_summary",
    "render_research_summary",
]
