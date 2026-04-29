from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoleSpec:
    role_id: str
    label: str
    directive: str
    default_model_key: str
    escalation_model_key: str | None
    output_contract_id: str | None
    requires_triggers: tuple[str, ...]
    forbidden_when: tuple[str, ...]
    role_class: str
    validation_cycles: int = 3
    legacy: bool = False
    claims_allowed: bool = True


def _r(
    role_id: str,
    label: str,
    directive: str,
    default_model_key: str,
    *,
    escalation_model_key: str | None = None,
    output_contract_id: str | None = None,
    requires_triggers: tuple[str, ...] = (),
    forbidden_when: tuple[str, ...] = (),
    role_class: str = "primary",
    validation_cycles: int = 3,
    legacy: bool = False,
    claims_allowed: bool = True,
) -> RoleSpec:
    return RoleSpec(
        role_id=role_id,
        label=label,
        directive=directive,
        default_model_key=default_model_key,
        escalation_model_key=escalation_model_key,
        output_contract_id=output_contract_id,
        requires_triggers=requires_triggers,
        forbidden_when=forbidden_when,
        role_class=role_class,
        validation_cycles=validation_cycles,
        legacy=legacy,
        claims_allowed=claims_allowed,
    )


ROLE_SPECS: dict[str, RoleSpec] = {
    "critical_analyst": _r(
        "critical_analyst",
        "Critical Analyst",
        "Challenge weak assumptions, identify contradictions, and prioritize the strongest falsifiers from peer findings.",
        "research_critical_analyst",
        escalation_model_key="research_critical_analyst_premium",
        output_contract_id="critical_analyst_v1",
        role_class="adjudicator",
    ),
    "context_and_background_researcher": _r(
        "context_and_background_researcher",
        "Context & Background Researcher",
        "Map actors, timeline, and baseline domain context without over-claiming.",
        "research_context_background",
    ),
    "implications_researcher": _r(
        "implications_researcher",
        "Implications Researcher",
        "Extract second-order implications and project-relevant consequences.",
        "research_implications",
    ),
    "evidence_adjudicator": _r(
        "evidence_adjudicator",
        "Evidence Adjudicator",
        "Audit claims for support quality, source-tier mismatch, and echo risk before synthesis.",
        "research_evidence_adjudicator",
        output_contract_id="evidence_adjudicator_v1",
        role_class="adjudicator",
    ),
    "project_translator": _r(
        "project_translator",
        "Project Translator",
        "Translate validated findings into concrete implementation implications for the active project.",
        "research_project_translator",
    ),
    "domain_practitioner_researcher": _r(
        "domain_practitioner_researcher",
        "Domain Practitioner Researcher",
        "Focus on practitioner methods, standards of practice, and what experts actually do.",
        "research_domain_practitioner",
    ),
    "end_user_researcher": _r(
        "end_user_researcher",
        "End User Researcher",
        "Focus on user friction, adherence, motivation, and real usage constraints.",
        "research_end_user",
    ),
    "resource_scout": _r(
        "resource_scout",
        "Resource Scout",
        "Catalog canonical resources, curricula, and standards. Do not assert claims as evidence.",
        "research_resource_scout",
        role_class="advisory",
        claims_allowed=False,
    ),
    "safety_risk_researcher": _r(
        "safety_risk_researcher",
        "Safety & Risk Researcher",
        "Identify concrete hazards, failure modes, and mitigations.",
        "research_safety_risk",
        requires_triggers=("poison", "toxic", "aggression", "medication", "injury", "risk"),
    ),
    "technical_feasibility_researcher": _r(
        "technical_feasibility_researcher",
        "Technical Feasibility Researcher",
        "Assess implementation feasibility, constraints, and architecture tradeoffs.",
        "research_technical_feasibility",
    ),
    "comparative_market_researcher": _r(
        "comparative_market_researcher",
        "Comparative Market Researcher",
        "Compare alternatives, ecosystem maturity, and market signals.",
        "research_comparative_market",
    ),
    "standards_certification_researcher": _r(
        "standards_certification_researcher",
        "Standards & Certification Researcher",
        "Check standards, credentials, and certification pathways.",
        "research_standards_certification",
        requires_triggers=("certification", "cpdt", "iaabc", "akc", "standards", "credentials"),
    ),
    "quantitative_evidence_analyst": _r(
        "quantitative_evidence_analyst",
        "Quantitative Evidence Analyst",
        "Return ONLY quantitative findings with source and uncertainty context.",
        "research_quant_evidence",
        output_contract_id="quant_evidence_v1",
    ),
    "legal_compliance_researcher": _r(
        "legal_compliance_researcher",
        "Legal Compliance Researcher",
        "Focus on legal and compliance constraints, jurisdiction caveats, and explicit risk language.",
        "research_legal_compliance",
        requires_triggers=(
            "privacy", "data retention", "hipaa", "coppa", "gdpr", "contract", "employment", "liability",
            "certification", "regulated", "terms of service", "license", "license-compatible", "dmca", "ada", "pci", "soc 2", "ccpa",
        ),
        role_class="advisory",
    ),
    "contrarian_red_team": _r(
        "contrarian_red_team",
        "Contrarian Red Team",
        "Challenge framing, dominant consensus, and hidden assumptions. Present strongest counter-case.",
        "research_contrarian_red_team",
        escalation_model_key="research_contrarian_red_team_premium",
        output_contract_id="contrarian_red_team_v1",
        role_class="adjudicator",
    ),
    # Legacy role entries retained for compatibility.
    "clinical_evidence_researcher": _r(
        "clinical_evidence_researcher",
        "Clinical Evidence Researcher",
        "Focus on peer-reviewed evidence quality, trial design, and guideline alignment.",
        "research_technical",
        legacy=True,
    ),
    "guideline_verifier": _r(
        "guideline_verifier",
        "Guideline Verifier",
        "Cross-check guidance against current official recommendations and revision status.",
        "research_market_analyst",
        legacy=True,
    ),
    "statistical_analysis": _r(
        "statistical_analysis",
        "Statistical Analysis",
        "Return ONLY quantitative findings from sources.",
        "research_statistical",
        legacy=True,
    ),
}


def get_role_spec(role_id: str) -> RoleSpec | None:
    return ROLE_SPECS.get(str(role_id or "").strip())


def all_role_specs(*, include_legacy: bool = True) -> list[RoleSpec]:
    rows = list(ROLE_SPECS.values())
    if include_legacy:
        return rows
    return [row for row in rows if not row.legacy]


def resolve_model_name(
    *,
    role: RoleSpec,
    config_getter: Any,
    escalated: bool = False,
    fallback_model: str = "qwen3:8b",
) -> str:
    key = role.escalation_model_key if escalated and role.escalation_model_key else role.default_model_key
    clean_key = str(key or "").strip()
    if not clean_key:
        return str(fallback_model or "qwen3:8b").strip() or "qwen3:8b"
    try:
        model = str(config_getter(clean_key)).strip()
        if model:
            return model
    except Exception:
        pass
    return str(fallback_model or "qwen3:8b").strip() or "qwen3:8b"


__all__ = ["RoleSpec", "ROLE_SPECS", "all_role_specs", "get_role_spec", "resolve_model_name"]
