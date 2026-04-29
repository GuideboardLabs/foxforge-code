from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopicPolicy:
    topic_type: str
    default_roles: tuple[str, ...]
    optional_roles: tuple[str, ...]
    disallowed_by_default: tuple[str, ...]
    confidence_cap: float | None
    preferred_source_tiers: tuple[str, ...]


TOPIC_POLICIES: dict[str, TopicPolicy] = {
    "_default": TopicPolicy(
        topic_type="_default",
        default_roles=(
            "context_and_background_researcher",
            "critical_analyst",
            "implications_researcher",
        ),
        optional_roles=("quantitative_evidence_analyst",),
        disallowed_by_default=("legal_compliance_researcher",),
        confidence_cap=None,
        preferred_source_tiers=("tier1", "tier2"),
    ),
    "general": TopicPolicy(
        topic_type="general",
        default_roles=(
            "context_and_background_researcher",
            "critical_analyst",
            "implications_researcher",
        ),
        optional_roles=("quantitative_evidence_analyst",),
        disallowed_by_default=("legal_compliance_researcher",),
        confidence_cap=None,
        preferred_source_tiers=("tier1", "tier2"),
    ),
    "technical": TopicPolicy(
        topic_type="technical",
        default_roles=(
            "technical_feasibility_researcher",
            "critical_analyst",
            "project_translator",
        ),
        optional_roles=("comparative_market_researcher", "quantitative_evidence_analyst"),
        disallowed_by_default=(),
        confidence_cap=None,
        preferred_source_tiers=("tier1", "tier2"),
    ),
    "animal_care": TopicPolicy(
        topic_type="animal_care",
        default_roles=(
            "domain_practitioner_researcher",
            "end_user_researcher",
            "resource_scout",
            "critical_analyst",
        ),
        optional_roles=("safety_risk_researcher", "standards_certification_researcher", "quantitative_evidence_analyst"),
        disallowed_by_default=("clinical_evidence_researcher", "legal_compliance_researcher"),
        confidence_cap=None,
        preferred_source_tiers=("tier1", "tier2", "tier3"),
    ),
    "medical": TopicPolicy(
        topic_type="medical",
        default_roles=("clinical_evidence_researcher", "guideline_verifier", "safety_risk_researcher", "critical_analyst"),
        optional_roles=("quantitative_evidence_analyst", "legal_compliance_researcher"),
        disallowed_by_default=(),
        confidence_cap=0.85,
        preferred_source_tiers=("tier1", "tier2"),
    ),
    "finance": TopicPolicy(
        topic_type="finance",
        default_roles=("comparative_market_researcher", "critical_analyst", "implications_researcher"),
        optional_roles=("quantitative_evidence_analyst", "legal_compliance_researcher"),
        disallowed_by_default=(),
        confidence_cap=None,
        preferred_source_tiers=("tier1", "tier2"),
    ),
    "sports": TopicPolicy(
        topic_type="sports",
        default_roles=("context_and_background_researcher", "quantitative_evidence_analyst", "critical_analyst"),
        optional_roles=(),
        disallowed_by_default=(),
        confidence_cap=None,
        preferred_source_tiers=("tier1", "tier2", "tier3"),
    ),
    "science": TopicPolicy(
        topic_type="science",
        default_roles=("context_and_background_researcher", "critical_analyst", "quantitative_evidence_analyst"),
        optional_roles=(),
        disallowed_by_default=(),
        confidence_cap=None,
        preferred_source_tiers=("tier1", "tier2"),
    ),
    "history": TopicPolicy(
        topic_type="history",
        default_roles=("context_and_background_researcher", "critical_analyst", "implications_researcher"),
        optional_roles=(),
        disallowed_by_default=(),
        confidence_cap=None,
        preferred_source_tiers=("tier1", "tier2", "tier3"),
    ),
    "politics": TopicPolicy(
        topic_type="politics",
        default_roles=("context_and_background_researcher", "critical_analyst", "implications_researcher"),
        optional_roles=("legal_compliance_researcher",),
        disallowed_by_default=(),
        confidence_cap=None,
        preferred_source_tiers=("tier1", "tier2", "tier3"),
    ),
    "current_events": TopicPolicy(
        topic_type="current_events",
        default_roles=("context_and_background_researcher", "critical_analyst", "implications_researcher"),
        optional_roles=("quantitative_evidence_analyst",),
        disallowed_by_default=(),
        confidence_cap=0.75,
        preferred_source_tiers=("tier1", "tier2", "tier3"),
    ),
    "law": TopicPolicy(
        topic_type="law",
        default_roles=("legal_compliance_researcher", "critical_analyst", "implications_researcher"),
        optional_roles=(),
        disallowed_by_default=(),
        confidence_cap=0.8,
        preferred_source_tiers=("tier1", "tier2"),
    ),
    "education": TopicPolicy(
        topic_type="education",
        default_roles=("domain_practitioner_researcher", "end_user_researcher", "critical_analyst"),
        optional_roles=("quantitative_evidence_analyst",),
        disallowed_by_default=(),
        confidence_cap=None,
        preferred_source_tiers=("tier1", "tier2", "tier3"),
    ),
    "domain": TopicPolicy(
        topic_type="domain",
        default_roles=("domain_practitioner_researcher", "end_user_researcher", "resource_scout", "critical_analyst"),
        optional_roles=("standards_certification_researcher", "quantitative_evidence_analyst"),
        disallowed_by_default=("legal_compliance_researcher",),
        confidence_cap=None,
        preferred_source_tiers=("tier1", "tier2", "tier3"),
    ),
}


PROJECT_TYPE_ROLE_HINTS: dict[str, tuple[str, ...]] = {
    "webapp": ("technical_feasibility_researcher", "project_translator"),
    "api": ("technical_feasibility_researcher", "project_translator"),
    "desktop": ("technical_feasibility_researcher", "project_translator"),
    "library": ("technical_feasibility_researcher", "project_translator"),
    "tool": ("technical_feasibility_researcher", "project_translator"),
}


RESEARCH_INTENT_ROLE_HINTS: dict[str, tuple[str, ...]] = {
    "general_research": (),
    "domain_foraging": ("domain_practitioner_researcher", "end_user_researcher", "resource_scout", "critical_analyst"),
    "product_research": ("comparative_market_researcher", "critical_analyst", "project_translator"),
    "technical_planning": ("technical_feasibility_researcher", "project_translator", "critical_analyst"),
    "market_research": ("comparative_market_researcher", "critical_analyst"),
    "standards_research": ("standards_certification_researcher", "critical_analyst"),
    "risk_research": ("safety_risk_researcher", "critical_analyst"),
    "final_synthesis": ("critical_analyst", "evidence_adjudicator", "contrarian_red_team"),
}


def topic_policy_for(topic_type: str) -> TopicPolicy:
    key = str(topic_type or "").strip().lower() or "_default"
    return TOPIC_POLICIES.get(key) or TOPIC_POLICIES["_default"]


__all__ = [
    "PROJECT_TYPE_ROLE_HINTS",
    "RESEARCH_INTENT_ROLE_HINTS",
    "TOPIC_POLICIES",
    "TopicPolicy",
    "topic_policy_for",
]
