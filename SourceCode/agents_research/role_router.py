from __future__ import annotations

import logging
import re
from typing import Any

from agents_research.role_registry import RoleSpec, get_role_spec
from agents_research.topic_policy import PROJECT_TYPE_ROLE_HINTS, RESEARCH_INTENT_ROLE_HINTS, topic_policy_for

LOGGER = logging.getLogger(__name__)


LEGAL_TRIGGER_WORDS: tuple[str, ...] = (
    "privacy", "data retention", "hipaa", "coppa", "gdpr", "contract", "employment", "liability",
    "certification", "regulated", "terms of service", "license", "license-compatible", "dmca", "ada", "pci", "soc 2", "ccpa",
)

QUANT_TRIGGER_WORDS: tuple[str, ...] = (
    "rate", "%", "percent", "benchmark", "sample", "sample-size", "sample size", "median", "mean", "variance",
    "p-value", "effect-size", "effect size", "confidence interval", "stddev", "distribution", "ratio",
)

QUANT_TOPICS: set[str] = {"finance", "sports", "science", "current_events"}


def _contains_term(text: str, term: str) -> bool:
    raw = str(text or "")
    needle = str(term or "").strip()
    if not raw or not needle:
        return False
    if re.search(r"[^a-zA-Z0-9]", needle):
        return needle.lower() in raw.lower()
    return re.search(rf"\b{re.escape(needle)}\b", raw, re.IGNORECASE) is not None


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(_contains_term(text, term) for term in terms)


def legal_role_applies(*, question: str, topic_type: str, project_context: str = "") -> bool:
    corpus = "\n".join([str(question or ""), str(topic_type or ""), str(project_context or "")]).strip()
    return _contains_any(corpus, LEGAL_TRIGGER_WORDS)


def quantitative_role_applies(*, question: str, topic_type: str) -> bool:
    topic = str(topic_type or "").strip().lower()
    if topic in QUANT_TOPICS:
        return True
    return _contains_any(str(question or ""), QUANT_TRIGGER_WORDS)


def _role_allowed_by_triggers(role: RoleSpec, *, corpus: str, topic_type: str, question: str, project_context: str) -> bool:
    if role.role_id == "legal_compliance_researcher":
        return legal_role_applies(question=question, topic_type=topic_type, project_context=project_context)
    if role.role_id in {"quantitative_evidence_analyst", "statistical_analysis"}:
        return quantitative_role_applies(question=question, topic_type=topic_type)

    triggers = tuple(str(x).strip() for x in role.requires_triggers if str(x).strip())
    if not triggers:
        return True
    return _contains_any(corpus, triggers)


def select_roles(
    *,
    topic_type: str,
    research_intent: str,
    project_type: str | None,
    user_query: str,
    user_overrides: list[str] | None,
    workspace_knowledge: str | None,
    project_context: str = "",
) -> list[RoleSpec]:
    topic_key = str(topic_type or "").strip().lower() or "general"
    intent_key = str(research_intent or "").strip().lower() or "general_research"
    project_key = str(project_type or "").strip().lower()
    corpus = "\n".join([str(user_query or ""), str(workspace_knowledge or ""), str(project_context or "")]).strip()

    policy = topic_policy_for(topic_key)

    chosen_ids: list[str] = []
    if user_overrides:
        for role_id in user_overrides:
            role = str(role_id or "").strip()
            if role and role not in chosen_ids:
                chosen_ids.append(role)

    if not chosen_ids:
        for role_id in policy.default_roles:
            if role_id not in chosen_ids:
                chosen_ids.append(role_id)

        for role_id in policy.optional_roles:
            if role_id not in chosen_ids:
                chosen_ids.append(role_id)

        for role_id in PROJECT_TYPE_ROLE_HINTS.get(project_key, ()):
            if role_id not in chosen_ids:
                chosen_ids.append(role_id)

        for role_id in RESEARCH_INTENT_ROLE_HINTS.get(intent_key, ()):
            if role_id not in chosen_ids:
                chosen_ids.append(role_id)

    out: list[RoleSpec] = []
    for role_id in chosen_ids:
        role = get_role_spec(role_id)
        if role is None:
            LOGGER.warning("role_router: unknown role '%s' for topic '%s'", role_id, topic_key)
            continue
        if topic_key in set(role.forbidden_when):
            continue
        if role.role_id in set(policy.disallowed_by_default) and not user_overrides:
            continue
        if not _role_allowed_by_triggers(
            role,
            corpus=corpus,
            topic_type=topic_key,
            question=user_query,
            project_context=project_context,
        ):
            continue
        out.append(role)

    if not out and not user_overrides:
        # Safe fallback to topic defaults when filtering removes every role.
        for role_id in policy.default_roles:
            role = get_role_spec(role_id)
            if role is None:
                continue
            if role.role_id in set(policy.disallowed_by_default):
                continue
            out.append(role)

    return out


__all__ = [
    "LEGAL_TRIGGER_WORDS",
    "QUANT_TRIGGER_WORDS",
    "legal_role_applies",
    "quantitative_role_applies",
    "select_roles",
]
