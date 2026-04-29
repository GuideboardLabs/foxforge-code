from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from agents_research.deep_researcher import _agent_specs, _legal_role_applies, _quantitative_role_applies


class RoleGatingTests(unittest.TestCase):
    def test_legal_role_applies_false_for_puppy_schedule(self) -> None:
        self.assertFalse(_legal_role_applies("Best schedule for puppy crate training", "animal_care", ""))

    def test_legal_role_applies_true_for_gdpr_retention(self) -> None:
        self.assertTrue(_legal_role_applies("GDPR retention for user-uploaded files", "technical", ""))

    def test_animal_care_specs_exclude_medical_and_legal_defaults(self) -> None:
        specs = _agent_specs(
            {"model": "qwen3:8b", "fallback_models": []},
            topic_type="animal_care",
            question="Best schedule for puppy crate training",
            project_context="",
            research_intent="domain_foraging",
            workspace_knowledge="",
            repo_root=ROOT,
        )
        personas = {str(row.get("persona", "")).strip() for row in specs}
        self.assertNotIn("clinical_evidence_researcher", personas)
        self.assertNotIn("legal_analysis", personas)
        self.assertNotIn("legal_compliance_researcher", personas)

    def test_medical_specs_no_legal_without_trigger(self) -> None:
        specs = _agent_specs(
            {"model": "qwen3:8b", "fallback_models": []},
            topic_type="medical",
            question="How to improve sleep quality",
            project_context="",
            research_intent="general_research",
            workspace_knowledge="",
            repo_root=ROOT,
        )
        personas = {str(row.get("persona", "")).strip() for row in specs}
        self.assertNotIn("legal_analysis", personas)
        self.assertNotIn("legal_compliance_researcher", personas)

    def test_quant_role_false_for_non_numeric_prompt(self) -> None:
        self.assertFalse(_quantitative_role_applies("What are common onboarding mistakes for a new team?", "general"))


if __name__ == "__main__":
    unittest.main()
