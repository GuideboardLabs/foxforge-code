from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from agents_research.role_router import select_roles


class RoleRouterTests(unittest.TestCase):
    def test_animal_care_excludes_clinical_and_legal_by_default(self) -> None:
        roles = select_roles(
            topic_type="animal_care",
            research_intent="domain_foraging",
            project_type=None,
            user_query="How should I structure puppy training sessions?",
            user_overrides=None,
            workspace_knowledge="",
            project_context="",
        )
        ids = {r.role_id for r in roles}
        self.assertNotIn("clinical_evidence_researcher", ids)
        self.assertNotIn("legal_compliance_researcher", ids)

    def test_user_override_beats_topic_policy(self) -> None:
        roles = select_roles(
            topic_type="general",
            research_intent="general_research",
            project_type=None,
            user_query="",
            user_overrides=["technical_feasibility_researcher"],
            workspace_knowledge="",
            project_context="",
        )
        ids = {r.role_id for r in roles}
        self.assertIn("technical_feasibility_researcher", ids)

    def test_trigger_gated_roles_only_when_keywords_present(self) -> None:
        no_trigger = select_roles(
            topic_type="general",
            research_intent="general_research",
            project_type=None,
            user_query="Help me plan a team workshop.",
            user_overrides=["legal_compliance_researcher"],
            workspace_knowledge="",
            project_context="",
        )
        self.assertEqual([], no_trigger)

        yes_trigger = select_roles(
            topic_type="general",
            research_intent="general_research",
            project_type=None,
            user_query="Need GDPR data retention policy guidance.",
            user_overrides=["legal_compliance_researcher"],
            workspace_knowledge="",
            project_context="",
        )
        self.assertEqual(["legal_compliance_researcher"], [r.role_id for r in yes_trigger])

    def test_technical_planning_selects_expected_roles(self) -> None:
        roles = select_roles(
            topic_type="technical",
            research_intent="technical_planning",
            project_type="webapp",
            user_query="Plan architecture for the stack",
            user_overrides=None,
            workspace_knowledge="",
            project_context="",
        )
        ids = {r.role_id for r in roles}
        self.assertIn("technical_feasibility_researcher", ids)
        self.assertIn("project_translator", ids)
        self.assertIn("critical_analyst", ids)


if __name__ == "__main__":
    unittest.main()
