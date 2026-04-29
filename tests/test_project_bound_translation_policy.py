from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from agents_research.translation_chain import run_translation_chain


class _Client:
    def chat(self, **_kwargs):
        return "- translated implication"


class ProjectBoundTranslationPolicyTests(unittest.TestCase):
    def test_non_technical_intent_uses_possible_technical_implications_heading(self) -> None:
        out = run_translation_chain(
            question="What should this project focus on next?",
            synthesis_summary="## Executive Summary\nBase summary",
            project_context="Active coding project: demo",
            client=_Client(),
            repo_root=ROOT,
            synthesis_cfg={"model": "qwen3:8b", "num_ctx": 4096, "timeout_sec": 60},
            stack_specific_actions=False,
        )
        summary = str(out.get("summary", ""))
        self.assertIn("### Possible Technical Implications", summary)
        self.assertNotIn("### Tech Lead", summary)

    def test_technical_intent_keeps_tech_lead_heading(self) -> None:
        out = run_translation_chain(
            question="How should we implement this in the current stack?",
            synthesis_summary="## Executive Summary\nBase summary",
            project_context="Active coding project: demo",
            client=_Client(),
            repo_root=ROOT,
            synthesis_cfg={"model": "qwen3:8b", "num_ctx": 4096, "timeout_sec": 60},
            stack_specific_actions=True,
        )
        summary = str(out.get("summary", ""))
        self.assertIn("### Tech Lead", summary)


if __name__ == "__main__":
    unittest.main()
