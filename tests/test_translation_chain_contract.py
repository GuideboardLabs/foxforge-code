from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from agents_research.translation_chain import enforce_final_summary_contract, ensure_project_specificity


class TranslationChainContractTests(unittest.TestCase):
    def test_project_specificity_appends_context_anchors_when_generic(self) -> None:
        summary = "## Executive Summary\nGeneral advice without project details."
        context = (
            "Active coding project: Milebones\n"
            "Stack: {\"backend\": \"flask\", \"frontend\": \"vue\"}\n"
            "Workspace: /home/sc/Foxforge-code/Projects/milebones"
        )
        out = ensure_project_specificity(summary, context)
        self.assertIn("## Project Context Anchors", out)
        self.assertIn("Milebones", out)

    def test_enforce_contract_adds_missing_sections(self) -> None:
        summary = (
            "## Executive Summary\n"
            "Evidence Confidence: Mixed - short.\n\n"
            "## Key Findings\n"
            "- [E] Supported finding.\n"
        )
        out = enforce_final_summary_contract(
            summary_md=summary,
            findings=[{"agent": "a1", "confidence": 2}],
            source_quality_warning="tier3-heavy evidence",
            analogy_sources=["https://example.com/analogy"],
        )
        self.assertIn("## Evidence / Inference / Speculation Buckets", out)
        self.assertIn("## Recommended Actions", out)
        self.assertIn("## Source Quality Notes", out)
        self.assertIn("## Rejected/Weak Sources", out)
        self.assertIn("## Design Inspiration (analogies - non-evidence)", out)


if __name__ == "__main__":
    unittest.main()
