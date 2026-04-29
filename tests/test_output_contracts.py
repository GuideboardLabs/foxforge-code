from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from agents_research.output_contracts import (
    render_domain_summary,
    render_implementation_brief,
    render_project_summary,
    render_research_summary,
)


class OutputContractsTests(unittest.TestCase):
    def test_renderers_emit_required_sections(self) -> None:
        primitives = {
            "capabilities": ["c1"],
            "milestones": ["m1"],
            "success_criteria": ["s1"],
            "failure_modes": ["f1"],
            "measurement_dimensions": ["d1"],
        }
        recs = [{"finding": "Do X", "strength": "prototype"}]

        research = render_research_summary(synthesis_md="## Executive Summary\ntext")
        domain = render_domain_summary(synthesis_md="", claims=[], primitives=primitives, recommendations=recs)
        project = render_project_summary(synthesis_md="", claims=[], primitives=primitives, recommendations=recs)
        brief = render_implementation_brief(synthesis_md="", claims=[], primitives=primitives, recommendations=recs)

        self.assertIn("## Executive Summary", research)
        self.assertIn("# Domain Summary", domain)
        self.assertIn("# Project Summary", project)
        self.assertIn("# Implementation Brief", brief)

    def test_empty_inputs_return_stub_instead_of_crashing(self) -> None:
        out = render_domain_summary(synthesis_md="", claims=[], primitives=None, recommendations=[])
        self.assertIn("No structured data", out)


if __name__ == "__main__":
    unittest.main()
