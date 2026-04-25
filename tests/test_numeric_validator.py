from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from agents_research.numeric_validator import validate


class NumericValidatorTests(unittest.TestCase):
    def test_downgrades_unverified_numeric_evidence(self) -> None:
        findings = [
            {
                "agent": "critical_analyst",
                "finding": "- [E] 78% of users reported better retention [source: https://example.com/a]",
                "source_evidence": [
                    {
                        "url": "https://example.com/a",
                        "snippet": "A qualitative case study reported better retention outcomes.",
                    }
                ],
            }
        ]
        rows, report = validate(findings, source_evidence=findings[0]["source_evidence"])
        self.assertEqual(int(report.get("numeric_claims_seen", 0)), 1)
        self.assertEqual(int(report.get("numeric_claims_flagged", 0)), 1)
        self.assertIn("[S]", rows[0]["finding"])
        self.assertIn("unverified statistic", rows[0]["finding"].lower())

    def test_keeps_supported_numeric_evidence(self) -> None:
        findings = [
            {
                "agent": "context_and_background_researcher",
                "finding": "- [E] 42% reported improved adherence [source: https://example.com/b]",
                "source_evidence": [
                    {
                        "url": "https://example.com/b",
                        "snippet": "In 2025, 42 percent of participants reported improved adherence.",
                    }
                ],
            }
        ]
        rows, report = validate(findings, source_evidence=findings[0]["source_evidence"])
        self.assertEqual(int(report.get("numeric_claims_seen", 0)), 1)
        self.assertEqual(int(report.get("numeric_claims_flagged", 0)), 0)
        self.assertIn("[E]", rows[0]["finding"])


if __name__ == "__main__":
    unittest.main()
