from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from agents_research.synthesizer import synthesize


def _base_summary(confidence: str = "High") -> str:
    return (
        "# Research Synthesis\n\n"
        "## Executive Summary\n"
        f"Evidence Confidence: {confidence} - baseline summary before post-processing. "
        "This section includes enough detail to satisfy validation heuristics for synthesis length and structure.\n\n"
        "## Key Findings\n"
        "- [E] A grounded claim with cited support.\n"
        "- [I] A secondary inference from the evidence.\n"
        "\n## Uncertainties & Risks\n"
        "- Source limitations remain and should be explicitly acknowledged.\n"
        "\n## Next Steps\n"
        "- Prioritize the highest confidence recommendation and validate in the project context.\n"
    )


class _Client:
    def __init__(self, confidence: str = "High") -> None:
        self._confidence = confidence

    def wait_for_available(self, model: str, **_kwargs) -> bool:  # noqa: ARG002
        return True

    def chat(self, **_kwargs) -> str:
        return _base_summary(self._confidence)


class SynthesizerTierConfidenceCapTests(unittest.TestCase):
    def test_all_tier3_caps_to_mixed_and_adds_warning(self) -> None:
        out = synthesize(
            "question",
            [
                {
                    "agent": "a1",
                    "finding": "- [E] claim [source: https://example.com/a]",
                    "source_tier_counts": {"tier1": 0, "tier2": 0, "tier3": 3, "tier4": 0},
                }
            ],
            client=_Client(confidence="High"),
            model_cfg={
                "model": "qwen3:8b",
                "synthesis_validation_cycles": 1,
                "synthesis_retry_attempts": 1,
            },
        )
        self.assertIn("## Source Quality Warning", out)
        self.assertIn("Evidence Confidence: Mixed", out)

    def test_mostly_tier4_caps_to_low(self) -> None:
        out = synthesize(
            "question",
            [
                {
                    "agent": "a1",
                    "finding": "- [A] analogy [source: https://example.com/a]",
                    "source_tier_counts": {"tier1": 0, "tier2": 0, "tier3": 1, "tier4": 4},
                }
            ],
            client=_Client(confidence="High"),
            model_cfg={
                "model": "qwen3:8b",
                "synthesis_validation_cycles": 1,
                "synthesis_retry_attempts": 1,
            },
        )
        self.assertIn("## Source Quality Warning", out)
        self.assertIn("Evidence Confidence: Low", out)


if __name__ == "__main__":
    unittest.main()
