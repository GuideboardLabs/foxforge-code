from __future__ import annotations

import json
import unittest

from tests.common import ROOT  # noqa: F401
from agents_research.translation_chain import run_translation_chain


class _Client:
    def chat(self, **kwargs):
        task = str(kwargs.get("task_class", "")).strip()
        if task == "recommendation_strength":
            return json.dumps(
                [
                    {
                        "finding": "Use queue backpressure",
                        "implication": "Prevent dropped events",
                        "strength": "implement_now",
                        "rationale": "high confidence",
                        "evidence_ref": ["claim_1"],
                        "primitive_ref": ["m1"],
                    }
                ]
            )
        return "- Stage output bullet"


class RecommendationStrengthTests(unittest.TestCase):
    def test_labels_from_allowed_enum_rendered(self) -> None:
        out = run_translation_chain(
            question="How should we process jobs?",
            synthesis_summary="## Executive Summary\nBase summary",
            project_context="Active coding project: demo",
            client=_Client(),
            repo_root=ROOT,
            synthesis_cfg={"model": "qwen3:8b", "num_ctx": 4096, "timeout_sec": 60},
            use_premium_tech_lead=False,
            primitives={"milestones": ["m1"]},
        )
        summary = str(out.get("summary", ""))
        self.assertIn("## Recommended Actions (with strength labels)", summary)
        self.assertIn("strength: implement_now", summary)


if __name__ == "__main__":
    unittest.main()
