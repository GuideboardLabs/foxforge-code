from __future__ import annotations

import json
import unittest

from tests.common import ROOT  # noqa: F401
from agents_research.genericity_gate import score_artifact


class _Client:
    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or {
            "genericity_score": 4,
            "usefulness_score": 1,
            "generic_phrases": ["smart goals", "follow agile"],
            "missing_specifics": ["no project constraints"],
            "reasoning": "too generic",
        }

    def chat(self, **_kwargs) -> str:
        return json.dumps(self.payload)


class GenericityGateTests(unittest.TestCase):
    def test_generic_stub_scores_high(self) -> None:
        out = score_artifact(
            artifact_kind="project_summary",
            artifact_md="Use SMART goals. Follow Agile. Iterate quickly.",
            primitives={"milestones": []},
            project_context="Project: demo",
            workspace_knowledge="",
            client=_Client(),
            model_cfg={"model": "qwen3:8b"},
        )
        self.assertGreaterEqual(int(out.get("genericity_score", 0)), 4)

    def test_primitive_grounded_scores_lower(self) -> None:
        low_client = _Client(
            {
                "genericity_score": 1,
                "usefulness_score": 4,
                "generic_phrases": [],
                "missing_specifics": [],
                "reasoning": "grounded",
            }
        )
        out = score_artifact(
            artifact_kind="project_summary",
            artifact_md="Milestone: dog sits on first cue in 4 of 5 trials across rooms.",
            primitives={"milestones": ["dog sits 4/5"]},
            project_context="dog training app",
            workspace_knowledge="",
            client=low_client,
            model_cfg={"model": "qwen3:8b"},
        )
        self.assertLessEqual(int(out.get("genericity_score", 5)), 2)


if __name__ == "__main__":
    unittest.main()
