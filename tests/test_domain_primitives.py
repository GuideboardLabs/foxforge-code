from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.common import ROOT  # noqa: F401
from agents_research.domain_primitives import extract_primitives, persist_primitives


class _Client:
    def chat(self, **_kwargs) -> str:
        return json.dumps(
            {
                "milestones": ["m1"],
                "success_criteria": ["s1"],
                "failure_modes": ["f1"],
                "measurement_dimensions": ["d1"],
                "capabilities": ["c1"],
                "notes": "ok",
            }
        )


class DomainPrimitivesTests(unittest.TestCase):
    def test_extract_schema_for_enabled_intent(self) -> None:
        out = extract_primitives(
            question="q",
            synthesis_md="summary",
            claims=[{"claim": "x"}],
            client=_Client(),
            model_cfg={"model": "qwen3:8b"},
            research_intent="technical_planning",
        )
        self.assertTrue(out.get("enabled"))
        self.assertIn("milestones", out)
        self.assertIn("success_criteria", out)

    def test_disabled_for_general_research(self) -> None:
        out = extract_primitives(
            question="q",
            synthesis_md="summary",
            claims=[],
            client=_Client(),
            model_cfg={"model": "qwen3:8b"},
            research_intent="general_research",
        )
        self.assertFalse(out.get("enabled"))

    def test_persist_primitives_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            path = persist_primitives(
                repo_root=repo,
                project_slug="demo",
                summary_path="20260101_research_summary.md",
                primitives={"enabled": True, "milestones": ["m1"]},
            )
            self.assertTrue(Path(path).is_file())
            body = Path(path).read_text(encoding="utf-8")
            self.assertIn("milestones", body)


if __name__ == "__main__":
    unittest.main()
