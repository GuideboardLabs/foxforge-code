from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.common import ROOT  # noqa: F401
from shared_tools.workspace_knowledge import (
    read_workspace_knowledge,
    resolve_default_patterns_path,
)


class WorkspaceKnowledgeDefaultsTests(unittest.TestCase):
    def test_resolve_default_patterns_path_for_known_stack(self) -> None:
        path = resolve_default_patterns_path(
            ROOT,
            {
                "backend": "fastapi",
                "frontend": "react",
                "database": "postgres",
                "language": "python",
            },
        )
        self.assertIsNotNone(path)
        self.assertTrue(path.is_file())
        self.assertTrue(str(path).endswith("python-webapp__fastapi__react__postgres/PATTERNS.md"))

    def test_uses_stack_default_patterns_when_workspace_has_no_patterns_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "FOXFORGE.md").write_text("Project marker", encoding="utf-8")

            default_patterns = ROOT / "SourceCode" / "scaffolds" / "cli-tool__none__none__json-file" / "PATTERNS.md"
            text = read_workspace_knowledge(
                workspace,
                default_patterns_path=default_patterns,
            )

            self.assertIn("--- [FOXFORGE.md] ---", text)
            self.assertIn("--- [PATTERNS.md (stack default)] ---", text)
            self.assertIn("Python CLI + JSON storage", text)

    def test_workspace_patterns_override_stack_default_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "PATTERNS.md").write_text("workspace patterns", encoding="utf-8")

            default_patterns = ROOT / "SourceCode" / "scaffolds" / "cli-tool__none__none__json-file" / "PATTERNS.md"
            text = read_workspace_knowledge(
                workspace,
                default_patterns_path=default_patterns,
            )

            self.assertIn("--- [PATTERNS.md] ---", text)
            self.assertIn("workspace patterns", text)
            self.assertNotIn("PATTERNS.md (stack default)", text)

    def test_uses_default_design_when_workspace_has_no_design(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            text = read_workspace_knowledge(
                workspace,
                default_design_path=ROOT / "DESIGN.md",
            )

            self.assertIn("--- [DESIGN.md (default)] ---", text)
            self.assertIn("Default Project Design", text)


if __name__ == "__main__":
    unittest.main()
