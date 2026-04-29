from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests.common import ROOT  # noqa: F401
from shared_tools.web_research import source_tier_for_url


class SourceTierAssignmentTests(unittest.TestCase):
    def test_fixture_expected_tiers(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "source_tier_expectations.json"
        rows = json.loads(fixture.read_text(encoding="utf-8"))
        for row in rows:
            url = str(row["url"])
            expected = str(row["expected_tier"])
            actual = source_tier_for_url(url)
            self.assertEqual(expected, actual, msg=f"url={url}")


if __name__ == "__main__":
    unittest.main()
