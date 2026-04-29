from __future__ import annotations

import re
from typing import Any


_CLAIM_LINE_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*(\[[EISA]\])?\s*(.+)$")


def extract_claims(findings: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    extraction_quality = "clean"
    for row in findings:
        if not isinstance(row, dict):
            continue
        agent = str(row.get("agent", "agent")).strip()
        model = str(row.get("model", "")).strip()
        body = str(row.get("finding", "")).strip()
        if not body:
            continue
        matched_in_row = 0
        for line in body.splitlines():
            m = _CLAIM_LINE_RE.match(line)
            if not m:
                continue
            label = str(m.group(1) or "").strip() or "[I]"
            claim_text = str(m.group(2) or "").strip()
            if not claim_text:
                continue
            claims.append(
                {
                    "claim_id": f"{agent}:{len(claims)+1}",
                    "agent": agent,
                    "model": model,
                    "label": label,
                    "claim": claim_text,
                }
            )
            matched_in_row += 1
        if matched_in_row == 0 and body:
            extraction_quality = "needs_structuring"
            claims.append(
                {
                    "claim_id": f"{agent}:{len(claims)+1}",
                    "agent": agent,
                    "model": model,
                    "label": "[I]",
                    "claim": body[:400],
                }
            )

    telemetry = {
        "claims_count": len(claims),
        "extraction_quality": extraction_quality if claims else "partial",
    }
    return claims, telemetry


__all__ = ["extract_claims"]
