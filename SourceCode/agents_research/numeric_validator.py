from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from shared_tools.model_routing import lane_model_config


_PERCENT_RE = re.compile(r"\b\d{1,3}(?:\.\d+)?\s*%")
_RATIO_RE = re.compile("\\b\\d+(?:\\.\\d+)?\\s*(?:x|\\u00d7)\\b", re.IGNORECASE)
_COUNT_RE = re.compile(r"\b(?:n\s*=\s*\d+|\d+\s*(?:participants?|users?|respondents?|samples?))\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_RANK_RE = re.compile(r"\b(?:top\s+\d+|#\d+|rank(?:ed|ing)?\s+\d+)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_NUMERIC_WORD_RE = re.compile(
    r"\b(?:majority|most|significant|double|tripled?|half|quarter|ratio|incidence|prevalence)\b",
    re.IGNORECASE,
)
_SOURCE_URL_RE = re.compile(r"https?://[^\s)\]]+", re.IGNORECASE)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _claim_tokens(line: str) -> list[str]:
    text = str(line or "")
    tokens: list[str] = []
    for pattern in (_PERCENT_RE, _RATIO_RE, _COUNT_RE, _YEAR_RE, _RANK_RE):
        tokens.extend([m.group(0).strip().lower() for m in pattern.finditer(text)])
    if not tokens:
        tokens.extend([m.group(0).strip().lower() for m in _NUMBER_RE.finditer(text)])
    for m in _NUMERIC_WORD_RE.finditer(text):
        tokens.append(m.group(0).strip().lower())
    # Deduplicate while preserving order.
    out: list[str] = []
    for token in tokens:
        if token and token not in out:
            out.append(token)
    return out


def _has_numeric_claim(line: str) -> bool:
    if _claim_tokens(line):
        return True
    # Also treat generic number tokens as numeric claims when paired with [E].
    if "[E]" in str(line or "") and re.search(r"\b\d+(?:\.\d+)?\b", str(line)):
        return True
    return False


def _collect_corpus(source_evidence: list[dict[str, str]] | None, findings: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in source_evidence or []:
        if not isinstance(row, dict):
            continue
        snippet = str(row.get("snippet", "")).strip()
        title = str(row.get("title", "")).strip()
        url = str(row.get("url", "")).strip()
        if title:
            parts.append(title)
        if snippet:
            parts.append(snippet)
        if url:
            parts.append(url)
    for item in findings:
        if not isinstance(item, dict):
            continue
        for row in item.get("source_evidence", []) if isinstance(item.get("source_evidence", []), list) else []:
            if not isinstance(row, dict):
                continue
            snippet = str(row.get("snippet", "")).strip()
            if snippet:
                parts.append(snippet)
    return _normalize_text("\n".join(parts))


def _token_supported(token: str, corpus_low: str) -> bool:
    t = _normalize_text(token)
    if not t:
        return False
    if t in corpus_low:
        return True
    # Flexible forms for percentages and multipliers.
    if t.endswith("%"):
        base = t[:-1].strip()
        if base and (f"{base} percent" in corpus_low or f"{base}%" in corpus_low):
            return True
    if t.endswith("x") or t.endswith("\u00d7"):
        base = t[:-1].strip()
        if base and (f"{base}x" in corpus_low or f"{base} times" in corpus_low):
            return True
    return False


def _llm_borderline_check(
    *,
    claim_line: str,
    evidence_excerpt: str,
    client: Any | None,
    repo_root: Path | None,
) -> bool:
    if client is None or repo_root is None:
        return False
    cfg = lane_model_config(repo_root, "chat_routing_gate") or {}
    model = str(cfg.get("model", "qwen3:4b")).strip() or "qwen3:4b"
    fallback = cfg.get("fallback_models", ["gemma3:4b"]) if isinstance(cfg.get("fallback_models", []), list) else ["gemma3:4b"]
    try:
        raw = client.chat(
            model=model,
            fallback_models=fallback,
            system_prompt=(
                "Decide if the numeric claim is explicitly supported by the evidence excerpt. "
                "Reply with JSON only: {\"supported\": true|false}."
            ),
            user_prompt=(
                f"Claim line:\n{claim_line[:600]}\n\n"
                f"Evidence excerpt:\n{evidence_excerpt[:2200]}\n"
            ),
            temperature=0.0,
            num_ctx=2048,
            think=False,
            timeout=20,
            retry_attempts=1,
            retry_backoff_sec=0.5,
            task_class="numeric_validator",
            tier="default",
        )
    except Exception:
        return False

    body = str(raw or "").strip()
    if not body:
        return False
    try:
        parsed = json.loads(body)
        return bool(parsed.get("supported", False))
    except Exception:
        return False


def validate(
    findings: list[dict[str, Any]],
    *,
    source_evidence: list[dict[str, str]] | None = None,
    client: Any | None = None,
    repo_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate numeric claims in findings against available source corpus.

    Unsupported numeric claims are downgraded from [E] to [S] and marked
    as unverified statistic.
    """
    corpus_low = _collect_corpus(source_evidence, findings)
    validated: list[dict[str, Any]] = []
    claims_seen = 0
    claims_flagged = 0

    for item in findings:
        row = dict(item)
        text = str(row.get("finding", ""))
        if not text.strip():
            validated.append(row)
            continue

        new_lines: list[str] = []
        for raw_line in text.splitlines():
            line = str(raw_line)
            if not _has_numeric_claim(line):
                new_lines.append(line)
                continue

            claims_seen += 1
            tokens = _claim_tokens(line)
            supported = bool(tokens) and all(_token_supported(token, corpus_low) for token in tokens)

            if not supported and tokens:
                # Borderline rescue only when a source URL is present nearby.
                urls = _SOURCE_URL_RE.findall(line)
                if urls:
                    excerpt = "\n".join([line, *urls])
                    supported = _llm_borderline_check(
                        claim_line=line,
                        evidence_excerpt=excerpt,
                        client=client,
                        repo_root=repo_root,
                    )

            if supported:
                new_lines.append(line)
                continue

            claims_flagged += 1
            revised = line
            if "[E]" in revised:
                revised = revised.replace("[E]", "[S]", 1)
            if "unverified statistic" not in revised.lower():
                revised = f"{revised} (unverified statistic)"
            new_lines.append(revised)

        row["finding"] = "\n".join(new_lines)
        validated.append(row)

    report = {
        "numeric_claims_seen": int(claims_seen),
        "numeric_claims_flagged": int(claims_flagged),
    }
    return validated, report


__all__ = ["validate"]
