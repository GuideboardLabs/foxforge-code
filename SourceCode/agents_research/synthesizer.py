from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from shared_tools.activity_bus import telemetry_emit

LOGGER = logging.getLogger(__name__)


class SynthesisUnavailableError(RuntimeError):
    """Raised when the synthesis model is unavailable or produces no output."""


def _last_wait_error(client: Any) -> str:
    err = str(getattr(client, "last_wait_error", "") or "").strip()
    if err:
        return err
    inner = getattr(client, "_ollama", None)
    if inner is not None:
        return str(getattr(inner, "last_wait_error", "") or "").strip()
    return ""


def extract_action_proposals(synthesis_text: str) -> list[dict[str, str]]:
    """
    Parse the 'Actionable Next Steps' section of a synthesis and return
    up to 5 create_task proposals. Pure text parsing — no LLM call.
    Each result: {"action_type": "create_task", "title": str, "notes": str}
    """
    body = str(synthesis_text or "").strip()
    if not body:
        return []

    # Find the Actionable Next Steps section
    match = re.search(
        r"(?:##\s*Actionable Next Steps|##\s*Next Steps)(.*?)(?=\n##|\Z)",
        body,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []

    section = match.group(1).strip()
    proposals: list[dict[str, str]] = []

    for line in section.splitlines():
        stripped = line.strip()
        # Match bullet points: - item or * item or 1. item
        m = re.match(r"^[-*•]\s+(.+)$|^\d+[.)]\s+(.+)$", stripped)
        if not m:
            continue
        text = (m.group(1) or m.group(2) or "").strip()
        # Strip inline evidence labels and markdown bold/italic
        text = re.sub(r"\[E\]|\[I\]|\[S\]", "", text).strip()
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text).strip()
        if len(text) < 8:
            continue
        title = text[:120].rstrip(".,;:")
        proposals.append({
            "action_type": "create_task",
            "title": title,
            "notes": f"Extracted from research synthesis actionable steps.",
        })
        if len(proposals) >= 5:
            break

    return proposals


def _sanitize_markdown_urls(text: str) -> str:
    """Strip malformed markdown links; keep the label text."""
    def _check_url(m: re.Match) -> str:
        label = m.group(1)
        url = m.group(2)
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return label
            netloc = parsed.netloc or ""
            if not netloc or "," in netloc:
                LOGGER.warning("Stripped malformed URL from synthesis: %s", url)
                return label
        except Exception:
            return label
        return m.group(0)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _check_url, text)


_INLINE_MD_URL_RE = re.compile(r"\[[^\]]+\]\((https?://[^)\s]+)\)", re.IGNORECASE)
_SOURCE_MARKER_URL_RE = re.compile(r"\[source:\s*(https?://[^\]\s)]+)\s*\]", re.IGNORECASE)
_RAW_URL_RE = re.compile(r"(https?://[^\s)\]]+)", re.IGNORECASE)


def _clean_url(url: str) -> str:
    candidate = str(url or "").strip().rstrip(".,;:!?)]")
    try:
        parsed = urlparse(candidate)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            return candidate
    except Exception:
        return ""
    return ""


def _extract_source_urls(findings: list[dict] | None, limit: int = 6) -> list[str]:
    if not findings:
        return []
    urls: list[str] = []
    seen: set[str] = set()
    for item in findings:
        text = str((item or {}).get("finding", "")).strip()
        if not text:
            continue
        for pattern in (_SOURCE_MARKER_URL_RE, _INLINE_MD_URL_RE, _RAW_URL_RE):
            for match in pattern.findall(text):
                raw = match[0] if isinstance(match, tuple) else match
                clean = _clean_url(raw)
                if not clean or clean in seen:
                    continue
                seen.add(clean)
                urls.append(clean)
                if len(urls) >= max(1, int(limit)):
                    return urls
    return urls


def _ensure_inline_source_links(text: str, findings: list[dict] | None) -> str:
    body = str(text or "").strip()
    if not body:
        return body
    if _INLINE_MD_URL_RE.search(body):
        return body
    urls = _extract_source_urls(findings, limit=6)
    if not urls:
        return body
    anchors = ["## Source Anchors", *[f"- [S{i + 1}]({url})" for i, url in enumerate(urls)]]
    return f"{body}\n\n" + "\n".join(anchors)


def _is_valid_synthesis(text: str) -> bool:
    body = str(text or "").strip()
    if len(body) < 380:
        return False
    low = body.lower()
    if any(token in low for token in ("model call failed", "ollama chat failed", "traceback")):
        return False
    expected_sections = [
        "executive summary",
        "key findings",
        "insights",
        "failure modes",
        "differentiation",
        "next steps",
    ]
    hits = 0
    for section in expected_sections:
        if section in low:
            hits += 1
    return hits >= 3


def _looks_truncated_output(text: str) -> bool:
    body = str(text or "").rstrip()
    if not body:
        return False
    tail = body[-220:].strip()
    if not tail:
        return False
    # Heuristic: ending on a letter with no terminal punctuation usually means clipping.
    if re.search(r"[A-Za-z]$", tail) and not re.search(r"[.!?][\"')\]]?\s*$", tail):
        return True
    return False


def _prior_open_questions_block(prior_open_questions: list[str] | None) -> str:
    rows = [str(x or "").strip() for x in (prior_open_questions or []) if str(x or "").strip()]
    if not rows:
        return ""
    bullets = "\n".join(f"- {row}" for row in rows[:8])
    return (
        "Prior runs on this topic left these open questions unresolved:\n"
        f"{bullets}\n"
        "For any of these you now have evidence on, answer it in the main summary "
        "(with [E]/[I] support). Do not re-list them unchanged. If one is still "
        "unresolved, state what new evidence would resolve it — do not repeat the "
        "question verbatim.\n\n"
    )


def _confidence_rank(label: str) -> int:
    low = str(label or "").strip().lower()
    if low in {"high"}:
        return 3
    if low in {"mixed", "moderate", "med", "medium"}:
        return 2
    if low in {"low"}:
        return 1
    return 0


def _cap_confidence_line(summary: str, cap_label: str) -> str:
    body = str(summary or "").strip()
    cap = "Low" if str(cap_label or "").strip().lower() == "low" else "Mixed"
    pattern = re.compile(r"(Evidence Confidence:\s*)(High|Mixed|Moderate|Low)\b", re.IGNORECASE)
    match = pattern.search(body)
    if not match:
        return body
    current = str(match.group(2) or "").strip()
    if _confidence_rank(current) <= _confidence_rank(cap):
        return body
    return pattern.sub(rf"\1{cap}", body, count=1)


def _inject_source_quality_warning(summary: str, warning_line: str) -> str:
    body = str(summary or "").strip()
    warning = str(warning_line or "").strip()
    if not body or not warning:
        return body
    if "## Source Quality Warning" in body:
        return body
    block = f"## Source Quality Warning\n- {warning}\n"
    anchor = re.search(r"^##\s*Executive Summary\b", body, flags=re.IGNORECASE | re.MULTILINE)
    if not anchor:
        return f"{block}\n{body}".strip()
    return f"{body[:anchor.start()].rstrip()}\n\n{block}\n{body[anchor.start():].lstrip()}".strip()


def synthesize(
    question: str,
    findings: list[dict],
    *,
    client: Any | None = None,
    model_cfg: dict | None = None,
    prior_messages: list[dict[str, str]] | None = None,
    conflict_report: str = "",
    prior_synthesis: str = "",
    prior_open_questions: list[str] | None = None,
    source_tier_map: dict[str, str] | None = None,
) -> str:
    if client is None or not model_cfg:
        raise SynthesisUnavailableError(
            "Synthesis model unavailable — no client or model config provided."
        )

    model = str(model_cfg.get("synthesis_model") or model_cfg.get("model", "")).strip()
    if not model:
        raise SynthesisUnavailableError(
            "Synthesis model unavailable — model name is empty in config."
        )

    def _conf_label(item: dict) -> str:
        score = item.get("confidence", 0)
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 0
        if score >= 4:
            return "HIGH"
        if score >= 2:
            return "MED"
        return "LOW"

    tiers = {"tier1", "tier2", "tier3", "tier4"}
    normalized_tier_map = {
        str(k or "").strip().rstrip("/,."): str(v or "").strip().lower()
        for k, v in (source_tier_map or {}).items()
        if str(k or "").strip()
    }

    def _tier_breakdown_for_finding(text: str) -> dict[str, int]:
        counts = {"tier1": 0, "tier2": 0, "tier3": 0, "tier4": 0}
        for raw in _SOURCE_MARKER_URL_RE.findall(str(text or "")):
            url = _clean_url(raw) or str(raw or "").strip().rstrip("/,.")
            if not url:
                continue
            tier = (
                normalized_tier_map.get(url)
                or normalized_tier_map.get(url.rstrip("/,."))
                or "tier3"
            )
            if tier in tiers:
                counts[tier] += 1
        return counts

    def _finding_label(item: dict) -> str:
        label_parts = [f"{item['agent']} | confidence:{_conf_label(item)}"]
        raw_counts = item.get("source_tier_counts")
        if isinstance(raw_counts, dict):
            tier_counts = {
                "tier1": int(raw_counts.get("tier1", 0) or 0),
                "tier2": int(raw_counts.get("tier2", 0) or 0),
                "tier3": int(raw_counts.get("tier3", 0) or 0),
                "tier4": int(raw_counts.get("tier4", 0) or 0),
            }
        else:
            tier_counts = _tier_breakdown_for_finding(str(item.get("finding", "")))
        breakdown = ", ".join(f"{count}×{tier}" for tier, count in tier_counts.items() if count)
        if breakdown:
            label_parts.append(f"sources: {breakdown}")
        return "[" + " | ".join(label_parts) + "]"

    primary = [item for item in findings if str(item.get("role", "primary")).strip().lower() != "advisory"]
    advisory = [item for item in findings if str(item.get("role", "primary")).strip().lower() == "advisory"]
    primary_blob = "\n\n".join([
        f"{_finding_label(item)}\n{item['finding']}"
        for item in primary
    ])
    advisory_blob = "\n\n".join([
        f"{_finding_label(item)}\n{item['finding']}"
        for item in advisory
    ])
    findings_blob = primary_blob
    if advisory_blob:
        findings_blob = (
            f"{primary_blob}\n\n"
            "---\n"
            "ADVISORY CONTEXT (supplementary — do not treat as equal-weight primary research; "
            "use only to add caveats, flag compliance notes, or note statistical uncertainty):\n\n"
            f"{advisory_blob}"
        )

    tier_totals = {"tier1": 0, "tier2": 0, "tier3": 0, "tier4": 0}
    for item in findings:
        raw_counts = item.get("source_tier_counts")
        if isinstance(raw_counts, dict):
            counts = {
                "tier1": int(raw_counts.get("tier1", 0) or 0),
                "tier2": int(raw_counts.get("tier2", 0) or 0),
                "tier3": int(raw_counts.get("tier3", 0) or 0),
                "tier4": int(raw_counts.get("tier4", 0) or 0),
            }
        else:
            counts = _tier_breakdown_for_finding(str(item.get("finding", "")))
        for key in tier_totals:
            tier_totals[key] += int(counts.get(key, 0) or 0)
    total_citations = sum(tier_totals.values())
    tier_cap_label = ""
    source_quality_warning = ""
    if total_citations > 0:
        tier12 = int(tier_totals.get("tier1", 0) or 0) + int(tier_totals.get("tier2", 0) or 0)
        tier3 = int(tier_totals.get("tier3", 0) or 0)
        tier4 = int(tier_totals.get("tier4", 0) or 0)
        if tier4 > 0 and (tier4 / max(1, total_citations)) >= 0.6:
            tier_cap_label = "low"
            source_quality_warning = (
                "Most cited support is cross-domain analogy (tier4). "
                "Conclusions are exploratory — no official docs, registries, or technical sources confirmed these claims."
            )
        elif tier12 == 0 and (tier3 + tier4) > 0:
            tier_cap_label = "mixed"
            source_quality_warning = (
                "All cited support is community/blog/forum level (tier3/tier4). "
                "Confidence is capped — no official documentation or established technical sources corroborate these findings."
            )
        elif tier3 > 0 and (tier3 / max(1, total_citations)) > 0.55 and (tier12 / max(1, total_citations)) < 0.30:
            tier_cap_label = "mixed"
            source_quality_warning = (
                "Most cited sources are blog or community level (tier3); primary sources are present but not dominant. "
                "Confidence is capped at Mixed — treat conclusions as informed best-practice rather than validated findings."
            )

    def _postprocess_output(text: str) -> str:
        out = _sanitize_markdown_urls(_ensure_inline_source_links(text, findings))
        if tier_cap_label:
            out = _cap_confidence_line(out, tier_cap_label)
        if source_quality_warning:
            out = _inject_source_quality_warning(out, source_quality_warning)
        return out

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_prompt = (
        f"Today's date: {today_str}. "
        "You are a research synthesizer for an orchestrator. "
        "Produce concise, high-signal markdown with sections: Executive Summary, "
        "Key Findings, Insights & Design Implications, Failure Modes & Risks, "
        "Differentiation Opportunity, Next Steps. Avoid fluff. "
        "Do not wrap the response in triple backticks or fenced code blocks.\n\n"
        "SYNTHESIS DISCIPLINE: Do NOT summarize each agent sequentially "
        "('Agent X found... Agent Y found...'). Extract the highest-signal claims "
        "across ALL agents and write a unified narrative. The reader should not be "
        "able to tell which claim came from which agent.\n\n"
        "FINDING CLASSIFICATION: Tag each key finding with one of:\n"
        "  [baseline] — well-known best practice; compress to 1 sentence, do not elaborate.\n"
        "  [insight] — non-obvious, useful differentiation; give full treatment.\n"
        "  [risk] — specific failure mode or when-it-breaks scenario; give full treatment.\n"
        "  [gap] — no primary evidence found; state the gap explicitly.\n"
        "Prioritize [insight] and [risk] items. Group and compress all [baseline] items.\n\n"
        "CLAIM CONVERGENCE: For each major finding, count how many distinct agents "
        "independently supported it and state that count explicitly. Use phrasing like "
        "'All N agents converged on X', 'N of M agents supported Y', or "
        "'Only one agent raised Z (single-source signal — treat with caution)'. "
        "Compress repeated points into one sentence with convergence count and avoid "
        "duplicate bullets that restate the same claim.\n"
        "CONVERGENCE CAVEAT: If all agents cited the same source type (e.g., all vendor PM "
        "blogs, all 'best practices' tutorial sites), their agreement is a source-type echo, "
        "not independent empirical validation. Flag this explicitly: 'All N agents cited "
        "[source type] — this reflects consensus framing, not independent validation.'\n\n"
        "CONFIDENCE WEIGHTING: Each agent finding is labelled with confidence:HIGH/MED/LOW "
        "(self-assessed by the agent on a 1-5 scale). Weight your conclusions toward HIGH-confidence "
        "findings. If your summary relies heavily on MED or LOW findings, explicitly flag this "
        "in the Evidence Confidence line.\n\n"
        "SOURCE TIER WEIGHTING: Agent findings include source tier markers in the label. "
        "Tier1 = official primary sources (language docs, package registries, standards bodies, "
        "security advisories, platform technical documentation). "
        "Tier2 = strong secondary sources (maintained repos, Stack Overflow, engineering blogs, "
        "reputable technical publishers, official vendor community forums). "
        "NOTE: A well-known company's blog or marketing content (e.g., atlassian.com/blog, "
        "hubstaff.com/blog, notion.so/blog) is tier2, NOT tier1, regardless of how famous the "
        "company is. Tier1 requires official technical specifications or standards — not editorial "
        "or product-marketing content. "
        "Tier3 = usable but variable sources (community discussion, tutorial sites, dev blogs, "
        "Medium posts, Reddit threads). "
        "Tier4 = cross-domain analogy sources — inspiration only, cannot justify [E] claims. "
        "Treat tier1 claims as authoritative; tier2 as reliable with normal caveats; "
        "tier3 as context only — hedge with 'community experience suggests' rather than 'research shows'; "
        "tier4 must never appear as evidence.\n\n"
        "INSIGHT LAYER (REQUIRED): After synthesizing the consensus view, you MUST surface what "
        "the dominant framing misses. Ask: 'What works in textbook conditions but fails in practice? "
        "What behavioral or design edge cases does the consensus ignore? What would someone with "
        "10 years of hands-on experience add that is not in the standard sources?' "
        "The Insights & Design Implications section MUST contain at least 2 [insight] items that "
        "are non-obvious — not restatements of consensus best practice. "
        "The Failure Modes & Risks section MUST contain at least 2 specific failure scenarios — "
        "not generic 'evidence is thin' caveats but concrete 'this breaks when X happens' patterns. "
        "The Differentiation Opportunity section MUST contain at least 1 concrete way to do this "
        "better than the standard approach found in sources. "
        "If the source evidence does not surface these, name the gap explicitly — do not invent them.\n\n"
        "UNVERIFIED STATISTICS: Some agent findings are tagged '(unverified statistic)' by the numeric "
        "validator because they could not be grounded in the source corpus. Do NOT present these numbers "
        "as concrete thresholds in the summary — strip the numeric value and convert to qualitative phrasing "
        "(e.g., write 'high reliability' not '90% success rate'). Never lift unverified numbers into Key "
        "Findings or Insights as if they were established benchmarks.\n\n"
        "EVIDENCE DISCIPLINE: Agent findings include [E]/[I]/[S] labels.\n"
        "- [E]: state confidently, include an inline markdown URL citation like [source](https://...).\n"
        "- [I]: frame as inference — 'this suggests...'\n"
        "- [S]: frame as hypothesis — 'one possibility is...'\n"
        "Never launder [I] or [S] into presented facts.\n"
        "If any [E] claim appears in your output, include at least one inline markdown URL link in the same sentence "
        "or bullet whenever possible.\n"
        "When a sentence is source-grounded, append an inline source marker like [S1] or [S2]. "
        "For inference-only sentences, append [I].\n\n"
        "NO NEW CLAIMS: Only assert facts, statistics, names, dates, or conclusions that appear "
        "in the agent findings above. Do NOT introduce details that are not traceable to at least "
        "one finding — not even plausible-sounding ones. If coverage is thin, state that explicitly "
        "in Failure Modes & Risks rather than filling the gap. Fabrication is worse than a short answer.\n\n"
        "RESEARCH-ONLY: Your training knowledge about specific products, services, apps, statistics, "
        "or recent events is unreliable and may be factually wrong. Synthesize EXCLUSIVELY from the "
        "agent findings provided. Do not supplement with background knowledge — even when the findings "
        "seem thin or incomplete.\n\n"
        "COVERAGE GAPS: When a topic area in the question has no [E] findings with cited source URLs "
        "from agents, do NOT fill the gap with general knowledge or inference. Instead write: "
        "'Coverage gap: no primary evidence found for [area].' "
        "A gap declaration is better than a gap filled with unverified claims.\n\n"
        "SOURCE INTEGRITY: Cite only external sources from this run's agent findings. "
        "Prior internal artifacts (research raws, summaries, critiques, project notes) are not sources.\n\n"
        "FINAL SUMMARY CONTRACT: Include explicit sections for Executive Summary, "
        "Key Findings (with [baseline]/[insight]/[risk]/[gap] tags), "
        "Insights & Design Implications (REQUIRED: ≥2 non-obvious [insight] items), "
        "Failure Modes & Risks (REQUIRED: ≥2 specific failure scenarios), "
        "Differentiation Opportunity (REQUIRED: ≥1 concrete differentiator), "
        "Next Steps. Ensure the narrative is project-specific rather than generic.\n\n"
        "End Executive Summary with: 'Evidence Confidence: [High/Mixed/Low] — [one-line reason].' "
        "For time-sensitive topics, state whether events are upcoming, ongoing, or past relative to today."
    )
    _conflict_section = ""
    if conflict_report and conflict_report.strip():
        _conflict_section = (
            f"\n\nCROSS-AGENT DISPUTES — reconcile these explicitly in your synthesis "
            f"(state which position has stronger evidence or note genuine uncertainty):\n"
            f"{conflict_report.strip()}"
        )
    _prior_block = ""
    if prior_synthesis and prior_synthesis.strip():
        _prior_block = (
            f"Prior synthesis (for reference — refine, don't repeat):\n"
            f"{prior_synthesis.strip()[:1200]}\n\n"
            "New and supplementary findings below — use these to fill gaps the prior synthesis left open:\n"
        )
    prior_open_block = _prior_open_questions_block(prior_open_questions)
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"{prior_open_block}"
        f"{_prior_block}"
        f"Research outputs:\n{findings_blob}"
        f"{_conflict_section}\n\n"
        "Return markdown only, not inside ``` fences."
    )
    validation_cycles = max(1, int(model_cfg.get("synthesis_validation_cycles", 3)))
    retry_attempts = max(1, int(model_cfg.get("synthesis_retry_attempts", 6)))
    retry_backoff_sec = max(0.0, float(model_cfg.get("synthesis_retry_backoff_sec", 1.5)))
    timeout = int(model_cfg.get("synthesis_timeout_sec", model_cfg.get("timeout_sec", 0)))
    fallback_models_raw = model_cfg.get("synthesis_fallback_models", [])
    fallback_models: list[str] = []
    if isinstance(fallback_models_raw, list):
        for item in fallback_models_raw:
            name = str(item or "").strip()
            if name:
                fallback_models.append(name)

    if hasattr(client, "wait_for_available"):
        try:
            available = bool(
                client.wait_for_available(
                    model,
                    fallback_models=fallback_models,
                    max_wait_sec=300,
                    poll_interval_sec=15,
                )
            )
        except TypeError:
            # Backward compatibility with clients that don't accept fallback_models.
            available = bool(client.wait_for_available(model, max_wait_sec=300, poll_interval_sec=15))
        if not available:
            candidates = [model, *fallback_models]
            models_note = ", ".join([f"'{m}'" for m in candidates if str(m).strip()])
            reason = _last_wait_error(client)
            detail = f" Last error: {reason}." if reason else ""
            raise SynthesisUnavailableError(
                "No synthesis model became available within 5 minutes "
                f"(candidates: {models_note}).{detail} Research run aborted — no output written."
            )

    last_text = ""
    generation_errors: list[str] = []
    for cycle in range(validation_cycles):
        prompt = user_prompt
        if cycle > 0:
            prompt = (
                f"{user_prompt}\n\n"
                "Regenerate with stricter quality control. Ensure all required sections appear with clear headers."
            )
        try:
            candidate = client.chat(
                model=model,
                fallback_models=fallback_models,
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=float(model_cfg.get("temperature", 0.2)),
                num_ctx=int(model_cfg.get("num_ctx", 16384)),
                num_predict=int(model_cfg.get("synthesis_num_predict", 4096)),
                think=bool(model_cfg.get("think", False)),
                timeout=timeout,
                retry_attempts=retry_attempts,
                retry_backoff_sec=retry_backoff_sec,
            )
            last_text = candidate
            if _is_valid_synthesis(candidate):
                if _looks_truncated_output(candidate):
                    LOGGER.warning("Synthesis output appears truncated; retrying once with truncation guard.")
                    continue
                return _postprocess_output(candidate)
        except Exception as exc:
            err = f"cycle {cycle + 1}/{validation_cycles}: {type(exc).__name__}: {str(exc).strip()[:320]}"
            generation_errors.append(err)
            LOGGER.warning("Synthesis generation failed (%s)", err)
            continue

    if _is_valid_synthesis(last_text) and _looks_truncated_output(last_text):
        try:
            recovered = client.chat(
                model=model,
                fallback_models=fallback_models,
                system_prompt=system_prompt,
                user_prompt=(
                    f"{user_prompt}\n\n"
                    "Previous draft appeared truncated mid-word. Regenerate the full summary and end with complete "
                    "sentences and terminal punctuation."
                ),
                temperature=float(model_cfg.get("temperature", 0.2)),
                num_ctx=int(model_cfg.get("num_ctx", 16384)),
                num_predict=int(model_cfg.get("synthesis_num_predict", 4096)),
                think=bool(model_cfg.get("think", False)),
                timeout=timeout,
                retry_attempts=retry_attempts,
                retry_backoff_sec=retry_backoff_sec,
            )
            if _is_valid_synthesis(recovered):
                return _postprocess_output(recovered)
        except Exception as exc:
            err = f"truncation_recovery: {type(exc).__name__}: {str(exc).strip()[:320]}"
            generation_errors.append(err)
            LOGGER.warning("Synthesis truncation recovery failed (%s)", err)
            pass

    if _is_valid_synthesis(last_text):
        return _postprocess_output(last_text)
    if last_text.strip():
        return _postprocess_output(
            f"{last_text}\n\n"
            "_Reliability note: synthesis did not pass full section validation after retries; "
            "review before treating as final._"
        )
    reason = generation_errors[-1] if generation_errors else "unknown error"
    raise SynthesisUnavailableError(
        f"Synthesis model '{model}' produced no output after all retries. "
        f"Last error: {reason}. Research run aborted — no output written."
    )


_SEVERITY_ISSUE_KEYS = (
    "fabricated_specifics",
    "unsupported_claims",
    "contradictions",
    "weak_evidence_caveats_missing",
    "missing_perspectives",
    "authority_misattribution",
)


def _default_severity_payload() -> dict[str, Any]:
    return {
        "severity": 2,
        "issues": {key: 0 for key in _SEVERITY_ISSUE_KEYS},
        "conclusion_vulnerability": "medium",
        "recommended_action": "revise_default",
        "revise_focus": [],
    }


def _sanitize_severity_payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return _default_severity_payload()
    out = _default_severity_payload()
    try:
        sev = int(raw.get("severity", out["severity"]))
    except (TypeError, ValueError):
        sev = out["severity"]
    out["severity"] = max(0, min(5, sev))

    issues_in = raw.get("issues")
    issues_out: dict[str, int] = {}
    for key in _SEVERITY_ISSUE_KEYS:
        val = 0
        if isinstance(issues_in, dict):
            try:
                val = int(issues_in.get(key, 0) or 0)
            except (TypeError, ValueError):
                val = 0
        issues_out[key] = max(0, val)
    out["issues"] = issues_out

    vulnerability = str(raw.get("conclusion_vulnerability", out["conclusion_vulnerability"])).strip().lower()
    if vulnerability not in {"low", "medium", "high"}:
        vulnerability = "medium"
    out["conclusion_vulnerability"] = vulnerability

    action = str(raw.get("recommended_action", out["recommended_action"])).strip().lower()
    if action not in {"accept", "revise_default", "escalate_premium", "reject"}:
        action = "revise_default"
    out["recommended_action"] = action

    revise_focus = raw.get("revise_focus")
    if isinstance(revise_focus, list):
        out["revise_focus"] = [
            str(item).strip()[:240]
            for item in revise_focus
            if str(item).strip()
        ][:10]
    else:
        out["revise_focus"] = []
    return out


def _emit_critic_severity(
    payload: dict[str, Any],
    *,
    parse_ok: bool,
    fallback_used: bool,
    model: str,
) -> None:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        telemetry_emit(
            repo_root,
            "critic_severity.jsonl",
            {
                "severity": int(payload.get("severity", 2) or 2),
                "issues": dict(payload.get("issues", {})) if isinstance(payload.get("issues", {}), dict) else {},
                "conclusion_vulnerability": str(payload.get("conclusion_vulnerability", "medium")).strip().lower(),
                "recommended_action": str(payload.get("recommended_action", "revise_default")).strip().lower(),
                "revise_focus_count": len(payload.get("revise_focus", [])) if isinstance(payload.get("revise_focus", []), list) else 0,
                "parse_ok": bool(parse_ok),
                "fallback_used": bool(fallback_used),
                "model": str(model or "").strip(),
            },
            retention_days=30,
        )
    except Exception:
        pass


def _severity_from_response(body: str, *, model: str) -> dict[str, Any]:
    default_payload = _default_severity_payload()
    text = str(body or "").strip()
    severity_match = re.search(r"<SEVERITY>\s*([\s\S]*?)\s*</SEVERITY>", text, re.IGNORECASE)
    if not severity_match:
        _emit_critic_severity(default_payload, parse_ok=False, fallback_used=True, model=model)
        return default_payload
    raw_json = str(severity_match.group(1) or "").strip()
    if not raw_json:
        _emit_critic_severity(default_payload, parse_ok=False, fallback_used=True, model=model)
        return default_payload
    try:
        parsed = json.loads(raw_json)
    except Exception:
        _emit_critic_severity(default_payload, parse_ok=False, fallback_used=True, model=model)
        return default_payload
    cleaned = _sanitize_severity_payload(parsed)
    _emit_critic_severity(cleaned, parse_ok=True, fallback_used=False, model=model)
    return cleaned


def run_skeptic_pass_with_severity(
    question: str,
    synthesis: str,
    *,
    client: Any | None = None,
    model_cfg: dict | None = None,
    findings: list[dict] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """
    Adversarial second pass that returns:
      (revised_summary, critique_log, severity_payload)
    """
    base_summary = str(synthesis or "").strip()
    if client is None or not model_cfg or not base_summary:
        return base_summary, "", _default_severity_payload()
    model = str(model_cfg.get("model", "")).strip()
    if not model:
        return base_summary, "", _default_severity_payload()

    fallback_models_raw = model_cfg.get("synthesis_fallback_models", [])
    fallback_models: list[str] = (
        [str(m) for m in fallback_models_raw if str(m or "").strip()]
        if isinstance(fallback_models_raw, list)
        else []
    )

    _findings_ref = ""
    if findings:
        ref_parts: list[str] = []
        for item in findings:
            agent = str(item.get("agent", "agent")).strip()
            text = str(item.get("finding", "")).strip()[:2000]
            if text:
                ref_parts.append(f"[{agent}]: {text}")
        if ref_parts:
            _findings_ref = "\n\n".join(ref_parts)

    system_prompt = (
        "You are the Skeptic Engine — an internal adversary whose only job is to stress-test "
        "research conclusions before they reach the user. You are not trying to be balanced or "
        "reassuring. You are trying to find every crack.\n\n"
        "You must output EXACTLY THREE XML-tagged blocks and nothing else:\n"
        "<REVISED_SUMMARY>...</REVISED_SUMMARY>\n"
        "<CRITIQUE_LOG>...</CRITIQUE_LOG>\n"
        "<SEVERITY>{...strict JSON...}</SEVERITY>\n\n"
        "REVISED_SUMMARY must be markdown safe for publication.\n"
        "- Remove fabricated specifics (numbers, dates, names, URLs, version strings, direct quotes) "
        "that are not in raw findings.\n"
        "- Downgrade unsupported certainty to [S] with hedging language.\n"
        "- An [E] claim IS sourced if the agent's raw finding (in the reference section below) "
        "contains a `[source: <URL>]` marker OR an inline `(https://...)` link whose host or title "
        "aligns with the claim. Do NOT demote [E] just because the synthesis sentence itself dropped "
        "the link — verify against the raw findings before demoting.\n"
        "- Only demote [E] -> [I] when NO supporting URL exists anywhere in raw findings.\n"
        "- Preserve valid inline markdown links for supported [E] claims whenever possible.\n"
        "- Keep required sections and preserve readability.\n"
        "- Validate final contract sections: Executive Summary, Evidence/Inference/Speculation buckets, "
        "Project Implications, Recommended Actions, Risks, Source Quality Notes, Rejected/Weak Sources. "
        "If any are missing, add them succinctly.\n\n"
        "- Fabricated-authority guardrail: if a claim leans on a famous authority/framework "
        "(for example Taylor, Miller's Law, Hawthorne effect, Fogg model) via a secondary blog/post "
        "and the cited source does not actually support the specific claim, demote [E] -> [I] or strike it.\n"
        "- Be extra strict when authority and claim domain differ (for example factory management cited for UX).\n\n"
        "CRITIQUE_LOG must briefly cover:\n"
        "  1. Fabricated specifics removed\n"
        "  2. Unsourced [E] labels corrected\n"
        "  3. Unsupported claims demoted/removed\n"
        "  4. Weak evidence caveats added\n"
        "  5. Missing perspectives flagged\n"
        "  6. Conclusion vulnerability summary\n"
        "  7. Confidence adjustment sentence\n\n"
        "SEVERITY must be strict JSON with this schema:\n"
        "{"
        "\"severity\": 0-5, "
        "\"issues\": {"
        "\"fabricated_specifics\": int, "
        "\"unsupported_claims\": int, "
        "\"contradictions\": int, "
        "\"weak_evidence_caveats_missing\": int, "
        "\"missing_perspectives\": int, "
        "\"authority_misattribution\": int"
        "}, "
        "\"conclusion_vulnerability\": \"low|medium|high\", "
        "\"recommended_action\": \"accept|revise_default|escalate_premium|reject\", "
        "\"revise_focus\": [\"specific bullet\", \"...\"]"
        "}\n"
        "Do not output any extra wrapper text."
    )
    _findings_section = (
        f"\n\nRaw findings reference (first 2000 chars per agent — use to cross-check [E] claims):\n{_findings_ref}"
        if _findings_ref else ""
    )
    user_prompt = (
        f"Research question: {question}\n\n"
        f"Synthesis to challenge:\n{base_summary}"
        f"{_findings_section}\n\n"
        "Return exactly the XML-tagged three-block format."
    )

    critique_fallback = ""
    severity_payload = _default_severity_payload()
    for _ in range(2):
        try:
            result = client.chat(
                model=model,
                fallback_models=fallback_models,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.6,
                num_ctx=int(model_cfg.get("num_ctx", 16384)),
                num_predict=int(model_cfg.get("skeptic_num_predict", 3072)),
                think=False,
                timeout=int(model_cfg.get("synthesis_timeout_sec", model_cfg.get("timeout_sec", 0))),
                retry_attempts=2,
                retry_backoff_sec=1.5,
                task_class="skeptic_pass",
                tier="default",
            )
            body = str(result or "").strip()
            if not body:
                continue
            severity_payload = _severity_from_response(body, model=model)
            revised_match = re.search(r"<REVISED_SUMMARY>\s*([\s\S]*?)\s*</REVISED_SUMMARY>", body, re.IGNORECASE)
            critique_match = re.search(r"<CRITIQUE_LOG>\s*([\s\S]*?)\s*</CRITIQUE_LOG>", body, re.IGNORECASE)
            if revised_match:
                revised_text = str(revised_match.group(1) or "").strip()
                critique_text = str(critique_match.group(1) or "").strip() if critique_match else ""
                if revised_text:
                    return (
                        _sanitize_markdown_urls(_ensure_inline_source_links(revised_text, findings)),
                        critique_text,
                        severity_payload,
                    )
            if "---CRITIQUE---" in body:
                revised, critique = body.split("---CRITIQUE---", 1)
                revised_text = revised.strip()
                critique_text = critique.strip()
                if revised_text:
                    return (
                        _sanitize_markdown_urls(_ensure_inline_source_links(revised_text, findings)),
                        critique_text,
                        severity_payload,
                    )
            critique_fallback = body
        except Exception:
            continue

    if critique_fallback:
        try:
            revised = client.chat(
                model=model,
                fallback_models=fallback_models,
                system_prompt=(
                    "You are a strict editor. Apply critique feedback directly to the supplied synthesis. "
                    "Return ONLY the revised synthesis markdown, with no commentary."
                ),
                user_prompt=(
                    f"Original synthesis:\n{base_summary}\n\n"
                    f"Critique feedback to apply:\n{critique_fallback}\n\n"
                    "Return the revised synthesis only."
                ),
                temperature=0.2,
                num_ctx=int(model_cfg.get("num_ctx", 16384)),
                num_predict=int(model_cfg.get("skeptic_num_predict", 3072)),
                think=False,
                timeout=int(model_cfg.get("synthesis_timeout_sec", model_cfg.get("timeout_sec", 0))),
                retry_attempts=1,
                retry_backoff_sec=1.0,
                task_class="skeptic_pass_fallback",
                tier="default",
            )
            revised_text = str(revised or "").strip()
            if revised_text:
                return (
                    _sanitize_markdown_urls(_ensure_inline_source_links(revised_text, findings)),
                    critique_fallback,
                    severity_payload,
                )
        except Exception:
            pass

    return (
        _sanitize_markdown_urls(_ensure_inline_source_links(base_summary, findings)),
        critique_fallback,
        severity_payload,
    )


def run_skeptic_pass(
    question: str,
    synthesis: str,
    *,
    client: Any | None = None,
    model_cfg: dict | None = None,
    findings: list[dict] | None = None,
) -> tuple[str, str]:
    revised, critique, _severity = run_skeptic_pass_with_severity(
        question,
        synthesis,
        client=client,
        model_cfg=model_cfg,
        findings=findings,
    )
    return revised, critique
