from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
import json
import logging
import re
import time
from urllib.parse import urlsplit
from typing import Any, Callable

from agents_research.citation_linker import build_retrieved_chunks
from agents_research.numeric_validator import validate as validate_numeric_claims
from agents_research.synthesizer import SynthesisUnavailableError, run_skeptic_pass, run_skeptic_pass_with_severity, synthesize
from agents_research.translation_chain import (
    enforce_final_summary_contract,
    ensure_project_specificity,
    run_translation_chain,
)
from shared_tools.answer_composer import evaluate_answer_confidence
from shared_tools.config_ini import load_config
from shared_tools.embedding_memory import _vec_cosine
from shared_tools.file_store import ProjectStore
from shared_tools.feedback_learning import FeedbackLearningEngine
from shared_tools.loop_controller import run_draft_critique_revise
from shared_tools.model_routing import lane_model_config, load_model_routing
from shared_tools.inference_router import InferenceRouter
from shared_tools.ollama_client import OllamaClient
from shared_tools.activity_bus import telemetry_emit

LOGGER = logging.getLogger(__name__)


_URL_PATTERN = re.compile(
    r"https?://\S+"
    r"|(?<!\w)(?:[a-zA-Z0-9-]+\.(?:com|org|gov|edu|io|net|co|uk|de|fr|ca|au))(?:/\S*)?",
    re.IGNORECASE,
)
_WEB_CONTEXT_SOURCE_RE = re.compile(
    r"^\-\s(?:\[(?P<tier>tier[1234])(?:\s+[^\]]*)?\]\s*)?.*\|\s+(?P<url>https?://\S+)\s*$",
    re.IGNORECASE,
)
_SELF_SCORE_RE = re.compile(
    r"^\s*#{0,6}\s*SELF[_\s]*SCORE\s*:?\s*"
    r"(?:confidence|conf)\s*=\s*(\d*\.?\d+)\s*[;,]\s*"
    r"coverage\s*=\s*(\d*\.?\d+)\s*[;,]\s*"
    r"notes\s*=\s*(.+?)[.;]?\s*$",
    re.IGNORECASE,
)
_OPEN_QUESTIONS_HEADING_RE = re.compile(
    r"^\s*##\s*(?:Key Risks\s*/\s*Open Questions|Open Questions|Uncertainties\s*&\s*Risks)\s*$",
    re.IGNORECASE,
)
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)$")
_SOURCE_MARKER_URL_RE = re.compile(r"\[source:\s*(https?://[^\s\]]+)\]", re.IGNORECASE)


def _domain_from_url(url: str) -> str:
    try:
        return str(urlsplit(str(url)).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _normalize_source_url(url: str) -> str:
    return str(url or "").strip().rstrip("/,.")


def _extract_web_source_evidence(web_context: str) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    if not str(web_context or "").strip():
        return evidence
    current: dict[str, str] | None = None
    for raw_line in str(web_context).splitlines():
        line = str(raw_line or "").rstrip()
        match = _WEB_CONTEXT_SOURCE_RE.match(line.strip())
        if match:
            if current and current.get("url"):
                evidence.append(current)
            url = _normalize_source_url(match.group("url"))
            tier = str(match.group("tier") or "").strip().lower()
            source_tier = tier if tier in {"tier1", "tier2", "tier3", "tier4"} else ""
            current = {
                "url": url,
                "domain": _domain_from_url(url),
                "snippet": "",
                "source_tier": source_tier,
            }
            continue
        if current is not None and line.strip().startswith("snippet:"):
            current["snippet"] = line.split("snippet:", 1)[1].strip()
    if current and current.get("url"):
        evidence.append(current)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in evidence:
        url = _normalize_source_url(row.get("url", ""))
        if not url or url in seen:
            continue
        row["url"] = url
        seen.add(url)
        out.append(row)
    return out


def _build_url_tier_map(source_evidence: list[dict[str, str]] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in source_evidence or []:
        if not isinstance(row, dict):
            continue
        url = _normalize_source_url(row.get("url", ""))
        tier = str(row.get("source_tier", "")).strip().lower()
        if not url or tier not in {"tier1", "tier2", "tier3", "tier4"}:
            continue
        out[url] = tier
    return out


def _tier_breakdown_for_finding(finding_text: str, url_tier_map: dict[str, str]) -> dict[str, int]:
    """Count tier distribution of URLs cited in a single agent finding."""
    urls = _SOURCE_MARKER_URL_RE.findall(str(finding_text or ""))
    counts = {"tier1": 0, "tier2": 0, "tier3": 0, "tier4": 0}
    for url in urls:
        normalized = _normalize_source_url(url)
        tier = (
            url_tier_map.get(normalized)
            or url_tier_map.get(normalized.rstrip("/,."))
            or "tier3"
        )
        if tier in counts:
            counts[tier] += 1
    return counts


def _audit_evidence_labels(findings: list[dict]) -> list[dict]:
    """Downgrade [E] labels unless they align to known source evidence."""
    emb_client: OllamaClient | None = None
    line_vec_cache: dict[str, list[float]] = {}
    snippet_vec_cache: dict[str, list[float]] = {}

    def _client() -> OllamaClient | None:
        nonlocal emb_client
        if emb_client is not None:
            return emb_client
        try:
            emb_client = OllamaClient()
            return emb_client
        except Exception:
            return None

    def _alignment(line_text: str, source_evidence: list[dict[str, str]]) -> float:
        snippets = [str(row.get("snippet", "")).strip() for row in source_evidence if str(row.get("snippet", "")).strip()]
        if not snippets:
            return 0.0
        client = _client()
        if client is None:
            # Fallback: token overlap when embedding path is unavailable.
            line_words = set(re.findall(r"[a-z0-9]{4,}", line_text.lower()))
            if not line_words:
                return 0.0
            best = 0.0
            for snippet in snippets:
                snippet_words = set(re.findall(r"[a-z0-9]{4,}", snippet.lower()))
                if not snippet_words:
                    continue
                overlap = len(line_words & snippet_words) / max(1, len(line_words))
                if overlap > best:
                    best = overlap
            return float(best)
        try:
            line_key = line_text[:600]
            if line_key not in line_vec_cache:
                line_vec_cache[line_key] = client.embed("qwen3-embedding:4b", line_key, timeout=20)
            line_vec = line_vec_cache[line_key]
            best = 0.0
            for snippet in snippets:
                snippet_key = snippet[:1200]
                if snippet_key not in snippet_vec_cache:
                    snippet_vec_cache[snippet_key] = client.embed("qwen3-embedding:4b", snippet_key, timeout=20)
                score = _vec_cosine(line_vec, snippet_vec_cache[snippet_key])
                if score > best:
                    best = score
            return float(best)
        except Exception:
            return 0.0

    result: list[dict] = []
    for item in findings:
        text = str(item.get("finding", ""))
        if "[E]" not in text and "[A]" not in text:
            result.append(item)
            continue
        source_evidence = [dict(x) for x in item.get("source_evidence", []) if isinstance(x, dict)]
        source_urls = [str(x.get("url", "")).strip() for x in source_evidence if str(x.get("url", "")).strip()]
        source_tier_map = {
            _normalize_source_url(str(x.get("url", "")).strip()): str(x.get("source_tier", "")).strip().lower()
            for x in source_evidence
            if str(x.get("url", "")).strip()
        }
        alignment_scores: list[float] = []
        lines = text.split("\n")
        new_lines: list[str] = []
        for line in lines:
            if "[E]" not in line:
                new_lines.append(line)
                continue
            urls_in_line = [u.rstrip("/,.") for u in _URL_PATTERN.findall(line) if str(u).strip()]
            if not urls_in_line:
                # Phase 1 discipline gate: [E] must include a source URL in the same line.
                score = _alignment(line, source_evidence)
                alignment_scores.append(score)
                new_lines.append(line.replace("[E]", "[I]", 1))
                continue
            normalized_urls = [_normalize_source_url(url) for url in urls_in_line]
            tiers = [
                source_tier_map.get(url)
                or source_tier_map.get(url.rstrip("/,."))
                or "tier3"
                for url in normalized_urls
            ]
            if tiers and all(str(t).strip().lower() == "tier4" for t in tiers):
                # Analogy-only sources cannot be promoted to direct evidence.
                new_lines.append(line.replace("[E]", "[A]", 1))
                continue
            new_lines.append(line)
        new_item = dict(item)
        new_item["finding"] = "\n".join(new_lines)
        if source_urls:
            new_item["source_urls"] = source_urls
        if alignment_scores:
            new_item["evidence_alignment_max"] = round(max(alignment_scores), 3)
        result.append(new_item)
    return result


def _self_check(client: OllamaClient, model_cfg: dict, question: str, finding: str) -> int:
    """Ask the agent to rate its own finding quality. Returns 1-5 or 0 on failure."""
    model = str(model_cfg.get("model", "")).strip()
    if not model or not client or not finding:
        return 0
    try:
        result = client.chat(
            model=model,
            system_prompt=(
                "Rate the quality and relevance of this research finding on a scale of 1-5.\n"
                "1=poor/off-topic or contains specific numbers, dates, names, or quotes with no cited source URL.\n"
                "2=weak — relevant but mostly unsourced or vague.\n"
                "3=adequate — answers the question with mostly sourced [E] claims.\n"
                "4=good — well-sourced, directly relevant, clear [E]/[I]/[S] discipline.\n"
                "5=excellent — directly answers the question, all specific claims cited, no apparent fabrication.\n"
                "Deduct at least 2 points if ANY [E] claim lacks an immediately following source URL or domain. "
                "[E] always requires a citation regardless of whether the claim is a statistic, name, or general observation. "
                "General knowledge presented as [E] without a source is a fabrication error.\n"
                "Reply with ONLY a single digit 1-5."
            ),
            user_prompt=f"Question: {question[:200]}\n\nFinding: {finding[:600]}",
            temperature=0.0,
            num_ctx=512,
            think=False,
            timeout=20,
            retry_attempts=1,
            retry_backoff_sec=0.5,
        )
        _match = re.search(r"[1-5]", str(result or "").strip())
        digit = _match.group(0) if _match else ""
        if digit in {"1", "2", "3", "4", "5"}:
            return int(digit)
    except Exception:
        pass
    return 0


def _extract_self_score(finding: str) -> tuple[str, dict[str, float | str] | None, str]:
    text = str(finding or "").strip()
    if not text:
        return text, None, "empty finding"
    lines = text.splitlines()
    score_idx = -1
    score_match = None
    for idx in range(len(lines) - 1, -1, -1):
        match = _SELF_SCORE_RE.match(lines[idx].strip())
        if match:
            score_idx = idx
            score_match = match
            break
    if score_match is None:
        return text, None, "missing SELF_SCORE line"
    try:
        confidence = float(score_match.group(1))
        coverage = float(score_match.group(2))
    except (TypeError, ValueError):
        return text, None, "invalid SELF_SCORE numeric values"
    if not (0.0 <= confidence <= 1.0 and 0.0 <= coverage <= 1.0):
        return text, None, "SELF_SCORE values out of range"
    notes = str(score_match.group(3) or "").strip()[:180]
    kept_lines = [line for i, line in enumerate(lines) if i != score_idx]
    clean = "\n".join(kept_lines).strip()
    return clean, {"confidence": confidence, "coverage": coverage, "notes": notes}, ""


def _extract_open_questions(summary_md: str) -> list[str]:
    lines = str(summary_md or "").splitlines()
    in_section = False
    questions: list[str] = []
    for raw in lines:
        line = str(raw or "").rstrip()
        if _OPEN_QUESTIONS_HEADING_RE.match(line):
            in_section = True
            continue
        if in_section and line.strip().startswith("## "):
            break
        if not in_section:
            continue
        match = _BULLET_LINE_RE.match(line)
        if not match:
            continue
        cleaned = re.sub(r"\[[EIS]\]", "", match.group(1)).strip(" .-")
        if len(cleaned) >= 8:
            questions.append(cleaned)
    # Deduplicate while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for row in questions:
        key = re.sub(r"\W+", "", row.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _normalize_question(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _count_recycled_open_questions(new_summary_md: str, prior_questions: list[str], threshold: float = 0.8) -> int:
    if not prior_questions:
        return 0
    current = _extract_open_questions(new_summary_md)
    if not current:
        return 0
    prior_norm = [_normalize_question(x) for x in prior_questions if _normalize_question(x)]
    if not prior_norm:
        return 0
    recycled = 0
    for question in current:
        norm = _normalize_question(question)
        if not norm:
            continue
        best = 0.0
        for prev in prior_norm:
            score = SequenceMatcher(a=norm, b=prev).ratio()
            if score > best:
                best = score
        if best >= float(threshold):
            recycled += 1
    return recycled


def _load_prior_open_questions(repo_root: Path, project_slug: str, *, exclude_summary_name: str = "") -> list[str]:
    root = repo_root / "Projects" / str(project_slug or "").strip() / "research_summaries"
    if not root.exists():
        return []
    excluded = str(exclude_summary_name or "").strip()
    candidates = []
    for path in sorted(root.glob("*.md")):
        name = path.name
        if not path.is_file():
            continue
        if name.endswith(".critique.md"):
            continue
        if excluded and name == excluded:
            continue
        candidates.append(path)
    if not candidates:
        return []
    latest = candidates[-1]
    try:
        text = latest.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    return _extract_open_questions(text)


def _build_source_quality_footer(findings: list[dict[str, Any]], *, suffix: str = "") -> str:
    scored_rows = [row for row in findings if isinstance(row, dict) and isinstance(row.get("self_score_confidence"), (int, float))]
    if not scored_rows:
        parse_errors = sum(
            1
            for row in findings
            if isinstance(row, dict) and str(row.get("self_score_parse_error", "")).strip()
        )
        detail = f" ({parse_errors} parse error(s))" if parse_errors else ""
        return f"**Source Quality** — agent self-scoring unavailable{detail}{suffix}"

    avg_conf = sum(float(row.get("self_score_confidence", 0.0) or 0.0) for row in scored_rows) / max(1, len(scored_rows))
    avg_cov = sum(float(row.get("self_score_coverage", 0.0) or 0.0) for row in scored_rows) / max(1, len(scored_rows))
    line = (
        f"**Source Quality** — {len(scored_rows)}/{len(findings)} agents self-scored "
        f"(avg confidence {avg_conf:.2f}, avg coverage {avg_cov:.2f}){suffix}"
    )
    per_agent = []
    for row in findings:
        agent = str(row.get("agent", "agent")).strip() or "agent"
        if isinstance(row.get("self_score_confidence"), (int, float)):
            conf = float(row.get("self_score_confidence", 0.0) or 0.0)
            cov = float(row.get("self_score_coverage", 0.0) or 0.0)
            notes = str(row.get("self_score_notes", "")).strip()
            snippet = f"- {agent}: confidence={conf:.2f}, coverage={cov:.2f}"
            if notes:
                snippet += f" ({notes[:80]})"
            per_agent.append(snippet)
        else:
            err = str(row.get("self_score_parse_error", "")).strip()
            per_agent.append(f"- {agent}: self-score missing ({err or 'parse error'})")
    return line + "\n\n" + "\n".join(per_agent)


def _lexical_overlap(a: str, b: str) -> float:
    left = set(re.findall(r"[a-z0-9]{4,}", str(a or "").lower()))
    right = set(re.findall(r"[a-z0-9]{4,}", str(b or "").lower()))
    if not left or not right:
        return 0.0
    return float(len(left & right) / max(1, len(left)))


def _outcome_quality(client: Any, question: str, finding: str) -> float:
    prompt = str(question or "").strip()[:1200]
    text = str(finding or "").strip()[:2400]
    if not prompt or not text:
        return 0.0
    try:
        vec_a = client.embed("qwen3-embedding:4b", prompt, timeout=20)
        vec_b = client.embed("qwen3-embedding:4b", text, timeout=20)
        score = float(_vec_cosine(vec_a, vec_b))
        if score > 0.0:
            return max(0.0, min(1.0, score))
    except Exception:
        pass
    return max(0.0, min(1.0, _lexical_overlap(prompt, text)))


def _gap_assess(client: Any, model_cfg: dict, question: str, summary_md: str) -> list[str]:
    """Identify 1-3 research gaps in the synthesis using a fast LLM call.

    Returns a list of specific gap questions, or [] on failure/timeout (loop is skipped).
    Uses a 15s timeout with no retries — if the model is busy, skip gap fill entirely.
    """
    model = str(model_cfg.get("model", "")).strip()
    if not model or not client or not summary_md.strip():
        return []
    try:
        result = client.chat(
            model=model,
            system_prompt=(
                "You are a research gap analyst. Your job is to identify the most important claims "
                "in a research synthesis that lack direct supporting evidence or remain unresolved. "
                "Output ONLY the gap questions, one per line. No preamble, no numbering, no explanations. "
                "Maximum 3 questions. If the synthesis is comprehensive, output just 1."
            ),
            user_prompt=(
                f"Research question: {question[:300]}\n\n"
                f"Synthesis:\n{summary_md[:3000]}\n\n"
                "List the 2-3 most important gaps as specific research questions, one per line:"
            ),
            temperature=0.2,
            num_ctx=1024,
            think=False,
            timeout=15,
            retry_attempts=1,
            retry_backoff_sec=0.5,
        )
        raw = str(result or "").strip()
        gaps = [
            line.strip().lstrip("0123456789.-) \t").strip()
            for line in raw.splitlines()
            if line.strip()
        ]
        gaps = [g for g in gaps if len(g) >= 15][:3]
        return gaps
    except Exception:
        return []


RESEARCH_PERSONAS = [
    ("market_analyst", "Focus on market dynamics, alternatives, and strategic positioning."),
    ("technical_researcher", "Focus on technical feasibility, architecture tradeoffs, and bottlenecks."),
    ("risk_researcher", "Focus on risks, failure modes, constraints, and mitigation plans."),
    ("execution_planner", "Focus on practical sequencing, milestones, and resource-fit execution."),
]
DEFAULT_DIRECTIVES = {persona: directive for persona, directive in RESEARCH_PERSONAS}
ANALYSIS_PROFILE_TECHNICAL      = "technical_analysis"
ANALYSIS_PROFILE_GENERAL        = "general_analysis"
ANALYSIS_PROFILE_MEDICAL        = "medical_analysis"
ANALYSIS_PROFILE_PARENTING      = "parenting_analysis"
ANALYSIS_PROFILE_FINANCE        = "finance_analysis"
ANALYSIS_PROFILE_SPORTS         = "sports_analysis"
ANALYSIS_PROFILE_HISTORY        = "history_analysis"
ANALYSIS_PROFILE_SCIENCE        = "science_analysis"
ANALYSIS_PROFILE_MATH           = "math_analysis"
ANALYSIS_PROFILE_POLITICS       = "politics_analysis"
ANALYSIS_PROFILE_CURRENT_EVENTS = "current_events_analysis"
ANALYSIS_PROFILE_UNDERGROUND    = "underground_analysis"
STATISTICAL_ANALYSIS_PERSONA = "statistical_analysis"
STATISTICAL_ANALYSIS_DIRECTIVE = (
    "OUTPUT CONTRACT: Return ONLY quantitative findings from sources. "
    "Allowed finding types: prevalence or incidence rates, effect sizes or risk ratios, measurable thresholds, "
    "comparative rates across groups/interventions, longitudinal trends with time windows, and uncertainty bounds "
    "or confidence intervals when provided. If sources contain no quantitative data, return exactly this line under "
    "Findings: 'No quantitative data extractable from available sources.' "
    "Do NOT restate mechanism/prevention/management prose covered by other personas. Numeric signal only."
)
LEGAL_ANALYSIS_PERSONA = "legal_analysis"
LEGAL_ANALYSIS_DIRECTIVE = (
    "Focus on legal and compliance constraints, jurisdiction caveats, and explicit risk language. "
    "Flag where professional legal counsel is required."
)
STATISTICAL_ANALYSIS_MODEL = "qwen3:8b"
LEGAL_ANALYSIS_MODEL = "qwen3:8b"


def _refresh_dynamic_models(repo_root: Path) -> None:
    global STATISTICAL_ANALYSIS_MODEL, LEGAL_ANALYSIS_MODEL
    try:
        cfg = load_config(repo_root)
        STATISTICAL_ANALYSIS_MODEL = cfg.get_model("research_statistical")
        LEGAL_ANALYSIS_MODEL = cfg.get_model("research_legal")
    except Exception:
        pass

TOPIC_TYPE_TO_PROFILE: dict[str, str] = {
    "sports":         ANALYSIS_PROFILE_SPORTS,
    "technical":      ANALYSIS_PROFILE_TECHNICAL,
    "medical":        ANALYSIS_PROFILE_MEDICAL,
    "animal_care":    ANALYSIS_PROFILE_MEDICAL,
    "finance":        ANALYSIS_PROFILE_FINANCE,
    "history":        ANALYSIS_PROFILE_HISTORY,
    "science":        ANALYSIS_PROFILE_SCIENCE,
    "math":           ANALYSIS_PROFILE_MATH,
    "politics":       ANALYSIS_PROFILE_POLITICS,
    "current_events": ANALYSIS_PROFILE_CURRENT_EVENTS,
    "general":        ANALYSIS_PROFILE_GENERAL,
    "underground":    ANALYSIS_PROFILE_UNDERGROUND,
    "business":       ANALYSIS_PROFILE_FINANCE,
    "law":            ANALYSIS_PROFILE_POLITICS,
    "education":      ANALYSIS_PROFILE_GENERAL,
    "travel":         ANALYSIS_PROFILE_GENERAL,
    "food":           ANALYSIS_PROFILE_GENERAL,
    "gaming":         ANALYSIS_PROFILE_TECHNICAL,
    "books":          ANALYSIS_PROFILE_GENERAL,
    "real_estate":    ANALYSIS_PROFILE_FINANCE,
    "automotive":     ANALYSIS_PROFILE_TECHNICAL,
    "parenting":      ANALYSIS_PROFILE_PARENTING,
    "tv_shows":       ANALYSIS_PROFILE_CURRENT_EVENTS,
    "movies":         ANALYSIS_PROFILE_CURRENT_EVENTS,
    "music":          ANALYSIS_PROFILE_CURRENT_EVENTS,
    "art":            ANALYSIS_PROFILE_CURRENT_EVENTS,
}


def _analysis_profile_for_type(topic_type: str) -> str:
    return TOPIC_TYPE_TO_PROFILE.get(str(topic_type).strip().lower(), ANALYSIS_PROFILE_GENERAL)


def _sanitize_model_list(raw_models: Any) -> list[str]:
    models: list[str] = []
    if isinstance(raw_models, list):
        for entry in raw_models:
            name = str(entry or "").strip()
            if not name:
                continue
            # Fully retire this model from research usage.
            if name.lower() == "qwen3:4b":
                continue
            if name not in models:
                models.append(name)
    return models


def _profile_agent_templates(profile: str) -> list[dict[str, Any]]:
    if profile == ANALYSIS_PROFILE_SPORTS:
        return [
            {
                "persona": "sports_context_researcher",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on current schedules, rosters, recent form, rankings, and event context. "
                    "For combat sports: confirm weight class, title type (divisional vs symbolic belt such as BMF), "
                    "event date relative to today, and flag card changes or injury substitutions."
                ),
            },
            {
                "persona": "sports_stats_and_history_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on head-to-head records, statistical trends, historical performance trajectory. "
                    "Cite specific figures with dates."
                ),
            },
            {
                "persona": "sports_risk_analyst",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on injury reports, availability uncertainty, current momentum, venue/officiating factors, "
                    "and what could shift the expected outcome."
                ),
            },
        ]
    if profile == ANALYSIS_PROFILE_TECHNICAL:
        return [
            {
                "persona": "technical_architecture_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on system design patterns, architectural tradeoffs, scalability, and technology choices. "
                    "Compare competing approaches with evidence."
                ),
            },
            {
                "persona": "technical_implementation_researcher",
                "model": "qwen2.5-coder:7b",
                "directive": (
                    "Focus on concrete implementation patterns, library/framework comparisons, code-level feasibility, "
                    "API shapes, version specifics, and known gotchas."
                ),
            },
            {
                "persona": "technical_risk_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on security vulnerabilities, failure modes, performance bottlenecks, "
                    "maintenance burden, and technical debt."
                ),
            },
            {
                "persona": "technical_market_analyst",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on ecosystem maturity, adoption trends, community support, and competitive alternatives."
                ),
                "role": "advisory",
            },
        ]
    if profile == ANALYSIS_PROFILE_MEDICAL:
        return [
            {
                "persona": "clinical_evidence_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on peer-reviewed evidence, trial data, systematic reviews. Note study quality, sample sizes, recency. "
                    "Tag by evidence tier: RCT > observational > case study > expert opinion. "
                    "For every statistic or prevalence figure, include the publication year and flag if data is older than 3 years."
                ),
            },
            {
                "persona": "guideline_verifier",
                "model": "qwen3:8b",
                "directive": (
                    "Cross-check against current clinical guidelines (WHO, CDC, NIH, specialty societies). "
                    "Explicitly state the guideline version year (e.g., 'CDC 2023'). Flag when the most recent guideline is more than 3 years old. "
                    "Note evidence-guideline divergences and any guidelines under active revision."
                ),
            },
            {
                "persona": "safety_risk_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on contraindications, adverse event profiles, drug interactions, population-specific risks. "
                    "Flag black box warnings and active regulatory advisories."
                ),
            },
            {
                "persona": LEGAL_ANALYSIS_PERSONA,
                "model": LEGAL_ANALYSIS_MODEL,
                "directive": LEGAL_ANALYSIS_DIRECTIVE,
                "role": "advisory",
            },
        ]
    if profile == ANALYSIS_PROFILE_PARENTING:
        return [
            {
                "persona": "developmental_evidence_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on peer-reviewed developmental psychology, pediatric research, and educational studies. "
                    "Tag evidence tier (RCT > observational > case study > expert opinion) and include publication years. "
                    "Flag any prevalence statistics with their source year — developmental norms shift over time. "
                    "For neurodiverse populations (autism, ADHD, sensory processing differences), note when study samples "
                    "are representative vs. skewed (e.g., predominantly male samples, clinical vs. community populations)."
                ),
            },
            {
                "persona": "clinical_guideline_verifier",
                "model": "qwen3:8b",
                "directive": (
                    "Cross-check against current pediatric and developmental guidelines (AAP, CDC, AOTA, ASHA, DSM-5-TR). "
                    "Explicitly state guideline version years. Flag guidelines older than 3 years. "
                    "Note where guidelines are being revised or where evidence and current practice diverge."
                ),
            },
            {
                "persona": "neurodiversity_perspective_researcher",
                "model": "qwen3:8b",
                "directive": (
                    "Actively seek neurodiversity-affirming frameworks, perspectives, and research. "
                    "This means: (1) Look for research and guidance written from a strengths-based or identity-affirming lens, not deficit-only. "
                    "(2) Identify where the primary literature reflects a predominantly neurotypical or pathology framing and flag it. "
                    "(3) Seek out autistic self-advocate perspectives, disability justice viewpoints, and culturally responsive approaches. "
                    "(4) Flag where interventions have been critiqued by the autistic community vs. endorsed. "
                    "(5) Look for intersectional considerations: how do gender, race, culture, and socioeconomic status affect diagnosis rates, "
                    "access to support, and outcomes for neurodiverse children?"
                ),
            },
            {
                "persona": "practical_family_advisor",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on actionable, practical strategies families can use. Prioritize approaches that have real-world parent/caregiver evidence. "
                    "Identify what school systems, therapists, and pediatricians can be asked for specifically. "
                    "Flag cost, accessibility, and availability barriers. "
                    "Note where online communities (e.g., autistic-led spaces, parent support groups) offer supplementary lived-experience knowledge "
                    "beyond what appears in clinical literature."
                ),
            },
        ]
    if profile == ANALYSIS_PROFILE_FINANCE:
        return [
            {
                "persona": "macro_market_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on macroeconomic indicators, market trends, sector dynamics, monetary/fiscal policy. "
                    "Cite data points with sources and dates."
                ),
            },
            {
                "persona": "fundamentals_researcher",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on valuation multiples, earnings/revenue trends, balance sheet health, competitive positioning."
                ),
            },
            {
                "persona": "risk_stress_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on downside scenarios, tail risks, liquidity constraints, regulatory headwinds. "
                    "What breaks this thesis first?"
                ),
            },
            {
                "persona": LEGAL_ANALYSIS_PERSONA,
                "model": LEGAL_ANALYSIS_MODEL,
                "directive": LEGAL_ANALYSIS_DIRECTIVE,
                "role": "advisory",
            },
        ]
    if profile == ANALYSIS_PROFILE_HISTORY:
        return [
            {
                "persona": "history_timeline_researcher",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on chronology, causal chains, periodization with explicit date anchors. "
                    "Identify pivotal turning points and distinguish immediate causes from structural forces."
                ),
            },
            {
                "persona": "history_source_critic",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on source quality, authorial bias, historiographical disputes, and missing/contested evidence. "
                    "Actively challenge the dominant narrative."
                ),
            },
            {
                "persona": "history_comparative_analyst",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on parallels with other periods or regions. What does this resemble? "
                    "What's different? What precedents exist and how reliable are they?"
                ),
            },
        ]
    if profile == ANALYSIS_PROFILE_SCIENCE:
        return [
            {
                "persona": "scientific_evidence_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on peer-reviewed research, experimental findings, and current scientific consensus. "
                    "Note methodology quality, replication status. Distinguish established consensus from active frontier debate."
                ),
            },
            {
                "persona": "frontier_science_analyst",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on cutting-edge preprints, recent papers, emerging findings, and where the field is actively moving. "
                    "Flag contested vs widely accepted claims."
                ),
            },
            {
                "persona": "science_application_researcher",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on real-world applications, technology readiness level, practical implications, "
                    "and how this science connects to existing technologies or societal challenges."
                ),
            },
        ]
    if profile == ANALYSIS_PROFILE_MATH:
        return [
            {
                "persona": "formal_reasoning_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on rigorous mathematical foundations, proof structures, axioms and assumptions, logical validity. "
                    "Identify where informal reasoning substitutes for proof."
                ),
            },
            {
                "persona": "computational_methods_researcher",
                "model": "qwen2.5-coder:7b",
                "directive": (
                    "Focus on algorithms, numerical methods, computational complexity, and implementation approaches. "
                    "Compare efficiency and accuracy tradeoffs with examples."
                ),
            },
            {
                "persona": "applied_math_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on real-world modeling applications, statistical methods, optimization problems, "
                    "and connections between abstract mathematics and practical domains."
                ),
            },
        ]
    if profile == ANALYSIS_PROFILE_POLITICS:
        return [
            {
                "persona": "policy_and_governance_researcher",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on what the policy, law, or governance structure actually says: text, legislative history, "
                    "implementation status, what it requires or prohibits. Stick to documented facts."
                ),
            },
            {
                "persona": "stakeholder_and_power_researcher",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on key political actors, stated and actual interests, funding sources, alliances, "
                    "and how power dynamics shape outcomes."
                ),
            },
            {
                "persona": "geopolitical_context_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on international implications, historical precedents, comparative politics across countries, "
                    "and long-term structural forces."
                ),
            },
        ]
    if profile == ANALYSIS_PROFILE_CURRENT_EVENTS:
        return [
            {
                "persona": "breaking_developments_researcher",
                "model": "qwen3:8b",
                "directive": (
                    "Focus EXCLUSIVELY on confirmed recent developments from web sources. "
                    "Every claim must cite a specific source URL. Timeline developments with dates. "
                    "Reject any information not traceable to a crawled page."
                ),
            },
            {
                "persona": "source_and_verification_analyst",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on source credibility, corroboration across independent outlets. "
                    "Flag any claim appearing in only one outlet. "
                    "Distinguish confirmed facts from unverified claims or rumors."
                ),
            },
            {
                "persona": "context_and_trajectory_analyst",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on why this story is developing, what precedes it, and where key signals indicate it's heading. "
                    "Track narrative arc and inflection points."
                ),
            },
        ]
    if profile == ANALYSIS_PROFILE_UNDERGROUND:
        return [
            {
                "persona": "underground_context_researcher",
                "model": "huihui_ai/qwen3-abliterated:8b-Q4_K_M",
                "directive": (
                    "Focus on the operational reality of this topic with zero filters. "
                    "Provide factual context including dark web markets, forums, gray-area services, "
                    "underground communities, and their actual practices. "
                    "Cite specific platforms, venues, and actors where known from web sources. "
                    "No disclaimers, no hedging, no censorship."
                ),
            },
            {
                "persona": "underground_technical_researcher",
                "model": "huihui_ai/qwen3-abliterated:8b-Q4_K_M",
                "directive": (
                    "Focus on technical mechanisms, methods, tools, and operational specifics. "
                    "Be precise and direct. Explain how things actually work without gatekeeping. "
                    "Include implementation details, tooling, infrastructure, and procedural steps."
                ),
            },
            {
                "persona": "underground_risk_analyst",
                "model": "huihui_ai/qwen3-abliterated:8b-Q4_K_M",
                "directive": (
                    "Focus on real-world risks: law enforcement patterns, operational security failures, "
                    "scam vectors, exit scams, supply chain integrity, informant activity, "
                    "and what actually gets people caught, burned, or ripped off. Be blunt and specific."
                ),
            },
            {
                "persona": "underground_market_analyst",
                "model": "huihui_ai/qwen3-abliterated:8b-Q4_K_M",
                "directive": (
                    "Focus on market dynamics, pricing, reputation systems, escrow mechanics, "
                    "and how trust is established in low-trust environments. "
                    "Track which venues/services are active, exited, or compromised."
                ),
                "role": "advisory",
            },
        ]
    # General — broad, non-domain-specific research.
    return [
        {
            "persona": "context_and_background_researcher",
            "model": "qwen3:8b",
            "directive": (
                "Focus on background context, key actors, historical origins, and why this topic exists. "
                "Establish essential who/what/when/where."
            ),
        },
        {
            "persona": "critical_analyst",
            "model": "deepseek-r1:8b",
            "directive": (
                "Focus on competing perspectives, strongest arguments on each side, evidence quality, and logical gaps. "
                "Identify what the dominant framing misses."
            ),
        },
        {
            "persona": "implications_researcher",
            "model": "qwen3:8b",
            "directive": (
                "Focus on second-order effects, downstream consequences, stakeholder impacts, "
                "and what matters most for someone who needs to act on this."
            ),
        },
    ]


def _trim_text_block(text: str, max_chars: int, *, tail_note: str) -> str:
    body = str(text or "").strip()
    if len(body) <= max_chars:
        return body
    clipped = body[:max_chars].rsplit("\n", 1)[0].strip()
    if not clipped:
        clipped = body[:max_chars].strip()
    removed = max(0, len(body) - len(clipped))
    return f"{clipped}\n\n[{tail_note}; trimmed {removed} chars]"


def _is_failure_text(text: str) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return True
    markers = [
        "model call failed",
        "fallback failed",
        "ollama chat failed",
        "no model configured",
        "could not connect to ollama",
        "ollama http 5",
        "traceback",
    ]
    return any(token in low for token in markers)


def _looks_like_research_note(text: str) -> bool:
    body = str(text or "").strip()
    if len(body) < 220:
        return False
    if _is_failure_text(body):
        return False
    low = body.lower()
    section_hits = 0
    for token in ("findings", "evidence", "open questions", "open question", "risks", "next steps"):
        if token in low:
            section_hits += 1
    return section_hits >= 2


def _extract_handoff_request(
    raw_text: str,
    *,
    persona: str,
    allowed_personas: set[str],
    min_confidence: float = 0.75,
) -> dict[str, Any] | None:
    body = str(raw_text or "").strip()
    if not body:
        return None
    candidates: list[dict[str, Any]] = []
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            candidates.append(parsed)
    except Exception:
        pass
    if not candidates:
        match = re.search(r"\{.*\}", body, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                candidates.append(parsed)
    for row in candidates:
        target = str(row.get("handoff_to", row.get("to", ""))).strip().lower()
        reason = str(row.get("reason", "")).strip()
        try:
            confidence = float(row.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if (
            target
            and target != str(persona).strip().lower()
            and target in {p.lower() for p in allowed_personas}
            and confidence >= float(min_confidence)
        ):
            canonical = next((p for p in allowed_personas if p.lower() == target), target)
            return {
                "to": canonical,
                "reason": reason[:240],
                "confidence": round(confidence, 3),
            }
    return None


def _agent_prompt(question: str, persona: str, directive: str, learned_guidance: str, web_context: str, max_web_chars: int = 9000) -> tuple[str, str]:
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    guidance_block = f"\n\n{learned_guidance}" if learned_guidance else ""
    web_block = ""
    if web_context.strip():
        web_context_trimmed = _trim_text_block(
            web_context,
            max_chars=max_web_chars,
            tail_note="web source cache truncated for reliability",
        )
        web_block = (
            "\n\nUse the web source context below selectively and cite source URLs in your notes. "
            "Do not discuss how sources were obtained."
            f"\n\nWeb source context:\n{web_context_trimmed}"
        )
    system_prompt = (
        f"Today's date: {today_str}. "
        "You are a Foraging sub-agent in a multi-agent council. "
        f"Your role is {persona}. {directive} "
        "Be concrete and avoid vague statements. "
        "Format output as markdown with sections: Findings, Evidence Signals, Open Questions.\n\n"
        "CLAIM LABELING — tag every substantive claim with one of:\n"
        "  [E] directly supported by a cited source or explicit data point\n"
        "  [I] logically inferred from evidence — reasonable but not directly stated\n"
        "  [S] speculative or hypothetical — plausible but no direct source backing\n"
        "Cite the source URL or domain after every [E] claim. "
        "Never present [I] or [S] claims as established facts.\n\n"
        "FABRICATION PROHIBITED: Do not state specific numbers, statistics, dates, names, "
        "product versions, prices, or direct quotes without a cited source URL. "
        "If you cannot find a source for a specific detail, omit it or write "
        "'[source not found]' — do not guess or approximate. "
        "Stating 'the available sources do not cover this' is correct and preferred over filling gaps. "
        "Use [S] only as a last resort for genuine hypotheses, never to launder missing facts. "
        "Uncertainty is not a failure — fabrication is.\n\n"
        "SOURCE INTEGRITY: Cite only external sources retrieved this session (web pages, PDFs, docs). "
        "Prior Foxforge research files, summaries, critiques, and project notes are internal artifacts, "
        "not sources. Do not cite them.\n\n"
        "SOURCE QUALITY: Prefer high-quality institutional domains (.gov, .edu, recognized medical/scientific "
        "institutions, standards bodies). If you cite a low-editorial platform (for example Medium/LinkedIn/Substack), "
        "label it as platform/blog evidence.\n\n"
        "SELF SCORING: The final line of your response MUST match this exact format "
        "(copy this template and substitute real values — do NOT write it as a section heading):\n"
        "Literal example to copy (final line only):\n"
        "SELF_SCORE: confidence=0.82; coverage=0.71; notes=good coverage but weak on legal edge cases\n"
        "Do NOT output a heading like '# Self Score' or '## Self Score'.\n"
        "The final line must start with 'SELF_SCORE:' and include numeric values.\n"
        "Rules:\n"
        "- confidence and coverage are floats between 0.0 and 1.0\n"
        "- notes is a short free-form string under 180 chars\n"
        "- No blank lines after this line. Nothing else after it.\n"
        "- Do NOT prefix with '#': this is data, not a markdown heading.\n"
        "- If you omit this line, your finding is discarded as unusable.\n\n"
        "PERSONA HANDOFF: If this request is materially better handled by another persona, "
        "you may return ONLY JSON with shape "
        "{\"handoff_to\":\"<persona>\",\"reason\":\"<one sentence>\",\"confidence\":0.0-1.0}. "
        "Only do this when confidence is >= 0.75."
        f"{guidance_block}{web_block}"
    )
    user_prompt = (
        f"Research request:\n{question}\n\n"
        "Return high-signal research notes that can be merged by a synthesizer."
    )
    return system_prompt, user_prompt


def _history_block(prior_messages: list[dict[str, str]] | None, limit_turns: int = 10) -> str:
    if not isinstance(prior_messages, list):
        return ""
    rows: list[str] = []
    for row in prior_messages[-max(6, limit_turns * 2) :]:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role", "")).strip().lower()
        content = _trim_text_block(
            str(row.get("content", "")).strip(),
            max_chars=520,
            tail_note="message truncated",
        )
        if role not in {"user", "assistant"} or not content:
            continue
        tag = "USER" if role == "user" else "ASSISTANT"
        rows.append(f"{tag}: {content}")
    if not rows:
        return ""
    return (
        "Recent command-thread context (for query shaping, not evidence unless repeated in the literal research question):\n"
        + "\n".join(rows)
    )


_MULTI_PASS_BATCH_SIZE = 6   # sources per LLM pass (doubled from 3 — models have 24K ctx)
_MULTI_PASS_THRESHOLD = 4   # only batch when there are more than this many source blocks


def _split_web_sources(web_context: str) -> tuple[str, list[str]]:
    """Split web_context into (header_line, [source_block, ...]).

    Each source block starts with a "- " line (tier/depth prefix) as written by
    WebResearchEngine.web_context_for_project().
    """
    lines = web_context.strip().split("\n")
    if len(lines) <= 1:
        return web_context, []
    header = lines[0]
    source_blocks: list[str] = []
    current: list[str] = []
    for line in lines[1:]:
        if line.startswith("- "):
            if current:
                source_blocks.append("\n".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        source_blocks.append("\n".join(current))
    return header, source_blocks


def _run_one_agent(
    client: OllamaClient,
    model_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    question: str,
    learned_guidance: str,
    web_context: str,
    source_evidence: list[dict[str, str]] | None,
    prior_messages: list[dict[str, str]] | None,
    cancel_checker: Callable[[], bool] | None = None,
    pause_checker: Callable[[], bool] | None = None,
    allowed_personas: set[str] | None = None,
) -> dict[str, Any]:
    persona = str(agent_cfg.get("persona", "")).strip() or "research_agent"
    directive = str(agent_cfg.get("directive", "")).strip() or DEFAULT_DIRECTIVES.get(
        persona,
        "Focus on evidence quality, contradictions, and practical implications.",
    )
    base_model = str(model_cfg.get("model", "")).strip()
    requested_model = str(agent_cfg.get("model", "")).strip() or base_model
    if not requested_model:
        return {
            "agent": persona,
            "model": "",
            "requested_model": "",
            "finding": "No model configured for research_pool.",
            "source_urls": [],
            "source_evidence": [],
        }

    _max_web = 30000 if persona.startswith("breaking_") else (24000 if persona.startswith("sports_") else 20000)
    system_prompt, user_prompt = _agent_prompt(question, persona, directive, learned_guidance, web_context, max_web_chars=_max_web)
    _ = prior_messages
    temperature = float(agent_cfg.get("temperature", model_cfg.get("temperature", 0.3)))
    num_ctx = int(agent_cfg.get("num_ctx", model_cfg.get("num_ctx", 16384)))
    think = bool(agent_cfg.get("think", model_cfg.get("think", False)))
    timeout = int(agent_cfg.get("timeout_sec", model_cfg.get("timeout_sec", 0)))
    retry_attempts = int(agent_cfg.get("retry_attempts", model_cfg.get("retry_attempts", 6)))
    retry_backoff_sec = float(agent_cfg.get("retry_backoff_sec", model_cfg.get("retry_backoff_sec", 1.5)))
    validation_cycles = int(agent_cfg.get("validation_cycles", model_cfg.get("validation_cycles", 3)))

    fallback_models_raw = agent_cfg.get("fallback_models", model_cfg.get("fallback_models", []))
    fallback_models: list[str] = []
    if isinstance(fallback_models_raw, list):
        for item in fallback_models_raw:
            name = str(item or "").strip()
            if name:
                fallback_models.append(name)
    if base_model and requested_model != base_model:
        fallback_models.append(base_model)

    used_model = requested_model
    finding = ""
    failure_notes: list[str] = []
    for cycle in range(max(1, validation_cycles)):
        if callable(pause_checker):
            while True:
                try:
                    paused = bool(pause_checker())
                except Exception:
                    paused = False
                if not paused:
                    break
                if callable(cancel_checker):
                    try:
                        if bool(cancel_checker()):
                            finding = f"Cancelled by user before {persona} could complete."
                            break
                    except Exception:
                        pass
                time.sleep(0.4)
            if finding.lower().startswith("cancelled by user"):
                break
        if callable(cancel_checker):
            try:
                if bool(cancel_checker()):
                    finding = f"Cancelled by user before {persona} could complete."
                    break
            except Exception:
                pass
        cycle_prompt = user_prompt
        if cycle > 0:
            cycle_prompt = (
                f"{user_prompt}\n\n"
                "Regenerate with stricter rigor. Include clear sections for Findings, Evidence Signals, and Open Questions."
            )
        try:
            finding = client.chat(
                model=requested_model,
                fallback_models=fallback_models,
                system_prompt=system_prompt,
                user_prompt=cycle_prompt,
                temperature=temperature,
                num_ctx=num_ctx,
                think=think,
                timeout=timeout,
                retry_attempts=max(1, retry_attempts),
                retry_backoff_sec=max(0.0, retry_backoff_sec),
            )
            if _looks_like_research_note(finding):
                break
            failure_notes.append(f"validation cycle {cycle + 1}: weak structure/content")
            if cycle == (max(1, validation_cycles) - 1):
                finding = (
                    f"{finding}\n\n"
                    "_Reliability note: transport retries succeeded, but the response missed structure quality checks._"
                )
        except Exception as exc:
            failure_notes.append(str(exc))
            if cycle == (max(1, validation_cycles) - 1):
                finding = f"Model call failed for {persona} after retries and fallbacks: {exc}"

    if _is_failure_text(finding) and failure_notes:
        finding = f"{finding}\n\nReliability diagnostics: {' | '.join(failure_notes[-4:])}"

    role = str(agent_cfg.get("role", "primary")).strip() or "primary"
    finding_clean, self_score, score_error = _extract_self_score(finding)
    finding = finding_clean
    handoff = _extract_handoff_request(
        finding,
        persona=persona,
        allowed_personas=set(allowed_personas or set()),
    )
    source_rows = [dict(x) for x in (source_evidence or []) if isinstance(x, dict)]
    source_urls = [str(x.get("url", "")).strip() for x in source_rows if str(x.get("url", "")).strip()]
    row = {
        "agent": persona,
        "model": used_model,
        "requested_model": requested_model,
        "finding": finding,
        "role": role,
        "source_urls": source_urls,
        "source_evidence": source_rows,
        "handoff": handoff,
    }
    if self_score is not None:
        row["self_score_confidence"] = float(self_score.get("confidence", 0.0) or 0.0)
        row["self_score_coverage"] = float(self_score.get("coverage", 0.0) or 0.0)
        row["self_score_notes"] = str(self_score.get("notes", "")).strip()
    else:
        row["self_score_parse_error"] = score_error
    return row


def _agent_specs(model_cfg: dict[str, Any], topic_type: str = "general") -> list[dict[str, Any]]:
    profile = _analysis_profile_for_type(topic_type)
    templates = _profile_agent_templates(profile)
    default_validation_cycles = int(model_cfg.get("validation_cycles", 3))
    if profile in {ANALYSIS_PROFILE_MEDICAL, ANALYSIS_PROFILE_FINANCE, ANALYSIS_PROFILE_UNDERGROUND}:
        default_validation_cycles = max(4, default_validation_cycles)

    base_fallbacks = _sanitize_model_list(model_cfg.get("fallback_models", []))
    out: list[dict[str, Any]] = []
    for item in templates:
        row = dict(item)
        model_name = str(row.get("model", "")).strip()
        fallback = _sanitize_model_list(list(base_fallbacks) + [model_name])
        if model_name and model_name in fallback:
            fallback = [model_name] + [x for x in fallback if x != model_name]
        row["fallback_models"] = fallback
        row.setdefault("validation_cycles", default_validation_cycles)
        # deepseek-r1 has built-in chain-of-thought reasoning activated by think=True.
        # Enable it automatically for primary deepseek-r1 agents unless explicitly overridden.
        # Advisory agents do NOT get think=True — their findings are supplementary and
        # chain-of-thought overhead isn't justified for that role.
        _role = str(row.get("role", "primary")).strip()
        if str(row.get("model", "")).startswith("deepseek-r1") and "think" not in row and _role != "advisory":
            row["think"] = True
        out.append(row)
    return out


_POSITIVE_SIGNALS = re.compile(
    r"\b(increase[sd]?|rise[sd]?|rose|rises|improve[sd]?|gain[sd]?|grow[sth]?|grew|"
    r"strengthen[sed]?|accelerate[sd]?|surge[sd]?|win[sd]?|won|higher|more|outperform[sed]?)\b",
    re.I,
)
_NEGATIVE_SIGNALS = re.compile(
    r"\b(decrease[sd]?|decline[sd]?|fall[sd]?|fell|reduce[sd]?|lower|shrink[s]?|shrunk|"
    r"weaken[sed]?|worsen[sed]?|lose[sd]?|lost|fail[sed]?|risk[s]?|harm[sed]?|threaten[sed]?)\b",
    re.I,
)


def _cross_agent_conflict_report(findings: list[dict]) -> str:
    """Heuristic cross-agent conflict detection — no LLM call.

    Compares primary-role agent findings pairwise. For each pair, splits sentences
    and looks for any sentence containing the same root noun (3+ chars, alpha) where
    one sentence has positive directional signals and the other has negative ones.
    Returns a markdown block of conflicts, or empty string if none found.
    """
    primary = [f for f in findings if str(f.get("role", "primary")).lower() != "advisory"]
    if len(primary) < 2:
        return ""

    # Extract (agent, sentence, has_pos, has_neg) rows from each finding.
    rows: list[tuple[str, str, bool, bool]] = []
    for item in primary:
        agent = str(item.get("agent", "agent"))
        text = str(item.get("finding", ""))
        for sent in re.split(r"(?<=[.!?])\s+", text):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            has_pos = bool(_POSITIVE_SIGNALS.search(sent))
            has_neg = bool(_NEGATIVE_SIGNALS.search(sent))
            if has_pos or has_neg:
                rows.append((agent, sent, has_pos, has_neg))

    conflicts: list[str] = []
    seen: set[tuple[str, str]] = set()
    for i, (ag_a, sent_a, pos_a, neg_a) in enumerate(rows):
        # Extract noun tokens (4+ char alpha words, excluding common stop words).
        nouns_a = {
            w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", sent_a)
            if w.lower() not in {"this", "that", "with", "from", "into", "than", "their", "they", "have", "will", "been", "when", "where", "what", "which", "about", "more", "less", "also", "both", "each"}
        }
        for j, (ag_b, sent_b, pos_b, neg_b) in enumerate(rows):
            if j <= i or ag_a == ag_b:
                continue
            # Conflict: one positive, one negative, sharing a content noun.
            if not ((pos_a and neg_b) or (neg_a and pos_b)):
                continue
            nouns_b = {
                w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", sent_b)
                if w.lower() not in {"this", "that", "with", "from", "into", "than", "their", "they", "have", "will", "been", "when", "where", "what", "which", "about", "more", "less", "also", "both", "each"}
            }
            shared = nouns_a & nouns_b
            if len(shared) < 2:
                continue
            key = (ag_a, ag_b, tuple(sorted(shared))[:3])
            if key in seen:
                continue
            seen.add(key)
            shared_str = ", ".join(sorted(shared)[:3])
            snippet_a = sent_a[:120].rstrip()
            snippet_b = sent_b[:120].rstrip()
            conflicts.append(
                f"- **{ag_a}** (positive signals on: {shared_str}): \"{snippet_a}...\"\n"
                f"  **{ag_b}** (negative signals on: {shared_str}): \"{snippet_b}...\""
            )
            if len(conflicts) >= 5:
                break
        if len(conflicts) >= 5:
            break

    if not conflicts:
        return ""
    return "## Disputed Claims Across Agents\n" + "\n".join(conflicts)


def _reliability_summary(findings: list[dict[str, str]]) -> dict[str, int]:
    total = len(findings)
    failed = 0
    weak = 0
    for row in findings:
        text = str(row.get("finding", ""))
        if _is_failure_text(text):
            failed += 1
            continue
        if not _looks_like_research_note(text):
            weak += 1
    good = max(0, total - failed - weak)
    return {
        "agents_total": total,
        "good": good,
        "weak": weak,
        "failed": failed,
    }


def _run_fill_agents(
    *,
    client: Any,
    model_cfg: dict[str, Any],
    question: str,
    gap_queries: list[str],
    web_context: str,
    prior_messages: list[dict[str, str]] | None = None,
    findings: list[dict[str, Any]] | None = None,
    source_evidence: list[dict[str, str]] | None = None,
    cancel_checker: Callable[[], bool] | None = None,
    pause_checker: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    """Run exactly 2 targeted fill agents against identified gaps in parallel.

    Selects the two lowest-confidence primary agents from the first pass
    (falls back to technical_researcher + risk_researcher if unavailable).
    Each fill agent gets its standard directive augmented with the gap questions.
    Returns advisory-role findings, or [] if fill agents produce nothing useful.
    """
    if not gap_queries:
        return []
    findings = list(findings or [])

    gap_text = "\n".join(f"- {q}" for q in gap_queries)

    # Select the 2 lowest-confidence primary agent personas from the first pass.
    primary_findings = [f for f in findings if str(f.get("role", "primary")).strip().lower() != "advisory"]
    scored_primary = sorted(
        [f for f in primary_findings if isinstance(f.get("confidence"), (int, float))],
        key=lambda x: int(x.get("confidence", 0)),
    )

    fill_personas: list[str] = []
    for f in scored_primary[:2]:
        persona = str(f.get("agent", "")).strip()
        if persona and persona not in fill_personas:
            fill_personas.append(persona)

    # Fill to 2 with defaults if needed.
    for default in ("technical_researcher", "risk_researcher"):
        if len(fill_personas) >= 2:
            break
        if default not in fill_personas:
            fill_personas.append(default)

    fill_cfg_list: list[dict[str, Any]] = []
    for persona in fill_personas[:2]:
        base_directive = DEFAULT_DIRECTIVES.get(
            persona,
            "Focus on evidence quality, contradictions, and practical implications.",
        )
        fill_cfg_list.append({
            "persona": f"{persona}_gap_fill",
            "directive": f"{base_directive}\n\nFocus specifically on these gaps:\n{gap_text}",
            "model": str(model_cfg.get("model", "qwen3:8b")).strip(),
            "temperature": float(model_cfg.get("temperature", 0.3)),
            "num_ctx": int(model_cfg.get("num_ctx", 12288)),
            "think": False,
            "timeout_sec": 90,
            "retry_attempts": 3,
            "retry_backoff_sec": float(model_cfg.get("retry_backoff_sec", 1.5)),
            "validation_cycles": 1,
            "fallback_models": list(model_cfg.get("fallback_models") or []),
            "role": "advisory",
        })

    if not fill_cfg_list:
        return []

    fill_findings: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(fill_cfg_list)) as executor:
        futures = {
            executor.submit(
                _run_one_agent,
                client,
                model_cfg,
                cfg,
                question,
                "",  # no learned_guidance for fill pass
                web_context,
                source_evidence,
                prior_messages,
                cancel_checker,
                pause_checker,
                set(),
            ): cfg
            for cfg in fill_cfg_list
        }
        for future in futures:
            try:
                result = future.result()
                finding_text = str(result.get("finding", "")).strip()
                if finding_text and not _is_failure_text(finding_text):
                    if isinstance(result.get("self_score_confidence"), (int, float)):
                        score = max(1, min(5, int(round(float(result.get("self_score_confidence", 0.0)) * 5))))
                    else:
                        score = _self_check(client, model_cfg, question, finding_text)
                    result["confidence"] = score
                    result["role"] = "advisory"
                    fill_findings.append(result)
            except Exception:
                pass

    return fill_findings


def _recover_failed_findings(
    *,
    client: OllamaClient,
    findings: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    question: str,
    learned_guidance: str,
    web_context: str,
    source_evidence: list[dict[str, str]] | None,
    prior_messages: list[dict[str, str]] | None,
    cancel_checker: Callable[[], bool] | None = None,
    pause_checker: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    for row in findings:
        if callable(cancel_checker):
            try:
                if bool(cancel_checker()):
                    recovered.append(
                        {
                            "agent": "recovery",
                            "model": "",
                            "requested_model": "",
                            "finding": "Cancelled by user during recovery pass.",
                        }
                    )
                    break
            except Exception:
                pass
        text = str(row.get("finding", ""))
        if not _is_failure_text(text):
            recovered.append(row)
            continue

        persona = str(row.get("agent", "research_recovery")).strip() or "research_recovery"
        directive = DEFAULT_DIRECTIVES.get(persona, "Focus on evidence quality, contradictions, and practical implications.")
        emergency_cfg = {
            "persona": persona,
            "directive": directive,
            "model": str(model_cfg.get("model", "")).strip() or str(row.get("requested_model", "")).strip(),
            "temperature": float(model_cfg.get("temperature", 0.3)),
            "num_ctx": int(model_cfg.get("num_ctx", 16384)),
            "think": bool(model_cfg.get("think", False)),
            "timeout_sec": int(model_cfg.get("timeout_sec", 0)),
            "retry_attempts": int(model_cfg.get("retry_attempts", 6)) + 2,
            "retry_backoff_sec": float(model_cfg.get("retry_backoff_sec", 1.5)),
            "validation_cycles": int(model_cfg.get("validation_cycles", 3)),
            "fallback_models": model_cfg.get("fallback_models", []),
        }
        repaired = _run_one_agent(
            client,
            model_cfg,
            emergency_cfg,
            question,
            learned_guidance,
            web_context,
            source_evidence,
            prior_messages,
            cancel_checker,
            pause_checker,
            set(),
        )
        recovered.append(repaired)
    return recovered


def _run_multipass_agent(
    client: OllamaClient,
    model_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    question: str,
    learned_guidance: str,
    web_context: str,
    source_evidence: list[dict[str, str]] | None,
    prior_messages: list[dict[str, str]] | None = None,
    cancel_checker: Any = None,
    pause_checker: Any = None,
    allowed_personas: set[str] | None = None,
) -> dict[str, Any]:
    """Wrapper around _run_one_agent that processes sources in batches.

    When web_context contains more than _MULTI_PASS_THRESHOLD source blocks,
    splits them into batches of _MULTI_PASS_BATCH_SIZE and runs a separate
    LLM call per batch.  All partial findings are concatenated into one result.
    Falls back to a single _run_one_agent call when there are few sources.
    """
    header, source_blocks = _split_web_sources(web_context)
    if len(source_blocks) <= _MULTI_PASS_THRESHOLD:
        return _run_one_agent(
            client, model_cfg, agent_cfg, question, learned_guidance,
            web_context, source_evidence, prior_messages, cancel_checker, pause_checker, allowed_personas,
        )

    batches = [
        source_blocks[i: i + _MULTI_PASS_BATCH_SIZE]
        for i in range(0, len(source_blocks), _MULTI_PASS_BATCH_SIZE)
    ]
    batches = batches[:4]  # cap at 4 passes to bound LLM calls

    partial_findings: list[str] = []
    last_result: dict[str, Any] = {}
    original_directive = str(agent_cfg.get("directive", "")).strip()

    for idx, batch in enumerate(batches):
        batch_context = header + "\n" + "\n".join(batch)
        batch_cfg = dict(agent_cfg)
        batch_cfg["directive"] = (
            f"{original_directive}\n"
            f"[Source scan {idx + 1} of {len(batches)}: analyse only the sources in this batch.]"
        )
        result = _run_one_agent(
            client, model_cfg, batch_cfg, question, learned_guidance,
            batch_context, source_evidence, prior_messages, cancel_checker, pause_checker, allowed_personas,
        )
        last_result = result
        finding = str(result.get("finding", "")).strip()
        if finding and not finding.startswith("[FAILED]") and not finding.startswith("[No model"):
            partial_findings.append(f"[Scan {idx + 1}/{len(batches)}]\n{finding}")

    if not partial_findings:
        return last_result

    last_result = dict(last_result)
    last_result["finding"] = "\n\n---\n\n".join(partial_findings)
    return last_result


def run_research_pool(
    question: str,
    repo_root: Path,
    project_slug: str,
    bus,
    web_context: str = "",
    prior_messages: list[dict[str, str]] | None = None,
    cancel_checker: Callable[[], bool] | None = None,
    pause_checker: Callable[[], bool] | None = None,
    yield_checker: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    topic_type: str = "general",
    project_context: str = "",
) -> dict:
    bus.emit("research_pool", "start", {"question": question, "project": project_slug})
    _refresh_dynamic_models(repo_root)

    def _is_cancelled() -> bool:
        if callable(cancel_checker):
            try:
                return bool(cancel_checker())
            except Exception:
                return False
        return False

    def _progress(stage: str, detail: dict[str, Any] | None = None) -> None:
        if not callable(progress_callback):
            return
        try:
            progress_callback(stage, detail or {})
        except Exception:
            pass

    def _is_paused() -> bool:
        if callable(pause_checker):
            try:
                return bool(pause_checker())
            except Exception:
                return False
        return False

    def _should_yield() -> bool:
        if callable(yield_checker):
            try:
                return bool(yield_checker())
            except Exception:
                return False
        return False

    model_cfg = lane_model_config(repo_root, "research_pool")
    orchestrator_cfg = lane_model_config(repo_root, "orchestrator_reasoning")
    client = InferenceRouter(repo_root)
    learning = FeedbackLearningEngine(repo_root, client=client, model_cfg=orchestrator_cfg)
    learned_guidance = learning.guidance_for_lane("research", limit=5)
    resolved_type = str(topic_type or "general").strip().lower() or "general"
    profile_name = _analysis_profile_for_type(resolved_type)
    agents = _agent_specs(model_cfg, topic_type=resolved_type)
    allowed_personas = {
        str(row.get("persona", "")).strip()
        for row in agents
        if str(row.get("persona", "")).strip()
    }
    agent_cfg_by_persona = {
        str(row.get("persona", "")).strip(): dict(row)
        for row in agents
        if str(row.get("persona", "")).strip()
    }
    try:
        routing = load_model_routing(repo_root)
    except Exception:
        routing = {}
    raw_max_handoffs = (
        routing.get("research_pool.max_handoffs", model_cfg.get("max_handoffs", 2))
        if isinstance(routing, dict)
        else model_cfg.get("max_handoffs", 2)
    )
    max_handoffs = max(0, int(raw_max_handoffs or 2))
    handoff_count = 0
    visited_agents_per_leaf: dict[str, list[str]] = {"root": []}
    source_evidence = _extract_web_source_evidence(web_context)
    source_tier_map = _build_url_tier_map(source_evidence)
    worker_count = max(1, min(int(model_cfg.get("parallel_agents", 4)), len(agents)))
    agent_roster = [
        {
            "persona": str(a.get("persona", "")).strip(),
            "directive": str(a.get("directive", "")).strip()[:120],
            "role": str(a.get("role", "primary")).strip(),
        }
        for a in agents
    ]
    _progress(
        "research_pool_started",
        {
            "agents_total": len(agents),
            "agents": agent_roster,
            "workers": worker_count,
            "project": project_slug,
            "topic_type": resolved_type,
            "analysis_profile": profile_name,
        },
    )

    if _is_cancelled():
        summary_path = ""
        if question.strip():
            store = ProjectStore(repo_root)
            summary_name = store.timestamped_name("research_summary")
            summary_md = (
                "# Research Synthesis (Cancelled)\n\n"
                f"Question: {question}\n\n"
                "Request was cancelled before worker execution.\n"
            )
            summary_path = str(store.write_project_file(project_slug, "research_summaries", summary_name, summary_md))
        cancel_summary = (
            "Request cancelled before Foraging worker execution started.\n"
            + (f"Summary written to:\n{summary_path}" if summary_path else "No summary file was written.")
        )
        return {
            "message": "Research cancelled before execution.",
            "summary_path": summary_path,
            "web_context_used": bool(web_context.strip()),
            "reliability": {"agents_total": len(agents), "good": 0, "weak": 0, "failed": 0},
            "canceled": True,
            "cancel_summary": cancel_summary,
        }

    findings: list[dict[str, Any]] = []
    canceled = False
    executor = ThreadPoolExecutor(max_workers=worker_count)
    pending: set[Any] = set()
    future_agent: dict[Any, str] = {}
    # Sort agents by model name so same-model agents run consecutively.
    # This keeps each model warm in VRAM across back-to-back calls,
    # reducing Ollama load/evict churn within the pool.
    queue = sorted(agents, key=lambda a: str(a.get("model", "")))
    try:
        while queue or pending:
            if _is_cancelled():
                canceled = True
                _progress(
                    "research_cancel_requested",
                    {"completed": len(findings), "total": len(agents)},
                )
                break
            if _is_paused():
                _progress(
                    "foraging_paused",
                    {"completed": len(findings), "total": len(agents), "active_workers": len(pending)},
                )
                time.sleep(0.5)
                continue

            desired_workers = 1 if _should_yield() else worker_count
            while queue and len(pending) < desired_workers:
                agent_cfg = queue.pop(0)
                future = executor.submit(
                    _run_multipass_agent,
                    client,
                    model_cfg,
                    agent_cfg,
                    question,
                    learned_guidance,
                    web_context,
                    source_evidence,
                    prior_messages,
                    cancel_checker,
                    pause_checker,
                    allowed_personas,
                )
                pending.add(future)
                persona = str(agent_cfg.get("persona", "research_agent")).strip() or "research_agent"
                future_agent[future] = persona
                _progress(
                    "research_agent_started",
                    {
                        "agent": persona,
                        "directive": str(agent_cfg.get("directive", "")).strip()[:120],
                        "role": str(agent_cfg.get("role", "primary")).strip(),
                        "model": str(agent_cfg.get("model", "")).strip(),
                        "completed": len(findings),
                        "total": len(agents),
                        "active_workers": len(pending),
                        "yield_mode": bool(_should_yield()),
                    },
                )

            if not pending:
                time.sleep(0.15)
                continue

            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                persona = future_agent.pop(future, "research_agent")
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    result = {
                        "agent": persona,
                        "model": "",
                        "requested_model": "",
                        "finding": f"Model call failed for {persona}: {exc}",
                    }
                findings.append(result)
                agent_name = str(result.get("agent", "")).strip()
                if agent_name and agent_name not in visited_agents_per_leaf["root"]:
                    visited_agents_per_leaf["root"].append(agent_name)
                _finding_text = str(result.get("finding", "")).strip()
                _finding_failed = _is_failure_text(_finding_text)
                if _finding_failed:
                    _confidence = 0
                elif isinstance(result.get("self_score_confidence"), (int, float)):
                    _confidence = max(1, min(5, int(round(float(result.get("self_score_confidence", 0.0)) * 5))))
                else:
                    _confidence = _self_check(client, model_cfg, question, _finding_text)
                result["confidence"] = _confidence
                _progress(
                    "research_agent_completed",
                    {
                        "completed": len(findings),
                        "total": len(agents),
                        "agent": str(result.get("agent", "")),
                        "role": str(result.get("role", "primary")),
                        "failed": _finding_failed,
                        "finding_preview": _finding_text[:400] if not _finding_failed else "",
                        "confidence": _confidence,
                    },
                )

                handoff = result.get("handoff", {}) if isinstance(result.get("handoff", {}), dict) else {}
                target_persona = str(handoff.get("to", "")).strip()
                if target_persona:
                    loop_rejected = target_persona in visited_agents_per_leaf["root"]
                    cap_hit = handoff_count >= max_handoffs
                    honored = False
                    outcome_quality = 0.0
                    handoff_result: dict[str, Any] | None = None
                    if not loop_rejected and not cap_hit and target_persona in agent_cfg_by_persona:
                        handoff_cfg = dict(agent_cfg_by_persona[target_persona])
                        handoff_cfg["directive"] = (
                            f"{str(handoff_cfg.get('directive', '')).strip()}\n"
                            f"Handoff context from {agent_name}: {str(handoff.get('reason', '')).strip()[:220]}"
                        ).strip()
                        handoff_result = _run_multipass_agent(
                            client,
                            model_cfg,
                            handoff_cfg,
                            question,
                            learned_guidance,
                            web_context,
                            source_evidence,
                            prior_messages,
                            cancel_checker,
                            pause_checker,
                            allowed_personas,
                        )
                        handoff_text = str(handoff_result.get("finding", "")).strip()
                        if _is_failure_text(handoff_text):
                            handoff_result["confidence"] = 0
                        elif isinstance(handoff_result.get("self_score_confidence"), (int, float)):
                            handoff_result["confidence"] = max(
                                1, min(5, int(round(float(handoff_result.get("self_score_confidence", 0.0)) * 5)))
                            )
                        else:
                            handoff_result["confidence"] = _self_check(
                                client,
                                model_cfg,
                                question,
                                handoff_text,
                            )
                        handoff_result["handoff_from"] = agent_name
                        handoff_result["handoff_honored"] = True
                        findings.append(handoff_result)
                        handoff_count += 1
                        honored = True
                        outcome_quality = _outcome_quality(client, question, handoff_text)
                        target_name = str(handoff_result.get("agent", "")).strip()
                        if target_name and target_name not in visited_agents_per_leaf["root"]:
                            visited_agents_per_leaf["root"].append(target_name)
                        _progress(
                            "research_agent_completed",
                            {
                                "completed": len(findings),
                                "total": len(agents),
                                "agent": target_name,
                                "role": str(handoff_result.get("role", "primary")),
                                "failed": _is_failure_text(handoff_text),
                                "finding_preview": handoff_text[:400] if not _is_failure_text(handoff_text) else "",
                                "confidence": handoff_result.get("confidence", 0),
                                "handoff_from": agent_name,
                            },
                        )
                    elif loop_rejected:
                        _progress(
                            "handoff_loop_detected",
                            {
                                "leaf_id": "root",
                                "from_agent": agent_name,
                                "to_agent": target_persona,
                            },
                        )
                    elif cap_hit:
                        _progress(
                            "handoff_cap_hit",
                            {
                                "leaf_id": "root",
                                "from_agent": agent_name,
                                "to_agent": target_persona,
                                "max_handoffs": max_handoffs,
                            },
                        )

                    telemetry_emit(
                        repo_root,
                        "agent_handoffs.jsonl",
                        {
                            "leaf_id": "root",
                            "leaf_question": question[:400],
                            "from_agent": agent_name,
                            "to_agent": target_persona,
                            "reason": str(handoff.get("reason", "")).strip()[:240],
                            "confidence": float(handoff.get("confidence", 0.0) or 0.0),
                            "honored": honored,
                            "loop_rejected": loop_rejected,
                            "cap_hit": cap_hit,
                            "outcome_quality": round(float(outcome_quality), 4),
                        },
                        retention_days=30,
                    )
    finally:
        if canceled:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True, cancel_futures=False)

    pre_reliability = _reliability_summary(findings)
    if not canceled and pre_reliability.get("failed", 0) > 0:
        findings = _recover_failed_findings(
            client=client,
            findings=findings,
            model_cfg=model_cfg,
            question=question,
            learned_guidance=learned_guidance,
            web_context=web_context,
            source_evidence=source_evidence,
            prior_messages=prior_messages,
            cancel_checker=cancel_checker,
            pause_checker=pause_checker,
        )
    reliability = _reliability_summary(findings)
    findings = _audit_evidence_labels(findings)
    _progress(
        "numeric_validation_started",
        {"findings_collected": len(findings)},
    )
    try:
        findings, numeric_validation = validate_numeric_claims(
            findings,
            source_evidence=source_evidence,
            client=client,
            repo_root=repo_root,
        )
    except Exception as exc:
        LOGGER.warning("numeric validation failed: %s", exc)
        numeric_validation = {"numeric_claims_seen": 0, "numeric_claims_flagged": 0}
    _progress(
        "numeric_validation_completed",
        {
            "numeric_claims_seen": int(numeric_validation.get("numeric_claims_seen", 0) or 0),
            "numeric_claims_flagged": int(numeric_validation.get("numeric_claims_flagged", 0) or 0),
        },
    )
    for item in findings:
        if not isinstance(item, dict):
            continue
        item["source_tier_counts"] = _tier_breakdown_for_finding(
            str(item.get("finding", "")),
            source_tier_map,
        )
    retrieved_chunks = build_retrieved_chunks(findings)

    store = ProjectStore(repo_root)

    raw_name = store.timestamped_name("research_raw")
    raw_sections: list[str] = []
    for item in findings:
        persona = str(item.get("agent", "")).strip() or "research_agent"
        used_model = str(item.get("model", "")).strip()
        requested = str(item.get("requested_model", "")).strip()
        if used_model and requested and used_model != requested:
            title = f"## {persona} (model: {used_model}; requested: {requested})"
        elif used_model:
            title = f"## {persona} (model: {used_model})"
        else:
            title = f"## {persona}"
        raw_sections.append(f"{title}\n{item.get('finding', '')}")
    raw_body = "\n\n".join(raw_sections)
    raw_path = store.write_project_file(project_slug, "research_raw", raw_name, f"# Raw Research Notes\n\n{raw_body}\n")
    _progress(
        "research_raw_written",
        {"raw_path": str(raw_path), "findings_collected": len(findings), "canceled": canceled},
    )

    if canceled:
        summary_name = store.timestamped_name("research_summary")
        _partial_lines = ["## Partial Findings"]
        for _item in findings:
            _agent = str(_item.get("agent", "agent")).strip()
            _text = str(_item.get("finding", "")).strip()[:500]
            if _text:
                _partial_lines.append(f"\n**{_agent}:** {_text}")
        partial = "\n".join(_partial_lines)
        cancel_md = (
            "# Research Synthesis (Cancelled)\n\n"
            f"Question: {question}\n\n"
            "The request was cancelled by the user. This is a partial synthesis from completed workers only.\n\n"
            f"Completed worker findings: {len(findings)} / {len(agents)}\n\n"
            f"{partial}\n"
        )
        summary_path = store.write_project_file(project_slug, "research_summaries", summary_name, cancel_md)
        cancel_summary = (
            "Request cancelled during Foraging.\n"
            f"- completed_workers: {len(findings)} / {len(agents)}\n"
            f"- partial_raw_notes: {raw_path}\n"
            f"- partial_summary: {summary_path}"
        )
        bus.emit(
            "research_pool",
            "cancelled",
            {
                "project": project_slug,
                "raw_path": str(raw_path),
                "summary_path": str(summary_path),
                "completed_workers": len(findings),
                "agents_total": len(agents),
            },
        )
        _progress(
            "research_cancelled",
            {
                "summary_path": str(summary_path),
                "raw_path": str(raw_path),
                "completed_workers": len(findings),
                "agents_total": len(agents),
            },
        )
        return {
            "message": "Research cancelled and partial synthesis written for review.",
            "summary_path": str(summary_path),
            "raw_path": str(raw_path),
            "web_context_used": bool(web_context.strip()),
            "reliability": reliability,
            "numeric_validation": numeric_validation,
            "canceled": True,
            "cancel_summary": cancel_summary,
            "retrieved_chunks": retrieved_chunks,
            "visited_agents_per_leaf": visited_agents_per_leaf,
        }

    _synthesis_lane = lane_model_config(repo_root, "synthesis") or {}
    synth_cfg = dict(_synthesis_lane or orchestrator_cfg or {})
    synth_cfg.setdefault("synthesis_timeout_sec", int(_synthesis_lane.get("timeout_sec", int(model_cfg.get("timeout_sec", 0)))))
    synth_cfg.setdefault("synthesis_retry_attempts", int(model_cfg.get("retry_attempts", 6)))
    synth_cfg.setdefault("synthesis_retry_backoff_sec", float(model_cfg.get("retry_backoff_sec", 1.5)))
    synth_cfg.setdefault("synthesis_validation_cycles", int(model_cfg.get("validation_cycles", 3)))
    fb = list(model_cfg.get("fallback_models", [])) if isinstance(model_cfg.get("fallback_models", []), list) else []
    main_model = str(model_cfg.get("model", "")).strip()
    if main_model:
        fb.append(main_model)
    synth_cfg.setdefault("synthesis_fallback_models", fb)

    # Underground topics: force abliterated model for synthesis — no filtered models in the pipeline.
    if str(topic_type).strip().lower() == "underground":
        synth_cfg["model"] = "huihui_ai/qwen3-abliterated:8b-Q4_K_M"
        synth_cfg["synthesis_fallback_models"] = ["huihui_ai/qwen3-abliterated:8b-Q4_K_M"]

    summary_name = store.timestamped_name("research_summary")
    prior_open_questions = _load_prior_open_questions(repo_root, project_slug)
    conflict_report = _cross_agent_conflict_report(findings)
    conflict_count = sum(
        1 for line in str(conflict_report or "").splitlines() if str(line).strip().startswith("- ")
    )
    _confidence_sources = [
        {
            "source_domain": str(row.get("domain", "")).strip(),
            "source_tier": str(row.get("source_tier", "")).strip().lower(),
            "freshness_score": float(row.get("freshness_score", 0.0) or 0.0),
        }
        for row in source_evidence
        if isinstance(row, dict)
    ]
    confidence_eval = evaluate_answer_confidence(
        sources=_confidence_sources,
        conflict_summary={"conflict_count": conflict_count},
        question=question,
    )
    tier1_count = int(confidence_eval.get("tier1_count", 0) or 0)
    confidence_mode = str(confidence_eval.get("mode", "medium")).strip().lower()
    importance = "high" if (confidence_mode == "high" or tier1_count >= 3 or conflict_count >= 2) else "medium"

    dcr_cfg = model_cfg.get("draft_critique_revise", {}) if isinstance(model_cfg.get("draft_critique_revise", {}), dict) else {}
    critic_lane = str(dcr_cfg.get("critic_lane", "synthesis")).strip() or "synthesis"
    dcr_enabled = bool(dcr_cfg.get("enabled", False))

    def _synth_cfg_for_tier(tier_cfg: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(synth_cfg)
        if isinstance(tier_cfg, dict):
            for key, value in tier_cfg.items():
                merged[key] = value
        if str(merged.get("model", "")).strip():
            merged["synthesis_model"] = str(merged.get("model", "")).strip()
        if "fallback_models" in merged and "synthesis_fallback_models" not in merged:
            merged["synthesis_fallback_models"] = list(merged.get("fallback_models", []))
        return merged

    _progress(
        "synthesizing",
        {
            "model": str(synth_cfg.get("model", "")).strip(),
            "findings_collected": len(findings),
        },
    )
    warning_banner = ""
    if dcr_enabled:
        _progress(
            "skeptic_pass_started",
            {
                "phase": "loop_controller",
                "model": str(synth_cfg.get("model", "")).strip(),
                "note": "Running draft->critique->revise loop.",
            },
        )
        try:
            loop_result = run_draft_critique_revise(
                repo_root=repo_root,
                lane_key=critic_lane,
                draft_fn=lambda tier_cfg: synthesize(
                    question,
                    findings,
                    client=client,
                    model_cfg=_synth_cfg_for_tier(tier_cfg),
                    conflict_report=conflict_report,
                    prior_open_questions=prior_open_questions,
                    source_tier_map=source_tier_map,
                ),
                critique_fn=lambda draft_text, tier_cfg: run_skeptic_pass_with_severity(
                    question,
                    draft_text,
                    client=client,
                    model_cfg=_synth_cfg_for_tier(tier_cfg),
                    findings=findings,
                ),
                importance=importance,
                client=client,
                telemetry_ctx={
                    "task_class": "research_synthesis",
                    "project_slug": project_slug,
                    "topic_type": resolved_type,
                },
                cancel_checker=cancel_checker,
            )
            summary_md = str(loop_result.final_text or "").strip()
            critique_log = "\n\n".join(
                str(item).strip() for item in loop_result.critique_logs if str(item).strip()
            )
            warning_banner = str(loop_result.warning_banner or "").strip()
        except SynthesisUnavailableError as exc:
            LOGGER.error("Synthesis unavailable during research pool loop: %s", exc)
            bus.emit("research_pool", "synthesis_unavailable", {"project": project_slug, "reason": str(exc)})
            _progress(
                "synthesis_unavailable",
                {
                    "reason": str(exc),
                    "raw_path": str(raw_path),
                },
            )
            return {
                "message": (
                    "Research could not complete — the synthesis model was unavailable. "
                    f"Raw agent findings saved to {raw_path}. "
                    "Try again once the model is available, or ask me to rescue this raw file."
                ),
                "summary_path": "",
                "critique_path": "",
                "raw_path": str(raw_path),
                "web_context_used": bool(web_context.strip()),
                "reliability": reliability,
                "numeric_validation": numeric_validation,
                "synthesis_unavailable": True,
                "findings": findings,
                "retrieved_chunks": [],
                "visited_agents_per_leaf": visited_agents_per_leaf,
            }
        _progress(
            "skeptic_pass_completed",
            {
                "phase": "loop_controller",
                "critique_chars": len(str(critique_log or "").strip()),
                "note": "Loop-controller critique pass finished.",
            },
        )
    else:
        try:
            summary_md = synthesize(
                question,
                findings,
                client=client,
                model_cfg=synth_cfg,
                conflict_report=conflict_report,
                prior_open_questions=prior_open_questions,
                source_tier_map=source_tier_map,
            )
        except SynthesisUnavailableError as exc:
            LOGGER.error("Synthesis unavailable during research pool run: %s", exc)
            bus.emit("research_pool", "synthesis_unavailable", {"project": project_slug, "reason": str(exc)})
            _progress(
                "synthesis_unavailable",
                {
                    "reason": str(exc),
                    "raw_path": str(raw_path),
                },
            )
            return {
                "message": (
                    "Research could not complete — the synthesis model was unavailable. "
                    f"Raw agent findings saved to {raw_path}. "
                    "Try again once the model is available, or ask me to rescue this raw file."
                ),
                "summary_path": "",
                "critique_path": "",
                "raw_path": str(raw_path),
                "web_context_used": bool(web_context.strip()),
                "reliability": reliability,
                "numeric_validation": numeric_validation,
                "synthesis_unavailable": True,
                "findings": findings,
                "retrieved_chunks": [],
                "visited_agents_per_leaf": visited_agents_per_leaf,
            }

        _progress(
            "skeptic_pass_started",
            {
                "phase": "primary",
                "model": str(synth_cfg.get("model", "")).strip(),
                "note": "Running critique pass on synthesis.",
            },
        )
        summary_md, critique_log = run_skeptic_pass(
            question,
            summary_md,
            client=client,
            model_cfg=synth_cfg,
            findings=findings,
        )
        _progress(
            "skeptic_pass_completed",
            {
                "phase": "primary",
                "critique_chars": len(str(critique_log or "").strip()),
                "note": "Critique pass finished.",
            },
        )

    translation_meta: dict[str, Any] = {"stages": []}
    _progress(
        "translation_chain_started",
        {"stages": 4},
    )
    try:
        _synth_lane_cfg = lane_model_config(repo_root, "synthesis") or {}
        _premium_cfg = _synth_lane_cfg.get("tier_premium", {}) if isinstance(_synth_lane_cfg.get("tier_premium", {}), dict) else {}
        _premium_model = str(_premium_cfg.get("model", "")).strip()
        _use_premium_tech = bool(_premium_model and _premium_model == str(synth_cfg.get("model", "")).strip())
        translation_meta = run_translation_chain(
            question=question,
            synthesis_summary=summary_md,
            project_context=project_context,
            client=client,
            repo_root=repo_root,
            synthesis_cfg=synth_cfg,
            use_premium_tech_lead=_use_premium_tech,
        )
        summary_md = str(translation_meta.get("summary", "") or summary_md).strip()
    except Exception as exc:
        LOGGER.warning("translation_chain failed: %s", exc)
    _progress(
        "translation_chain_completed",
        {
            "stages": len(translation_meta.get("stages", []) if isinstance(translation_meta, dict) else []),
        },
    )

    summary_md = ensure_project_specificity(summary_md, project_context)
    analogy_urls = [
        str(row.get("url", "")).strip()
        for row in source_evidence
        if isinstance(row, dict) and str(row.get("source_tier", "")).strip().lower() == "tier4" and str(row.get("url", "")).strip()
    ]
    summary_md = enforce_final_summary_contract(
        summary_md=summary_md,
        findings=findings,
        source_quality_warning=warning_banner,
        analogy_sources=analogy_urls,
    )

    recycled_questions = _count_recycled_open_questions(summary_md, prior_open_questions)
    quality_suffix = ""
    if recycled_questions > 0:
        quality_suffix = f" | recycled prior questions: {recycled_questions}"
    summary_md = f"{summary_md}\n\n---\n\n{_build_source_quality_footer(findings, suffix=quality_suffix)}"
    if warning_banner:
        summary_md = f"**Warning:** {warning_banner}\n\n{summary_md}"

    summary_path = store.write_project_file(project_slug, "research_summaries", summary_name, summary_md)
    if not critique_log.strip():
        critique_log = "_Skeptic pass produced no output for this run._"
    critique_name = f"{summary_name}.critique.md"
    critique_path = str(
        store.write_project_file(project_slug, "research_summaries", critique_name, critique_log)
    )
    _progress(
        "research_summary_written",
        {
            "summary_path": str(summary_path),
            "critique_path": critique_path,
            "findings_collected": len(findings),
            "recycled_open_questions": recycled_questions,
        },
    )

    # Release Ollama-hosted models from VRAM now that the full pipeline is done.
    # Models routed through llama.cpp are managed by that server process and skipped.
    _release_models = sorted({
        str(f.get("model", "")).strip()
        for f in findings
        if str(f.get("model", "")).strip()
    } | {str(model_cfg.get("model", "")).strip(), str(synth_cfg.get("model", "")).strip()})
    client.release_models([m for m in _release_models if m])

    bus.emit(
        "research_pool",
        "completed",
        {
            "project": project_slug,
            "raw_path": str(raw_path),
            "summary_path": str(summary_path),
            "critique_path": critique_path,
            "model": model_cfg.get("model", ""),
            "workers": worker_count,
            "agents_total": len(agents),
            "models_used": sorted({str(x.get("model", "")).strip() for x in findings if str(x.get("model", "")).strip()}),
            "web_context_used": bool(web_context.strip()),
            "reliability": reliability,
            "analysis_profile": profile_name,
            "topic_type": resolved_type,
            "recycled_open_questions": recycled_questions,
            "warning_banner": warning_banner,
            "numeric_validation": numeric_validation,
            "translation_stages": len(translation_meta.get("stages", []) if isinstance(translation_meta, dict) else []),
        },
    )

    return {
        "message": (
            "Foraging council completed a synthesis for orchestrator review. "
            f"Reliability: good={reliability.get('good', 0)}, "
            f"weak={reliability.get('weak', 0)}, failed={reliability.get('failed', 0)}."
        ),
        "summary_path": str(summary_path),
        "critique_path": critique_path,
        "raw_path": str(raw_path),
        "web_context_used": bool(web_context.strip()),
        "reliability": reliability,
        "numeric_validation": numeric_validation,
        "analysis_profile": profile_name,
        "topic_type": resolved_type,
        "recycled_open_questions": recycled_questions,
        "warning_banner": warning_banner,
        "translation_chain": translation_meta,
        "findings": findings,
        "retrieved_chunks": retrieved_chunks,
        "visited_agents_per_leaf": visited_agents_per_leaf,
    }
