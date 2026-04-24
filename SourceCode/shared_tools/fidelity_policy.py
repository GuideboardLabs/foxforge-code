"""Fidelity policy — single source of truth for factual accuracy constraints across all writing pools.

Resolves a FidelityLevel from (type_id, topic_type) and generates the prompt snippets each
pool injects into its writer and critic system/user prompts.

Three levels:
  STRICT   — Never invent specifics. Thin research → write generally. [I]/[S] must be framed.
  GROUNDED — Ground claims in research. Gaps allowed with explicit hedging. Never as fact.
  CREATIVE — Free invention of characters/plot/dialogue. Real-world anchors must match research.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable


class FidelityLevel(Enum):
    STRICT = "strict"
    GROUNDED = "grounded"
    CREATIVE = "creative"


# ---------------------------------------------------------------------------
# Type → baseline fidelity level
# ---------------------------------------------------------------------------

_TYPE_BASELINE: dict[str, FidelityLevel] = {
    # STRICT: factual output types where invented specifics cause real harm
    "essay_long":    FidelityLevel.STRICT,
    "essay_short":   FidelityLevel.STRICT,
    "video_script":  FidelityLevel.STRICT,
    "guide":         FidelityLevel.STRICT,
    "tutorial":      FidelityLevel.STRICT,
    "newsletter":    FidelityLevel.STRICT,
    "press_release": FidelityLevel.STRICT,
    "medical":       FidelityLevel.STRICT,
    "finance":       FidelityLevel.STRICT,
    "sports":        FidelityLevel.STRICT,
    "history":       FidelityLevel.STRICT,
    # legacy document target strings
    "essay":         FidelityLevel.STRICT,
    "report":        FidelityLevel.STRICT,
    "brief":         FidelityLevel.STRICT,
    "analysis":      FidelityLevel.STRICT,
    "explainer":     FidelityLevel.STRICT,
    "lit_review":    FidelityLevel.STRICT,
    # GROUNDED: personal or general writing where some reconstruction is normal
    "blog":           FidelityLevel.GROUNDED,
    "social_post":    FidelityLevel.GROUNDED,
    "email":          FidelityLevel.GROUNDED,
    "memoir_chapter": FidelityLevel.GROUNDED,
    "memoir":         FidelityLevel.GROUNDED,
    "book_chapter":   FidelityLevel.GROUNDED,
    "game_design_doc": FidelityLevel.GROUNDED,
    # CREATIVE: fictional output where invention is the job
    "novel_chapter": FidelityLevel.CREATIVE,
    "novel":         FidelityLevel.CREATIVE,
    "screenplay":    FidelityLevel.CREATIVE,
}

# Topics that force STRICT regardless of type baseline (verifiable ground truth)
_STRICT_TOPICS: frozenset[str] = frozenset({
    "history", "science", "technical", "math", "medical", "finance",
    "law", "current_events", "sports", "combat_sports", "sports_event",
})

# Types exempt from topic override — always stay at their baseline
# (memoir is personal reconstruction; imposing STRICT defeats the form)
_TOPIC_OVERRIDE_EXEMPT: frozenset[str] = frozenset({
    "memoir_chapter", "memoir",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fidelity_for(type_id: str, topic_type: str = "general") -> FidelityLevel:
    """Resolve the effective fidelity level for a given type and topic.

    Topic override rules:
    - Exempt types (memoir): always return baseline, no topic upgrade.
    - CREATIVE types with a strict topic: upgrade to GROUNDED (real anchors must be accurate,
      but invented characters/plot are still allowed).
    - All other types with a strict topic: upgrade to STRICT.
    """
    tid = str(type_id or "").strip().lower()
    topic = str(topic_type or "general").strip().lower()
    base = _TYPE_BASELINE.get(tid, FidelityLevel.GROUNDED)
    if tid in _TOPIC_OVERRIDE_EXEMPT:
        return base
    if topic in _STRICT_TOPICS:
        # Creative types upgrade to GROUNDED, not STRICT — still allows invented plot/characters
        if base == FidelityLevel.CREATIVE:
            return FidelityLevel.GROUNDED
        return FidelityLevel.STRICT
    return base


def writer_constraint_block(level: FidelityLevel) -> str:
    """System prompt snippet injected into every section/scene writer."""
    if level == FidelityLevel.STRICT:
        return (
            "RESEARCH-ONLY FACTS: Your training knowledge about specific events, people, names, dates, "
            "and incidents is unreliable and may be factually wrong — do not use it as a source. "
            "The research context provided below is the only authoritative source of specific facts. "
            "Do not include any proper noun (name of a person, place, or organization), date, statistic, "
            "or event detail unless it explicitly appears in the research context. "
            "If the research is silent on a specific detail, omit it or write at the level of generality "
            "the research supports. Vague but accurate beats specific but wrong."
        )
    if level == FidelityLevel.GROUNDED:
        return (
            "FACTUAL CLAIMS: Your training knowledge about specific events may be incomplete or wrong. "
            "Ground all specific claims in the research context. "
            "If a detail is not in the research, fill the gap with explicit hedging "
            "('may', 'likely', 'it is possible that') — never state an uncertain detail as established fact. "
            "Vague but accurate beats specific but invented."
        )
    # CREATIVE
    return (
        "CREATIVE LATITUDE: You may freely invent characters, dialogue, settings, and plot. "
        "However, any real-world facts referenced — historical dates, real persons, real events, "
        "real places — must match the research context exactly. Do not invent real-world facts even in fiction."
    )


def evidence_key_block(level: FidelityLevel, has_raw_notes: bool) -> str:
    """User prompt snippet explaining [E]/[I]/[S] markers, prepended to research context."""
    if not has_raw_notes:
        return ""
    if level == FidelityLevel.STRICT:
        return (
            "\nEvidence markers in research: [E] = verified (assert confidently), "
            "[I] = inferred (frame as 'this suggests\u2026'), "
            "[S] = speculative (frame as 'one possibility is\u2026'). "
            "Never present [I] or [S] findings as established facts.\n"
        )
    if level == FidelityLevel.GROUNDED:
        return (
            "\nEvidence markers: [E] = verified, [I] = inferred, [S] = speculative. "
            "Distinguish [I] and [S] from established fact.\n"
        )
    return ""  # CREATIVE — no [E]/[I]/[S] constraint on invented content


def thin_research_warning(level: FidelityLevel, research_chars: int) -> str:
    """System prompt snippet injected into planners/outliners when research is sparse.

    Fires only at STRICT/GROUNDED levels and only when research_chars is below threshold.
    Prevents the model from filling thin-research gaps with training-data facts.
    """
    if level == FidelityLevel.CREATIVE or research_chars >= 300:
        return ""
    severity = "empty" if research_chars == 0 else "very sparse"
    return (
        f"\n⚠ THIN RESEARCH WARNING: The research context is {severity} ({research_chars} chars). "
        "You have no verified facts to draw from, and your training knowledge about specific "
        "events, names, and dates for this topic may be wrong. "
        "Write only at a general, conceptual level. "
        "Do NOT include specific names of people, exact dates, event details, or statistics — "
        "you have no sourced basis for them. "
        "Acknowledge limited documentation explicitly rather than filling gaps with guesses.\n"
    )


def critic_fabrication_block(
    level: FidelityLevel,
    raw_notes: str,
    trim_fn: Callable[[str, int], str],
    research_context: str = "",
) -> str:
    """User prompt snippet appended to critic input for cross-referencing claims.

    Includes both raw_notes (with [E]/[I]/[S] labels) and the summarized research_context
    so the critic has the fullest possible reference even when raw notes are absent.
    """
    if level == FidelityLevel.CREATIVE:
        return ""
    has_raw = bool(raw_notes.strip())
    has_research = bool(research_context.strip())
    if not has_raw and not has_research:
        return ""

    parts: list[str] = ["\n\n"]
    if has_raw:
        parts.append(
            f"Raw research notes ([E]=evidence, [I]=inferred, [S]=speculative):\n"
            f"{trim_fn(raw_notes, 5000)}\n\n"
        )
    if has_research:
        parts.append(
            f"Summarized research context (use as authoritative reference):\n"
            f"{trim_fn(research_context, 3000)}\n\n"
        )

    if level == FidelityLevel.STRICT:
        parts.append(
            "FABRICATION CHECK: Cross-reference every named person, date, statistic, and specific "
            "event in the draft against the research above. "
            "Flag any claim that cannot be traced to the provided research. "
            "The model's training knowledge about specific events is unreliable — "
            "only claims explicitly supported by the research above are acceptable. "
            "Claims backed only by [I] or [S] that are written as established facts must also be flagged."
        )
    else:  # GROUNDED
        parts.append(
            "GROUNDING CHECK: Flag any specific claim (name, date, statistic, event) stated as fact "
            "that cannot be traced to the research above. "
            "Pure speculation written as established fact must be flagged."
        )

    return "".join(parts)


def planner_grounding_rule(level: FidelityLevel) -> str:
    """System prompt snippet added to planner/outliner prompts."""
    if level == FidelityLevel.STRICT:
        return (
            "GROUNDING RULE: Each thesis must reflect actual content from the research context. "
            "If the research does not support a thesis, write 'THIN COVERAGE' instead of inventing support. "
            "A section with thin coverage should be scoped down or merged — do not pad with speculation."
        )
    if level == FidelityLevel.GROUNDED:
        return (
            "GROUNDING RULE: Each thesis should be grounded in the research where possible. "
            "If coverage is thin, note it briefly rather than padding with speculation."
        )
    return ""  # CREATIVE — no grounding rule for planners
