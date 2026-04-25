from __future__ import annotations

import re
from urllib.parse import urlparse
from typing import Any

from shared_tools.embedding_memory import _vec_cosine


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _domain(url: str) -> str:
    try:
        host = str(urlparse(str(url or "")).hostname or "").lower()
    except Exception:
        host = ""
    return host.removeprefix("www.")


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _lexical_sim(a: str, b: str) -> float:
    ta = set(_TOKEN_RE.findall(_norm(a)))
    tb = set(_TOKEN_RE.findall(_norm(b)))
    if not ta or not tb:
        return 0.0
    return float(len(ta & tb) / max(1, len(ta | tb)))


def _embed(text: str, client: Any, *, model: str) -> list[float] | None:
    if client is None:
        return None
    try:
        vec = client.embed(model, text[:1600], timeout=20)
        if isinstance(vec, list) and vec:
            return [float(x) for x in vec]
    except Exception:
        return None
    return None


def _sim(a: str, b: str, *, client: Any, model: str, cache: dict[str, list[float] | None]) -> float:
    key_a = a[:1600]
    key_b = b[:1600]
    if key_a not in cache:
        cache[key_a] = _embed(key_a, client, model=model)
    if key_b not in cache:
        cache[key_b] = _embed(key_b, client, model=model)
    va = cache.get(key_a)
    vb = cache.get(key_b)
    if isinstance(va, list) and isinstance(vb, list) and va and vb:
        return float(_vec_cosine(va, vb))
    return _lexical_sim(a, b)


def _tier_weight(tier: str) -> float:
    key = str(tier or "").strip().lower()
    if key == "tier1":
        return 1.0
    if key == "tier2":
        return 0.78
    if key == "tier3":
        return 0.45
    if key == "tier4":
        return 0.22
    return 0.45


def rank_bucket_sources(
    *,
    persona_query: str,
    sources: list[dict[str, Any]],
    top_k: int = 8,
    client: Any | None = None,
    embedding_model: str = "qwen3-embedding:4b",
) -> list[dict[str, Any]]:
    """Rank a source bucket by tier, topical relevance, and intra-bucket diversity."""
    cleaned: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for row in sources:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or row.get("source_url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = str(row.get("title", "")).strip()
        snippet = str(row.get("snippet", "")).strip()
        text = f"{title}. {snippet}".strip()
        tier = str(row.get("source_tier") or row.get("tier") or "tier3").strip().lower() or "tier3"
        source_score = float(row.get("source_score", 0.0) or 0.0)
        cleaned.append(
            {
                **row,
                "url": url,
                "title": title,
                "snippet": snippet,
                "domain": _domain(url),
                "source_tier": tier,
                "_text": text,
                "_tier_weight": _tier_weight(tier),
                "_base_quality": max(0.05, source_score),
            }
        )

    if not cleaned:
        return []

    vec_cache: dict[str, list[float] | None] = {}
    for row in cleaned:
        rel = _sim(persona_query, str(row.get("_text", "")), client=client, model=embedding_model, cache=vec_cache)
        rel = max(0.0, min(1.0, rel))
        row["_relevance"] = rel
        row["_quality"] = row["_tier_weight"] * max(0.12, rel) * max(0.2, float(row.get("_base_quality", 0.0) or 0.0))

    # Highest first for greedy MMR-like selection.
    candidates = sorted(cleaned, key=lambda x: float(x.get("_quality", 0.0)), reverse=True)
    selected: list[dict[str, Any]] = []
    per_domain_count: dict[str, int] = {}

    while candidates and len(selected) < max(1, int(top_k)):
        best_idx = 0
        best_score = -1.0

        for idx, row in enumerate(candidates):
            text = str(row.get("_text", ""))
            redundancy = 0.0
            for prior in selected:
                redundancy = max(
                    redundancy,
                    _sim(text, str(prior.get("_text", "")), client=client, model=embedding_model, cache=vec_cache),
                )
            domain = str(row.get("domain", "")).strip()
            same_domain_penalty = 0.18 * per_domain_count.get(domain, 0)
            score = float(row.get("_quality", 0.0)) * (1.0 - (0.5 * redundancy)) * (1.0 - same_domain_penalty)
            if score > best_score:
                best_score = score
                best_idx = idx

        chosen = candidates.pop(best_idx)
        rel = float(chosen.get("_relevance", 0.0))
        if rel < 0.17:
            chosen["analogy"] = True
            chosen["source_tier"] = "tier4"
        else:
            chosen["analogy"] = False
        chosen["diversity_score"] = round(max(0.0, best_score), 4)
        chosen["topical_relevance"] = round(rel, 4)
        selected.append(chosen)
        domain = str(chosen.get("domain", "")).strip()
        if domain:
            per_domain_count[domain] = per_domain_count.get(domain, 0) + 1

    out: list[dict[str, Any]] = []
    for row in selected:
        cleaned_row = dict(row)
        for key in ["_text", "_tier_weight", "_base_quality", "_relevance", "_quality"]:
            cleaned_row.pop(key, None)
        out.append(cleaned_row)
    return out


__all__ = ["rank_bucket_sources"]
