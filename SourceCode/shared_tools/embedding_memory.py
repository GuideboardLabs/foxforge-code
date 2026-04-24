from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from shared_tools.config_ini import load_config

_TOKEN_RE = re.compile(r'[a-z0-9]{3,}')
_STOP = {'the','and','for','with','from','that','this','have','into','what','when','where','who','your','about'}

_EMBED_MODEL = "qwen3-embedding:4b"
_VECTOR_THRESHOLD = 0.45
_BOW_THRESHOLD = 0.05


def _tokens(text: str) -> Counter[str]:
    words = [w for w in _TOKEN_RE.findall(str(text or '').lower()) if w not in _STOP]
    return Counter(words)


def _bow_cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _vec_cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _split_sections(text: str) -> list[str]:
    """Split markdown text on ## / ### boundaries. Each chunk capped at 2000 chars."""
    parts = re.split(r'(?m)^(?=#{2,3} )', text)
    sections: list[str] = []
    for part in parts:
        part = part.strip()
        if part:
            sections.append(part[:2000])
    return sections or [text[:2000]]


class EmbeddingMemory:
    """Local retrieval helper. Uses qwen3-embedding:4b vectors when available,
    silently falls back to bag-of-words cosine if Ollama/model is unavailable.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)

    def _embed_model(self) -> str:
        try:
            cfg = load_config(self.repo_root)
            return cfg.get_model("embeddings")
        except Exception:
            return _EMBED_MODEL

    def _summary_files(self, project: str) -> list[Path]:
        root = self.repo_root / 'Projects' / (project.strip() or 'general') / 'research_summaries'
        if not root.exists():
            return []
        return sorted(root.glob('*.md'), reverse=True)[:20]

    def _cache_path(self, project: str) -> Path:
        cache_dir = self.repo_root / 'Runtime' / 'memory' / 'embed_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{project.strip() or 'general'}.json"

    def _load_cache(self, project: str) -> dict[str, Any]:
        path = self._cache_path(project)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def _save_cache(self, project: str, cache: dict[str, Any]) -> None:
        try:
            self._cache_path(project).write_text(
                json.dumps(cache, ensure_ascii=False), encoding='utf-8'
            )
        except Exception:
            pass

    def _get_client(self) -> Any:
        try:
            from shared_tools.ollama_client import OllamaClient
            return OllamaClient()
        except Exception:
            return None

    def retrieve(self, project: str, query: str, *, limit: int = 3) -> list[dict[str, Any]]:
        files = self._summary_files(project)
        if not files:
            return []

        # Try vector path first
        try:
            client = self._get_client()
            if client is not None:
                return self._retrieve_vector(project, query, files, client, limit=limit)
        except Exception:
            pass

        # Fall back to bag-of-words
        return self._retrieve_bow(query, files, limit=limit)

    def _retrieve_vector(
        self,
        project: str,
        query: str,
        files: list[Path],
        client: Any,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        embed_model = self._embed_model()
        query_vec = client.embed(embed_model, query[:2000])
        cache = self._load_cache(project)
        cache_dirty = False
        # best section hit per file: path → {score, preview}
        best_per_file: dict[str, dict[str, Any]] = {}

        for path in files:
            try:
                mtime = path.stat().st_mtime
                path_str = str(path)
                text = path.read_text(encoding='utf-8', errors='ignore')
                sections = _split_sections(text)

                for i, section in enumerate(sections):
                    key = f"{path_str}#s{i}"
                    entry = cache.get(key)
                    if entry and abs(float(entry.get("mtime", 0)) - mtime) < 1.0:
                        vec = entry["vector"]
                    else:
                        vec = client.embed(embed_model, section)
                        cache[key] = {"mtime": mtime, "vector": vec}
                        cache_dirty = True

                    score = _vec_cosine(query_vec, vec)
                    if score < _VECTOR_THRESHOLD:
                        continue
                    prev = best_per_file.get(path_str)
                    if prev is None or score > prev['score']:
                        best_per_file[path_str] = {
                            'path': path_str,
                            'score': round(score, 3),
                            'preview': section[:700].strip(),
                        }
            except Exception:
                continue

        if cache_dirty:
            self._save_cache(project, cache)

        items = sorted(best_per_file.values(), key=lambda x: x['score'], reverse=True)
        return items[:limit]

    def _retrieve_bow(self, query: str, files: list[Path], *, limit: int) -> list[dict[str, Any]]:
        qv = _tokens(query)
        items: list[dict[str, Any]] = []
        for path in files:
            try:
                text = path.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            score = _bow_cosine(qv, _tokens(text[:8000]))
            if score <= _BOW_THRESHOLD:
                continue
            items.append({'path': str(path), 'score': round(score, 3), 'preview': text[:700].strip()})
        items.sort(key=lambda x: x['score'], reverse=True)
        return items[:limit]

    def context_text(self, project: str, query: str, *, limit: int = 2) -> str:
        rows = self.retrieve(project, query, limit=limit)
        if not rows:
            return ''
        lines = ['Relevant prior local summaries:']
        for row in rows:
            lines.append(f"- score={row['score']:.2f} file={row['path']}")
            lines.append(f"  preview: {row['preview']}")
        return '\n'.join(lines)
