from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared_tools.config_ini import load_config
from shared_tools.db import connect, row_to_dict
from shared_tools.document_ingestion import extract_text
from shared_tools.migrations import initialize_database
from shared_tools.ollama_client import OllamaClient
from shared_tools.topic_engine import TopicEngine

_VALID_SOURCE_KINDS = frozenset({"book", "review", "reference", "notes", "general"})
_PENDING_STATUSES = frozenset({"queued", "extracting", "matching_topic", "chunking", "embedding", "summarizing"})
_READY_STATUS = "ready"
_FAILED_STATUS = "failed"
_EMBED_MODEL = "qwen3-embedding:4b"
_VECTOR_THRESHOLD = 0.35
_MAX_TITLE_CHARS = 180
_CHUNK_TARGET_CHARS = 1500
_CHUNK_OVERLAP_CHARS = 200
_MAX_ITEM_LIMIT = 250
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _slugify(text: str, *, limit: int = 64) -> str:
    slug = re.sub(r"[^\w\s-]", "", str(text or "").strip().lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return slug[:limit] or "item"


def _tokens(text: str) -> Counter[str]:
    return Counter(_TOKEN_RE.findall(str(text or "").lower()))


def _bow_cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[token] * b[token] for token in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _vec_cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _safe_kind(value: str) -> str:
    key = str(value or "").strip().lower()
    return key if key in _VALID_SOURCE_KINDS else "general"


class LibraryService:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)
        initialize_database(self.repo_root)
        self._lock = threading.Lock()

    @property
    def items_root(self) -> Path:
        path = self.repo_root / "Runtime" / "library" / "items"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _connect(self):
        return connect(self.repo_root)

    def _embed_model(self) -> str:
        try:
            cfg = load_config(self.repo_root)
            return cfg.get_model("embeddings")
        except Exception:
            return _EMBED_MODEL

    def _orchestrator_model(self) -> str:
        try:
            cfg = load_config(self.repo_root)
            return cfg.get_model("orchestrator_reasoning")
        except Exception:
            return ""

    def _topic_engine(self) -> TopicEngine:
        return TopicEngine(self.repo_root)

    def _item_dir(self, item_id: str) -> Path:
        path = self.items_root / str(item_id).strip()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _normalize_topic_id(self, topic_id: str | None) -> str:
        key = str(topic_id or "").strip()
        if key.lower() == "general":
            return ""
        return key

    def _normalize_project_slug(self, project_slug: str | None) -> str:
        key = str(project_slug or "").strip().lower()
        if not key or key == "general":
            return ""
        return _slugify(key, limit=80)

    def _load_topics(self) -> list[dict[str, Any]]:
        path = self.repo_root / "Runtime" / "projects" / "projects.json"
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []

    def _all_projects(self) -> list[str]:
        projects_root = self.repo_root / "Projects"
        if not projects_root.exists():
            return []
        rows = sorted(
            {
                entry.name.strip().lower()
                for entry in projects_root.iterdir()
                if entry.is_dir() and entry.name.strip()
            }
        )
        return [row for row in rows if row and row != "general"]

    def _hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _title_from_name(self, source_name: str) -> str:
        name = Path(str(source_name or "").strip()).stem.replace("_", " ").replace("-", " ").strip()
        return _compact(name).title()[:_MAX_TITLE_CHARS] or "Library Item"

    def _clean_row(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        data = dict(row)
        data["project_slug"] = self._normalize_project_slug(data.get("project_slug"))
        data["topic_id"] = self._normalize_topic_id(data.get("topic_id"))
        data["source_kind"] = _safe_kind(str(data.get("source_kind", "")))
        data["file_size"] = int(data.get("file_size", 0) or 0)
        data["chunk_count"] = int(data.get("chunk_count", 0) or 0)
        data["is_pending"] = data.get("status") in _PENDING_STATUSES
        return data

    def counts(self) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM library_items
                GROUP BY status;
                """.strip()
            ).fetchall()
        by_status = {str(row["status"]): int(row["count"]) for row in rows}
        total = sum(by_status.values())
        pending = sum(count for status, count in by_status.items() if status in _PENDING_STATUSES)
        return {
            "total": total,
            "pending": pending,
            "ready": int(by_status.get(_READY_STATUS, 0)),
            "failed": int(by_status.get(_FAILED_STATUS, 0)),
        }

    def panel_payload(self, *, limit: int = 120) -> dict[str, Any]:
        rows = self.list_items(limit=limit)
        return {
            "items": rows,
            "counts": self.counts(),
            "topics": [
                {
                    "id": str(topic.get("id", "")).strip(),
                    "name": str(topic.get("name", "")).strip(),
                    "slug": str(topic.get("slug", "")).strip(),
                }
                for topic in self._load_topics()
                if str(topic.get("id", "")).strip()
            ],
            "projects": self._all_projects(),
        }

    def list_items(self, *, limit: int = 120) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit or 120), _MAX_ITEM_LIMIT))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT i.*,
                       COALESCE(c.chunk_count, 0) AS chunk_count
                FROM library_items AS i
                LEFT JOIN (
                    SELECT item_id, COUNT(*) AS chunk_count
                    FROM library_chunks
                    GROUP BY item_id
                ) AS c
                  ON c.item_id = i.id
                ORDER BY i.updated_at DESC
                LIMIT ?;
                """.strip(),
                (safe_limit,),
            ).fetchall()
        return [self._clean_row(row_to_dict(row)) for row in rows if self._clean_row(row_to_dict(row))]

    def get_item(self, item_id: str) -> dict[str, Any] | None:
        key = str(item_id or "").strip()
        if not key:
            return None
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT i.*,
                       COALESCE(c.chunk_count, 0) AS chunk_count
                FROM library_items AS i
                LEFT JOIN (
                    SELECT item_id, COUNT(*) AS chunk_count
                    FROM library_chunks
                    GROUP BY item_id
                ) AS c
                  ON c.item_id = i.id
                WHERE i.id = ?;
                """.strip(),
                (key,),
            ).fetchone()
        item = self._clean_row(row_to_dict(row))
        if not item:
            return None
        summary_path = Path(str(item.get("summary_path", "")).strip()) if str(item.get("summary_path", "")).strip() else None
        markdown_path = Path(str(item.get("markdown_path", "")).strip()) if str(item.get("markdown_path", "")).strip() else None
        if summary_path and summary_path.exists():
            try:
                item["summary_markdown"] = summary_path.read_text(encoding="utf-8")
            except OSError:
                item["summary_markdown"] = ""
        else:
            item["summary_markdown"] = ""
        if markdown_path and markdown_path.exists():
            item["markdown_size"] = int(markdown_path.stat().st_size)
        else:
            item["markdown_size"] = 0
        return item

    def update_item(
        self,
        item_id: str,
        *,
        title: str | None = None,
        source_kind: str | None = None,
        topic_id: str | None = None,
        project_slug: str | None = None,
    ) -> dict[str, Any] | None:
        current = self.get_item(item_id)
        if current is None:
            return None
        previous_target_slug = self._artifact_target_slug(current)
        next_title = _compact(title)[:_MAX_TITLE_CHARS] if title is not None else str(current.get("title", ""))
        next_kind = _safe_kind(source_kind if source_kind is not None else str(current.get("source_kind", "")))
        next_topic = self._normalize_topic_id(topic_id if topic_id is not None else str(current.get("topic_id", "")))
        next_project = self._normalize_project_slug(project_slug if project_slug is not None else str(current.get("project_slug", "")))
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE library_items
                SET title = ?, source_kind = ?, topic_id = ?, project_slug = ?, updated_at = ?
                WHERE id = ?;
                """.strip(),
                (next_title, next_kind, next_topic, next_project, now, item_id),
            )
        updated = self.get_item(item_id)
        if updated:
            try:
                self._remove_project_summary_artifacts(str(updated.get("id", "")).strip(), target_slug=previous_target_slug)
                if updated.get("status") == _READY_STATUS:
                    self._write_project_summary_artifact(updated)
            except Exception:
                pass
        return updated

    def delete_item(self, item_id: str) -> bool:
        current = self.get_item(item_id)
        if current is None:
            return False
        try:
            self._remove_project_summary_artifacts(item_id, target_slug=self._artifact_target_slug(current))
        except Exception:
            pass
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM library_chunks WHERE item_id = ?;", (item_id,))
            conn.execute("DELETE FROM library_items WHERE id = ?;", (item_id,))
        item_dir = self.items_root / item_id
        shutil.rmtree(item_dir, ignore_errors=True)
        return True

    def intake_file(
        self,
        source_path: Path,
        *,
        source_name: str = "",
        mime: str = "",
        source_kind: str = "general",
        title: str = "",
        topic_id: str = "",
        project_slug: str = "",
        source_origin: str = "manual_upload",
        conversation_id: str = "",
    ) -> dict[str, Any]:
        src = Path(source_path)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Library source file not found: {src}")
        content_hash = self._hash_file(src)
        normalized_topic_id = self._normalize_topic_id(topic_id)
        normalized_project = self._normalize_project_slug(project_slug)
        clean_name = str(source_name or src.name).strip() or src.name
        clean_title = _compact(title)[:_MAX_TITLE_CHARS] or self._title_from_name(clean_name)
        safe_kind = _safe_kind(source_kind)
        now = _now_iso()
        reused_item_id = ""

        with self._lock, self._connect() as conn:
            existing = conn.execute(
                """
                SELECT id
                FROM library_items
                WHERE content_hash = ?
                ORDER BY updated_at DESC
                LIMIT 1;
                """.strip(),
                (content_hash,),
            ).fetchone()
            if existing is not None:
                reused_item_id = str(existing["id"]).strip()
                conn.execute(
                    """
                    UPDATE library_items
                    SET title = COALESCE(NULLIF(?, ''), title),
                        source_kind = ?,
                        topic_id = COALESCE(NULLIF(?, ''), topic_id),
                        project_slug = COALESCE(NULLIF(?, ''), project_slug),
                        source_origin = ?,
                        conversation_id = COALESCE(NULLIF(?, ''), conversation_id),
                        updated_at = ?
                    WHERE id = ?;
                    """.strip(),
                    (
                        clean_title,
                        safe_kind,
                        normalized_topic_id,
                        normalized_project,
                        str(source_origin or "manual_upload").strip() or "manual_upload",
                        str(conversation_id or "").strip(),
                        now,
                        reused_item_id,
                    ),
                )
            else:
                item_id = f"lib_{uuid.uuid4().hex[:10]}"
                item_dir = self._item_dir(item_id)
                target_ext = src.suffix.lower()
                source_file = item_dir / f"source{target_ext}"
                shutil.copy2(src, source_file)
                conn.execute(
                    """
                    INSERT INTO library_items(
                        id, title, source_name, source_kind, mime, ext, file_size, content_hash,
                        source_path, markdown_path, summary_path, topic_id, project_slug, status,
                        error_text, source_origin, conversation_id, created_at, updated_at, ingested_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', ?, ?, 'queued', '', ?, ?, ?, ?, '');
                    """.strip(),
                    (
                        item_id,
                        clean_title,
                        clean_name,
                        safe_kind,
                        str(mime or "").strip().lower(),
                        target_ext,
                        int(source_file.stat().st_size),
                        content_hash,
                        str(source_file),
                        normalized_topic_id,
                        normalized_project,
                        str(source_origin or "manual_upload").strip() or "manual_upload",
                        str(conversation_id or "").strip(),
                        now,
                        now,
                    ),
                )

        if reused_item_id:
            item = self.get_item(reused_item_id)
            if item is None:
                raise RuntimeError("Failed to load reused library item.")
            item["reused"] = True
            return item

        item = self.get_item(item_id)
        if item is None:
            raise RuntimeError("Failed to create library item.")
        return item

    def enqueue_ingest(self, item_id: str) -> None:
        target = str(item_id or "").strip()
        if not target:
            return
        thread = threading.Thread(
            target=self._ingest_item,
            args=(target,),
            daemon=True,
            name=f"foxforge-library-{target[:8]}",
        )
        thread.start()

    def read_markdown(self, item_id: str) -> dict[str, Any] | None:
        item = self.get_item(item_id)
        if item is None:
            return None
        path = Path(str(item.get("markdown_path", "")).strip())
        if not path.exists():
            return None
        return {
            "id": item_id,
            "path": str(path),
            "name": path.name,
            "content": path.read_text(encoding="utf-8"),
        }

    def source_file(self, item_id: str) -> Path | None:
        item = self.get_item(item_id)
        if item is None:
            return None
        path = Path(str(item.get("source_path", "")).strip())
        return path if path.exists() and path.is_file() else None

    def retrieve(
        self,
        query: str,
        *,
        topic_id: str = "",
        project_slug: str = "",
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        text = _compact(query)
        if not text:
            return []
        normalized_topic = self._normalize_topic_id(topic_id)
        normalized_project = self._normalize_project_slug(project_slug)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT i.id, i.title, i.topic_id, i.project_slug, i.updated_at,
                       c.chunk_index, c.heading, c.chunk_text, c.embedding_json, c.char_count
                FROM library_items AS i
                INNER JOIN library_chunks AS c
                  ON c.item_id = i.id
                WHERE i.status = ?
                ORDER BY i.updated_at DESC, c.chunk_index ASC;
                """.strip(),
                (_READY_STATUS,),
            ).fetchall()

        query_tokens = _tokens(text)
        query_vec = self._try_embed(text[:2000])
        items: list[dict[str, Any]] = []
        for row in rows:
            row_dict = row_to_dict(row) or {}
            item_topic = self._normalize_topic_id(row_dict.get("topic_id"))
            item_project = self._normalize_project_slug(row_dict.get("project_slug"))
            priority = self._priority_bucket(
                item_topic=item_topic,
                item_project=item_project,
                active_topic=normalized_topic,
                active_project=normalized_project,
            )
            if priority >= 3:
                continue
            chunk_text = str(row_dict.get("chunk_text", "")).strip()
            if not chunk_text:
                continue
            score = 0.0
            if query_vec is not None:
                try:
                    chunk_vec = json.loads(str(row_dict.get("embedding_json", "") or ""))
                    if isinstance(chunk_vec, list) and chunk_vec:
                        score = _vec_cosine(query_vec, [float(x) for x in chunk_vec])
                except (json.JSONDecodeError, ValueError, TypeError):
                    score = 0.0
            if score <= 0.0:
                score = _bow_cosine(query_tokens, _tokens(chunk_text[:3000]))
            threshold = _VECTOR_THRESHOLD if query_vec is not None else 0.05
            if priority == 0:
                threshold *= 0.45
            elif priority == 1:
                threshold *= 0.8
            if score < threshold:
                continue
            items.append(
                {
                    "item_id": str(row_dict.get("id", "")).strip(),
                    "title": str(row_dict.get("title", "")).strip() or "Library Item",
                    "topic_id": item_topic,
                    "project_slug": item_project,
                    "priority": priority,
                    "score": round(float(score), 3),
                    "heading": str(row_dict.get("heading", "")).strip(),
                    "preview": chunk_text[:700].strip(),
                    "updated_at": str(row_dict.get("updated_at", "")).strip(),
                }
            )

        items.sort(key=lambda row: (int(row["priority"]), -float(row["score"]), str(row["updated_at"])), reverse=False)
        return items[: max(1, min(int(limit or 3), 10))]

    def context_text(
        self,
        query: str,
        *,
        topic_id: str = "",
        project_slug: str = "",
        limit: int = 3,
    ) -> str:
        rows = self.retrieve(query, topic_id=topic_id, project_slug=project_slug, limit=limit)
        if not rows:
            return ""
        lines = ["Relevant Library context:"]
        for row in rows:
            heading = f" / {row['heading']}" if row.get("heading") else ""
            lines.append(f"- [{row['title']}{heading}] score={row['score']:.2f}")
            lines.append(f"  preview: {row['preview']}")
        return "\n".join(lines)

    def _priority_bucket(
        self,
        *,
        item_topic: str,
        item_project: str,
        active_topic: str,
        active_project: str,
    ) -> int:
        if active_project and item_project and item_project == active_project:
            return 0
        if active_topic and item_topic and item_topic == active_topic:
            return 0
        if not item_project and not item_topic:
            return 1
        if not active_project and not active_topic:
            return 1
        return 3

    def _set_status(
        self,
        item_id: str,
        status: str,
        *,
        error_text: str = "",
        markdown_path: str | None = None,
        summary_path: str | None = None,
        topic_id: str | None = None,
        project_slug: str | None = None,
        ingested_at: str | None = None,
        title: str | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            current = conn.execute(
                "SELECT * FROM library_items WHERE id = ?;",
                (item_id,),
            ).fetchone()
            if current is None:
                return
            base = row_to_dict(current) or {}
            conn.execute(
                """
                UPDATE library_items
                SET status = ?,
                    error_text = ?,
                    markdown_path = ?,
                    summary_path = ?,
                    topic_id = ?,
                    project_slug = ?,
                    ingested_at = ?,
                    title = ?,
                    updated_at = ?
                WHERE id = ?;
                """.strip(),
                (
                    str(status or "").strip() or str(base.get("status", "")),
                    str(error_text or "").strip(),
                    str(markdown_path if markdown_path is not None else base.get("markdown_path", "")).strip(),
                    str(summary_path if summary_path is not None else base.get("summary_path", "")).strip(),
                    self._normalize_topic_id(topic_id if topic_id is not None else str(base.get("topic_id", ""))),
                    self._normalize_project_slug(project_slug if project_slug is not None else str(base.get("project_slug", ""))),
                    str(ingested_at if ingested_at is not None else base.get("ingested_at", "")).strip(),
                    _compact(title if title is not None else str(base.get("title", "")))[:_MAX_TITLE_CHARS],
                    _now_iso(),
                    item_id,
                ),
            )

    def _try_embed(self, text: str) -> list[float] | None:
        try:
            return OllamaClient().embed(self._embed_model(), text, timeout=3)
        except Exception:
            return None

    def suggest_topic(self, *, title: str, preview_text: str) -> str:
        topics = self._load_topics()
        if not topics:
            return ""
        query = _compact(f"{title}\n{preview_text[:2000]}")
        if not query:
            return ""
        query_vec = self._try_embed(query[:2000])
        query_tokens = _tokens(query)
        best_id = ""
        best_score = 0.0
        for topic in topics:
            topic_id = str(topic.get("id", "")).strip()
            if not topic_id:
                continue
            topic_text = _compact(
                f"{topic.get('name', '')}\n{topic.get('description', '')}\n{topic.get('seed_question', '')}"
            )
            score = 0.0
            if query_vec is not None:
                topic_vec = self._try_embed(topic_text[:2000])
                if topic_vec is not None:
                    score = _vec_cosine(query_vec, topic_vec)
            if score <= 0.0:
                score = _bow_cosine(query_tokens, _tokens(topic_text))
            if score > best_score:
                best_id = topic_id
                best_score = score
        threshold = 0.33 if query_vec is not None else 0.12
        return best_id if best_score >= threshold else ""

    def _normalize_markdown(self, text: str, *, title: str, item: dict[str, Any]) -> str:
        body = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        meta = [
            "---",
            f'title: "{title.replace(chr(34), chr(39))}"',
            f'source_name: "{str(item.get("source_name", "")).replace(chr(34), chr(39))}"',
            f'source_kind: "{str(item.get("source_kind", "general"))}"',
            f'source_origin: "{str(item.get("source_origin", "manual_upload"))}"',
            f'topic_id: "{str(item.get("topic_id", ""))}"',
            f'project_slug: "{str(item.get("project_slug", ""))}"',
            "---",
            "",
            f"# {title}",
            "",
        ]
        return "\n".join(meta) + body + "\n"

    def _split_window(self, text: str) -> list[str]:
        clean = _compact(text)
        if not clean:
            return []
        if len(clean) <= _CHUNK_TARGET_CHARS:
            return [clean]
        chunks: list[str] = []
        start = 0
        while start < len(clean):
            end = min(len(clean), start + _CHUNK_TARGET_CHARS)
            chunk = clean[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(clean):
                break
            start = max(start + _CHUNK_TARGET_CHARS - _CHUNK_OVERLAP_CHARS, start + 1)
        return chunks

    def _chunk_markdown(self, markdown_text: str) -> list[dict[str, Any]]:
        parts = re.split(r"(?m)^(#{1,6}\s+.+)$", str(markdown_text or ""))
        chunks: list[dict[str, Any]] = []
        current_heading = ""
        for idx, part in enumerate(parts):
            block = str(part or "").strip()
            if not block:
                continue
            if block.startswith("#"):
                current_heading = re.sub(r"^#{1,6}\s+", "", block).strip()
                continue
            windows = self._split_window(block)
            for window in windows:
                chunks.append({"heading": current_heading, "chunk_text": window})
        if chunks:
            return chunks
        paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", str(markdown_text or "")) if segment.strip()]
        fallback_text = "\n\n".join(paragraphs)
        return [{"heading": "", "chunk_text": window} for window in self._split_window(fallback_text)]

    def _summary_markdown(self, *, item: dict[str, Any], title: str, summary_text: str) -> str:
        source_name = str(item.get("source_name", "")).strip()
        topic_id = str(item.get("topic_id", "")).strip() or "unlinked"
        project_slug = str(item.get("project_slug", "")).strip() or "unlinked"
        return "\n".join(
            [
                f"# Summary: {title}",
                "",
                f"- Source: {source_name}",
                f"- Kind: {item.get('source_kind', 'general')}",
                f"- Topic: {topic_id}",
                f"- Project: {project_slug}",
                "",
                "## Overview",
                "",
                summary_text.strip() or "No summary available.",
                "",
            ]
        )

    def _heuristic_summary(self, chunks: list[dict[str, Any]]) -> str:
        snippets: list[str] = []
        for row in chunks[:3]:
            text = _compact(str(row.get("chunk_text", "")))
            if text:
                snippets.append(text[:360])
        return "\n\n".join(snippets)[:1400] or "No summary available."

    def _model_summary(self, *, title: str, markdown_text: str) -> str:
        model = self._orchestrator_model()
        if not model:
            return ""
        prompt = (
            "Summarize this private library document for reuse in local retrieval. "
            "Write 2 short paragraphs focusing on core ideas, scope, and what kinds of future questions "
            "this document can answer. Do not use markdown bullets.\n\n"
            f"Title: {title}\n\n"
            f"{markdown_text[:9000]}"
        )
        try:
            return OllamaClient().chat(
                model=model,
                system_prompt="You summarize documents for a local retrieval system. Be concise and factual.",
                user_prompt=prompt,
                temperature=0.2,
                num_ctx=16384,
                think=False,
                timeout=5,
            ).strip()
        except Exception:
            return ""

    def _artifact_target_slug(self, item: dict[str, Any]) -> str:
        project_slug = self._normalize_project_slug(item.get("project_slug"))
        if project_slug:
            return project_slug
        topic_id = self._normalize_topic_id(item.get("topic_id"))
        if topic_id:
            topic = self._topic_engine().get_topic(topic_id)
            if isinstance(topic, dict):
                return str(topic.get("slug", "")).strip().lower()
        return ""

    def _write_project_summary_artifact(self, item: dict[str, Any]) -> None:
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            return
        target_slug = self._artifact_target_slug(item)
        if not target_slug:
            return
        summary_path = Path(str(item.get("summary_path", "")).strip())
        if not summary_path.exists():
            return
        target_dir = self.repo_root / "Projects" / target_slug / "research_summaries"
        target_dir.mkdir(parents=True, exist_ok=True)
        self._remove_project_summary_artifacts(item_id)
        target_name = (
            f"library__{item_id}__"
            f"{_slugify(item.get('title', item.get('source_name', 'item')))}.md"
        )
        shutil.copy2(summary_path, target_dir / target_name)

    def _remove_project_summary_artifacts(self, item_id: str, *, target_slug: str | None = None) -> None:
        key = str(item_id or "").strip()
        if not key:
            return
        if target_slug is not None:
            target_dir = self.repo_root / "Projects" / str(target_slug).strip().lower() / "research_summaries"
            dirs = [target_dir]
        else:
            dirs = [
                project_dir / "research_summaries"
                for project_dir in (self.repo_root / "Projects").glob("*")
                if project_dir.is_dir()
            ]
        pattern = f"library__{key}__*.md"
        for directory in dirs:
            if not directory.exists():
                continue
            for path in directory.glob(pattern):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    continue

    def _ingest_item(self, item_id: str) -> None:
        item = self.get_item(item_id)
        if item is None:
            return
        source_path = Path(str(item.get("source_path", "")).strip())
        if not source_path.exists():
            self._set_status(item_id, _FAILED_STATUS, error_text="Source file is missing.")
            return
        try:
            self._set_status(item_id, "extracting", error_text="")
            raw_text = extract_text(source_path, str(item.get("mime", "")))
            if not raw_text.strip():
                self._set_status(item_id, _FAILED_STATUS, error_text="Text extraction produced no usable content.")
                return
            title = str(item.get("title", "")).strip() or self._title_from_name(str(item.get("source_name", "")))
            topic_id = self._normalize_topic_id(item.get("topic_id"))
            if not topic_id:
                self._set_status(item_id, "matching_topic", title=title)
                topic_id = self.suggest_topic(title=title, preview_text=raw_text[:3000])
            project_slug = self._normalize_project_slug(item.get("project_slug"))
            markdown_text = self._normalize_markdown(raw_text, title=title, item={**item, "topic_id": topic_id, "project_slug": project_slug})
            markdown_path = self._item_dir(item_id) / "content.md"
            markdown_path.write_text(markdown_text, encoding="utf-8")
            self._set_status(
                item_id,
                "chunking",
                markdown_path=str(markdown_path),
                topic_id=topic_id,
                project_slug=project_slug,
                title=title,
            )
            chunks = self._chunk_markdown(markdown_text)
            with self._lock, self._connect() as conn:
                conn.execute("DELETE FROM library_chunks WHERE item_id = ?;", (item_id,))
            self._set_status(item_id, "embedding", markdown_path=str(markdown_path), topic_id=topic_id, project_slug=project_slug, title=title)
            client = OllamaClient()
            chunk_rows: list[tuple[Any, ...]] = []
            now = _now_iso()
            for idx, chunk in enumerate(chunks):
                chunk_text = str(chunk.get("chunk_text", "")).strip()
                if not chunk_text:
                    continue
                embedding_json = ""
                try:
                    vec = client.embed(self._embed_model(), chunk_text[:2000], timeout=3)
                    embedding_json = json.dumps(vec, ensure_ascii=True)
                except Exception:
                    embedding_json = ""
                chunk_rows.append(
                    (
                        item_id,
                        idx,
                        str(chunk.get("heading", "")).strip(),
                        chunk_text,
                        embedding_json,
                        len(_TOKEN_RE.findall(chunk_text)),
                        len(chunk_text),
                        now,
                    )
                )
            if chunk_rows:
                with self._lock, self._connect() as conn:
                    conn.executemany(
                        """
                        INSERT INTO library_chunks(
                            item_id, chunk_index, heading, chunk_text, embedding_json,
                            token_count, char_count, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                        """.strip(),
                        chunk_rows,
                    )
            self._set_status(item_id, "summarizing", markdown_path=str(markdown_path), topic_id=topic_id, project_slug=project_slug, title=title)
            summary_text = self._model_summary(title=title, markdown_text=markdown_text)
            if not summary_text:
                summary_text = self._heuristic_summary(chunks)
            summary_path = self._item_dir(item_id) / "summary.md"
            summary_path.write_text(
                self._summary_markdown(
                    item={**item, "topic_id": topic_id, "project_slug": project_slug},
                    title=title,
                    summary_text=summary_text,
                ),
                encoding="utf-8",
            )
            self._set_status(
                item_id,
                _READY_STATUS,
                markdown_path=str(markdown_path),
                summary_path=str(summary_path),
                topic_id=topic_id,
                project_slug=project_slug,
                ingested_at=_now_iso(),
                title=title,
            )
            ready_item = self.get_item(item_id)
            if ready_item is not None:
                self._write_project_summary_artifact(ready_item)
        except Exception as exc:
            self._set_status(item_id, _FAILED_STATUS, error_text=str(exc)[:800])
