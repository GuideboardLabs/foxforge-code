from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS plans (
          id TEXT PRIMARY KEY,
          project_slug TEXT NOT NULL,
          kind TEXT NOT NULL,
          prompt TEXT,
          body_md TEXT NOT NULL,
          created_at TEXT NOT NULL,
          created_workspace_digest TEXT,
          executed_at TEXT,
          execution_result TEXT,
          superseded_by TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_plans_project_created ON plans(project_slug, created_at)")
    conn.commit()


@dataclass(slots=True)
class Plan:
    id: str
    project_slug: str
    kind: str
    prompt: str
    body_md: str
    created_at: str
    created_workspace_digest: str
    executed_at: str
    execution_result: str
    superseded_by: str


class PlanStore:
    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root)
        db_path = self.repo_root / "Runtime" / "state" / "foxforge.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            _ensure_schema(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_plan(self, project_slug: str, kind: str, prompt: str, body_md: str, *, workspace_path: str = "") -> str:
        plan_id = f"plan_{os.urandom(6).hex()}"
        digest = ""
        if workspace_path:
            digest = workspace_digest(workspace_path)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO plans(id, project_slug, kind, prompt, body_md, created_at, created_workspace_digest, executed_at, execution_result, superseded_by)
                VALUES(?, ?, ?, ?, ?, ?, ?, '', '', '')
                """,
                (plan_id, project_slug, kind, prompt, body_md, _now_iso(), digest),
            )
            conn.commit()
        return plan_id

    def latest_plan(self, project_slug: str) -> Plan | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM plans WHERE project_slug = ? ORDER BY created_at DESC LIMIT 1",
                (project_slug,),
            ).fetchone()
        return _to_plan(row)

    def get_plan(self, plan_id: str) -> Plan | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
        return _to_plan(row)

    def mark_executed(self, plan_id: str, result: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE plans SET executed_at = ?, execution_result = ? WHERE id = ?",
                (_now_iso(), str(result or ""), plan_id),
            )
            conn.commit()

    def supersede(self, old_id: str, new_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE plans SET superseded_by = ? WHERE id = ?", (new_id, old_id))
            conn.commit()

    def is_stale(self, plan: Plan, workspace_path: str | Path) -> bool:
        expected = str(plan.created_workspace_digest or "").strip()
        if not expected:
            return False
        return expected != workspace_digest(workspace_path)


def _to_plan(row: sqlite3.Row | None) -> Plan | None:
    if row is None:
        return None
    return Plan(
        id=str(row["id"]),
        project_slug=str(row["project_slug"]),
        kind=str(row["kind"]),
        prompt=str(row["prompt"] or ""),
        body_md=str(row["body_md"] or ""),
        created_at=str(row["created_at"] or ""),
        created_workspace_digest=str(row["created_workspace_digest"] or ""),
        executed_at=str(row["executed_at"] or ""),
        execution_result=str(row["execution_result"] or ""),
        superseded_by=str(row["superseded_by"] or ""),
    )


def _should_skip(path: Path) -> bool:
    parts = set(path.parts)
    return any(name in parts for name in ("__pycache__", "node_modules", ".venv", ".git"))


def workspace_digest(workspace_path: str | Path) -> str:
    root = Path(workspace_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return ""

    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _should_skip(path.relative_to(root)):
            continue
        rel = str(path.relative_to(root))
        stat = path.stat()
        rows.append(f"{rel}|{stat.st_mtime_ns}|{stat.st_size}")

    h = hashlib.sha256()
    for row in rows:
        h.update(row.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


__all__ = ["Plan", "PlanStore", "workspace_digest"]
