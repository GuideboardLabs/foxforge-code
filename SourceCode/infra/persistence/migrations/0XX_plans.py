"""Plans table migration placeholder for Foxforge-code."""

from __future__ import annotations

import sqlite3


def apply(conn: sqlite3.Connection) -> None:
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
