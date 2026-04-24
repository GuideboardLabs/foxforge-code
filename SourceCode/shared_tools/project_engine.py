from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any


class ProjectType(str, Enum):
    PYTHON_WEBAPP = "python-webapp"
    NODE_WEBAPP = "node-webapp"
    CLI_TOOL = "cli-tool"
    DESKTOP_APP = "desktop-app"
    UNKNOWN = "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug_from_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s_-]", "", str(name or "")).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", cleaned).strip("-")
    return slug[:48] or "project"


def _atomic_write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def _normalize_stack(value: Any) -> dict[str, str]:
    row = value if isinstance(value, dict) else {}
    return {
        "backend": str(row.get("backend", "") or "").strip().lower(),
        "frontend": str(row.get("frontend", "") or "").strip().lower(),
        "database": str(row.get("database", "") or "").strip().lower(),
        "language": str(row.get("language", "") or "").strip().lower(),
    }


def _project_type_from_stack(stack: dict[str, str]) -> str:
    backend = stack.get("backend", "")
    language = stack.get("language", "")
    frontend = stack.get("frontend", "")
    if backend == "avalonia" or language == "dotnet":
        return ProjectType.DESKTOP_APP.value
    if frontend == "none" and backend in {"", "none"}:
        return ProjectType.CLI_TOOL.value
    if backend in {"flask", "fastapi", "django"}:
        return ProjectType.PYTHON_WEBAPP.value
    if backend in {"express", "hono", "nextjs-api"} or language in {"node", "typescript", "javascript"}:
        return ProjectType.NODE_WEBAPP.value
    return ProjectType.UNKNOWN.value


@dataclass(slots=True)
class ProjectRecord:
    id: str
    name: str
    slug: str
    description: str
    stack: dict[str, str]
    project_type: str
    workspace_path: str
    created_at: str
    updated_at: str


class ProjectEngine:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)
        self.projects_dir = self.repo_root / "Runtime" / "projects"
        self.projects_path = self.projects_dir / "projects.json"
        self._lock = Lock()
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        if not self.projects_path.exists():
            self.projects_path.write_text("[]", encoding="utf-8")

    def _load(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.projects_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(payload, list):
            return []
        return [row for row in payload if isinstance(row, dict)]

    def _save(self, projects: list[dict[str, Any]]) -> None:
        _atomic_write(self.projects_path, projects)

    def list_projects(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._load()
            rows.sort(key=lambda row: str(row.get("updated_at", "")), reverse=True)
            return rows

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        key = str(project_id or "").strip()
        with self._lock:
            return next((row for row in self._load() if str(row.get("id", "")) == key), None)

    def get_by_slug(self, slug: str) -> dict[str, Any] | None:
        key = str(slug or "").strip().lower()
        with self._lock:
            return next((row for row in self._load() if str(row.get("slug", "")).strip().lower() == key), None)

    def create_project(
        self,
        name: str,
        description: str,
        stack: dict[str, Any] | None = None,
        workspace_path: str = "",
    ) -> dict[str, Any]:
        clean_name = str(name or "").strip()
        if not clean_name:
            raise ValueError("Project name cannot be empty.")
        clean_description = str(description or "").strip()
        if not clean_description:
            raise ValueError("Project description cannot be empty.")

        clean_stack = _normalize_stack(stack or {})
        now = _now_iso()
        row = {
            "id": f"proj_{uuid.uuid4().hex[:10]}",
            "name": clean_name,
            "slug": _slug_from_name(clean_name),
            "description": clean_description,
            "stack": clean_stack,
            "project_type": _project_type_from_stack(clean_stack),
            "workspace_path": str(workspace_path or "").strip(),
            "created_at": now,
            "updated_at": now,
        }

        with self._lock:
            rows = self._load()
            rows.append(row)
            self._save(rows)
        return row

    def update_project(self, project_id: str, **fields: Any) -> dict[str, Any] | None:
        key = str(project_id or "").strip()
        with self._lock:
            rows = self._load()
            hit: dict[str, Any] | None = None
            for row in rows:
                if str(row.get("id", "")) != key:
                    continue
                if "name" in fields:
                    name = str(fields.get("name") or "").strip()
                    if name:
                        row["name"] = name
                        row["slug"] = _slug_from_name(name)
                if "description" in fields:
                    row["description"] = str(fields.get("description") or "").strip()
                if "workspace_path" in fields:
                    row["workspace_path"] = str(fields.get("workspace_path") or "").strip()
                if "stack" in fields:
                    stack = _normalize_stack(fields.get("stack") or {})
                    row["stack"] = stack
                    row["project_type"] = _project_type_from_stack(stack)
                row["updated_at"] = _now_iso()
                hit = row
                break
            if hit is None:
                return None
            self._save(rows)
        return hit

    def delete_project(self, project_id: str) -> bool:
        key = str(project_id or "").strip()
        with self._lock:
            rows = self._load()
            updated = [row for row in rows if str(row.get("id", "")) != key]
            if len(updated) == len(rows):
                return False
            self._save(updated)
        return True


__all__ = ["ProjectEngine", "ProjectRecord", "ProjectType"]
