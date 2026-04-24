from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKENDS = ("flask", "fastapi", "django", "express", "hono", "nextjs-api")
FRONTENDS = ("react", "vue", "svelte", "htmx", "plain-html", "none")
DATABASES = ("sqlite", "postgres", "mongodb", "json-file")


@dataclass(slots=True)
class PrebuiltStack:
    name: str
    backend: str
    frontend: str
    database: str
    language: str
    notes: str = ""


PREBUILT_STACKS: tuple[PrebuiltStack, ...] = (
    PrebuiltStack("Python CRUD", "fastapi", "react", "postgres", "python"),
    PrebuiltStack("Flask Classic", "flask", "htmx", "sqlite", "python"),
    PrebuiltStack("Django Full", "django", "none", "postgres", "python"),
    PrebuiltStack("Node Modern", "hono", "svelte", "sqlite", "node"),
    PrebuiltStack("Next Full-stack", "nextjs-api", "react", "postgres", "node"),
    PrebuiltStack("CLI Tool", "none", "none", "json-file", "python"),
    PrebuiltStack("Desktop App", "avalonia", "none", "none", "dotnet"),
)


def _normalize_stack(data: dict[str, Any]) -> dict[str, str]:
    return {
        "backend": str(data.get("backend", "") or "").strip().lower(),
        "frontend": str(data.get("frontend", "") or "").strip().lower(),
        "database": str(data.get("database", "") or "").strip().lower(),
        "language": str(data.get("language", "") or "").strip().lower(),
    }


def derive_project_type(stack: dict[str, Any]) -> str:
    normalized = _normalize_stack(stack)
    backend = normalized["backend"]
    language = normalized["language"]
    frontend = normalized["frontend"]

    if backend == "avalonia" or language == "dotnet":
        return "desktop-app"
    if frontend == "none" and backend in {"none", ""}:
        if language in {"node", "javascript", "typescript"}:
            return "cli-tool"
        return "cli-tool"
    if language in {"python", ""} and backend in {"flask", "fastapi", "django"}:
        return "python-webapp"
    if backend in {"express", "hono", "nextjs-api"} or language in {"node", "javascript", "typescript"}:
        return "node-webapp"
    return "cli-tool"


def stack_key(stack: dict[str, Any]) -> str:
    normalized = _normalize_stack(stack)
    project_type = derive_project_type(normalized)
    backend = normalized["backend"] or "none"
    frontend = normalized["frontend"] or "none"
    database = normalized["database"] or "none"
    return f"{project_type}__{backend}__{frontend}__{database}".replace("/", "-")


def load_custom_stacks(repo_root: str | Path) -> list[dict[str, Any]]:
    root = Path(repo_root)
    path = root / "Runtime" / "stacks" / "custom_stacks.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def save_custom_stack(repo_root: str | Path, *, name: str, backend: str, frontend: str, database: str, language: str, notes: str = "") -> dict[str, Any]:
    root = Path(repo_root)
    path = root / "Runtime" / "stacks" / "custom_stacks.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "name": str(name).strip(),
        "backend": str(backend).strip().lower(),
        "frontend": str(frontend).strip().lower(),
        "database": str(database).strip().lower(),
        "language": str(language).strip().lower(),
        "notes": str(notes or "").strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    items = load_custom_stacks(root)
    items.insert(0, row)
    path.write_text(json.dumps(items, indent=2, ensure_ascii=True), encoding="utf-8")
    return row


__all__ = [
    "BACKENDS",
    "FRONTENDS",
    "DATABASES",
    "PREBUILT_STACKS",
    "PrebuiltStack",
    "derive_project_type",
    "stack_key",
    "load_custom_stacks",
    "save_custom_stack",
]
