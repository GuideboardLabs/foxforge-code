from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _default_stack() -> dict[str, str]:
    return {
        "backend": "fastapi",
        "frontend": "none",
        "database": "sqlite",
        "language": "python",
    }


def _detect_from_requirements(text: str) -> dict[str, str]:
    low = text.lower()
    stack = _default_stack()
    if "django" in low:
        stack["backend"] = "django"
        stack["frontend"] = "none"
    elif "flask" in low:
        stack["backend"] = "flask"
        stack["frontend"] = "htmx"
    elif "fastapi" in low:
        stack["backend"] = "fastapi"
    if "psycopg" in low or "asyncpg" in low:
        stack["database"] = "postgres"
    elif "pymongo" in low:
        stack["database"] = "mongodb"
    elif "sqlite" in low:
        stack["database"] = "sqlite"
    return stack


def _detect_from_package_json(payload: dict[str, Any]) -> dict[str, str]:
    deps = {}
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        section = payload.get(key)
        if isinstance(section, dict):
            deps.update({str(k).lower(): str(v) for k, v in section.items()})

    stack = {
        "backend": "express",
        "frontend": "none",
        "database": "json-file",
        "language": "node",
    }
    if "hono" in deps:
        stack["backend"] = "hono"
    elif "next" in deps:
        stack["backend"] = "nextjs-api"
        stack["frontend"] = "react"
    if "react" in deps:
        stack["frontend"] = "react"
    elif "vue" in deps:
        stack["frontend"] = "vue"
    elif "svelte" in deps:
        stack["frontend"] = "svelte"

    if "pg" in deps or "postgres" in deps:
        stack["database"] = "postgres"
    elif "mongoose" in deps or "mongodb" in deps:
        stack["database"] = "mongodb"
    elif "sqlite3" in deps or "better-sqlite3" in deps:
        stack["database"] = "sqlite"
    return stack


def recommend_greenfield(description: str) -> dict[str, Any]:
    text = str(description or "").lower()
    stack = _default_stack()
    rationale = "Defaulting to a simple Python web stack."

    if any(token in text for token in ("desktop", "avalonia", ".net", "dotnet")):
        stack = {"backend": "avalonia", "frontend": "none", "database": "none", "language": "dotnet"}
        rationale = "Desktop signals detected; selected .NET + Avalonia."
    elif any(token in text for token in ("node", "typescript", "javascript", "express", "hono", "next")):
        stack = {"backend": "hono", "frontend": "svelte", "database": "sqlite", "language": "node"}
        rationale = "Node/JS signals detected; selected modern Node stack."
    elif "django" in text:
        stack = {"backend": "django", "frontend": "none", "database": "postgres", "language": "python"}
        rationale = "Django requested explicitly."
    elif "flask" in text:
        stack = {"backend": "flask", "frontend": "htmx", "database": "sqlite", "language": "python"}
        rationale = "Flask requested explicitly."

    return {
        "recommended": stack,
        "rationale": rationale,
    }


def recommend_detect(workspace_path: str | Path) -> dict[str, Any]:
    root = Path(workspace_path).expanduser().resolve()
    signals: list[str] = []
    stack: dict[str, str] | None = None

    package_json = root / "package.json"
    if package_json.exists():
        try:
            payload = json.loads(package_json.read_text(encoding="utf-8"))
            stack = _detect_from_package_json(payload)
            signals.append("package.json")
        except (json.JSONDecodeError, OSError):
            signals.append("package.json(parse-failed)")

    req = root / "requirements.txt"
    if stack is None and req.exists():
        stack = _detect_from_requirements(req.read_text(encoding="utf-8"))
        signals.append("requirements.txt")

    pyproject = root / "pyproject.toml"
    if stack is None and pyproject.exists():
        text = pyproject.read_text(encoding="utf-8")
        stack = _detect_from_requirements(text)
        signals.append("pyproject.toml")

    if (root / "go.mod").exists():
        stack = {"backend": "none", "frontend": "none", "database": "json-file", "language": "go"}
        signals.append("go.mod")
    if (root / "Cargo.toml").exists():
        stack = {"backend": "none", "frontend": "none", "database": "json-file", "language": "rust"}
        signals.append("Cargo.toml")

    if stack is None:
        stack = _default_stack()
        signals.append("fallback-default")

    if (root / "next.config.js").exists() or (root / "next.config.mjs").exists():
        stack["backend"] = "nextjs-api"
        stack["frontend"] = "react"
        stack["language"] = "node"
        signals.append("next.config.*")
    if (root / "svelte.config.js").exists() or (root / "svelte.config.mjs").exists():
        stack["frontend"] = "svelte"
        signals.append("svelte.config.*")

    html_hits = list(root.glob("**/*.html"))[:25]
    htmx_found = False
    for html in html_hits:
        try:
            if re.search(r"\bhtmx\b", html.read_text(encoding="utf-8", errors="ignore"), flags=re.IGNORECASE):
                htmx_found = True
                break
        except OSError:
            continue
    if htmx_found:
        stack["frontend"] = "htmx"
        signals.append("htmx-html-signal")

    return {
        "recommended": stack,
        "rationale": "Detected stack from workspace manifests and framework markers.",
        "detected": True,
        "signals": signals,
    }


__all__ = ["recommend_greenfield", "recommend_detect"]
