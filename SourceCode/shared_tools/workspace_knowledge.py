"""workspace_knowledge.py — Auto-load project markdown context files from a workspace root.

Scans for known filenames in priority order. Each file is capped and silently
skipped if absent. Returns a single concatenated string for injecting into
chat/plan/forage context.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from shared_tools.stack_catalog import stack_key

_DISCOVERY_ORDER = [
    "FOXFORGE.md",
    "CLAUDE.md",
    "PATTERNS.md",
    "DESIGN.md",
    "CODE.md",
    "ARCHITECTURE.md",
]

_PER_FILE_CAP = 3000
_TOTAL_CAP = 10000


def _resolve_workspace_doc_path(
    workspace_root: Path,
    filename: str,
    default_design_path: Optional[Path],
    default_patterns_path: Optional[Path],
) -> tuple[Path | None, str]:
    project_path = workspace_root / filename
    if project_path.is_file():
        return project_path, filename
    if filename == "PATTERNS.md" and default_patterns_path and default_patterns_path.is_file():
        return default_patterns_path, "PATTERNS.md (stack default)"
    if filename == "DESIGN.md" and default_design_path and default_design_path.is_file():
        return default_design_path, "DESIGN.md (default)"
    return None, filename


def resolve_default_patterns_path(
    repo_root: str | Path,
    stack: Mapping[str, Any] | None,
) -> Path | None:
    """Return stack scaffold PATTERNS.md path when available."""
    if not stack:
        return None
    try:
        key = stack_key(dict(stack))
    except Exception:
        return None
    candidate = Path(str(repo_root)).expanduser() / "SourceCode" / "scaffolds" / key / "PATTERNS.md"
    return candidate if candidate.is_file() else None


def read_workspace_knowledge(
    workspace_path: str | Path,
    max_chars: int = _TOTAL_CAP,
    default_design_path: str | Path | None = None,
    default_patterns_path: str | Path | None = None,
) -> str:
    """Return concatenated content of known markdown context files from workspace_path.

    Files are read in _DISCOVERY_ORDER. Each is capped at _PER_FILE_CAP chars.
    The combined result is capped at max_chars. Missing files are silently skipped.
    Returns empty string if workspace doesn't exist or no files are found.
    """
    root = Path(str(workspace_path or "")).expanduser()
    if not root.exists():
        return ""
    default_design = (
        Path(str(default_design_path)).expanduser()
        if str(default_design_path or "").strip()
        else None
    )
    default_patterns = (
        Path(str(default_patterns_path)).expanduser()
        if str(default_patterns_path or "").strip()
        else None
    )

    parts: list[str] = []
    total = 0

    for name in _DISCOVERY_ORDER:
        path, display_name = _resolve_workspace_doc_path(
            root,
            name,
            default_design,
            default_patterns,
        )
        if path is None:
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not text:
            continue
        chunk = text[:_PER_FILE_CAP]
        header = f"--- [{display_name}] ---"
        entry = f"{header}\n{chunk}"
        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining > len(header) + 40:
                parts.append(entry[:remaining])
            break
        parts.append(entry)
        total += len(entry)

    return "\n\n".join(parts)
