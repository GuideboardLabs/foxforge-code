"""Coding-only Make type registry for Foxforge-code."""

from __future__ import annotations

from typing import Any


MAKE_CATALOG: dict[str, dict[str, Any]] = {
    "tool": {
        "label": "Tool / Script",
        "category": "code",
        "short_description": "Shell or Python tool script.",
        "process_note": "Build a runnable coding tool with tests or smoke checks.",
        "lane": "make_tool",
        "destination": "tools",
        "model_lane": "make_tool",
    },
    "web_app": {
        "label": "Web App",
        "category": "code",
        "short_description": "Stack-driven web app scaffold.",
        "process_note": "Generate backend and frontend scaffold from selected stack.",
        "lane": "make_app",
        "destination": "web_apps",
        "model_lane": "make_app",
    },
    "desktop_app": {
        "label": "Desktop App",
        "category": "code",
        "short_description": ".NET + Avalonia desktop scaffold.",
        "process_note": "Build desktop skeleton using MVVM conventions.",
        "lane": "make_desktop_app",
        "destination": "desktop_apps",
        "model_lane": "make_desktop_app",
    },
}


def lane_for_type(type_id: str) -> str:
    row = MAKE_CATALOG.get(str(type_id or "").strip())
    if row:
        return str(row.get("lane", "make_tool"))
    return "make_tool"


def get_type(type_id: str) -> dict[str, Any] | None:
    row = MAKE_CATALOG.get(str(type_id or "").strip())
    return dict(row) if row else None


def list_types() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for type_id, row in MAKE_CATALOG.items():
        out.append({"type_id": type_id, **row})
    return out


__all__ = ["MAKE_CATALOG", "lane_for_type", "get_type", "list_types"]
