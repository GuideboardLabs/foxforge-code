from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static


class ProjectListScreen(Screen):
    def __init__(self, rows: list[dict[str, str]] | None = None) -> None:
        super().__init__()
        self._rows = rows or []

    def compose(self) -> ComposeResult:
        if not self._rows:
            yield Static("No projects yet. Use /new to create one.")
            return
        lines = ["Projects:"]
        for row in self._rows:
            lines.append(f"- {row.get('slug', '')}: {row.get('name', '')} [{row.get('project_type', '')}]")
        yield Static("\n".join(lines))
