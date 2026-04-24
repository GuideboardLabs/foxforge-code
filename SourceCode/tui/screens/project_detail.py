from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static


class ProjectDetailScreen(Screen):
    def __init__(self, *, project_slug: str) -> None:
        super().__init__()
        self._project_slug = project_slug

    def compose(self) -> ComposeResult:
        yield Static(f"Project: {self._project_slug}")
