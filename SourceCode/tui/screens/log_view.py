from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import RichLog


class LogViewScreen(Screen):
    def compose(self) -> ComposeResult:
        log = RichLog(highlight=True)
        log.write("Orchestrator event stream view.")
        yield log
