from __future__ import annotations

from textual.widgets import Static

from SourceCode.tui.fox import random_fox
from SourceCode.tui.theme import theme


class FoxBanner(Static):
    def render_fox(self) -> None:
        self.update(f"[bold {theme.PRIMARY}]" + random_fox() + "[/]")

    def on_mount(self) -> None:
        self.render_fox()
