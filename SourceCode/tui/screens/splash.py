from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static

from SourceCode.tui.widgets.fox_banner import FoxBanner
from SourceCode.tui.theme import theme


class SplashScreen(Screen):
    def compose(self) -> ComposeResult:
        yield FoxBanner()
        yield Static(
            "\n".join(
                [
                    f"[bold {theme.PRIMARY}]Foxforge-code[/]",
                    f"[{theme.AMBER}]local-only terminal coding assistant[/]",
                    "",
                    "Use /new to create a project or /help for commands.",
                ]
            )
        )
