from __future__ import annotations

from textual.widgets import RichLog

from SourceCode.tui.theme import theme


class ReasoningStream(RichLog):
    def on_mount(self) -> None:
        self.write(f"[bold {theme.PRIMARY}]Foxforge-code reasoning stream ready.[/]")

    def thinking(self, text: str) -> None:
        self.write(f"[italic {theme.DIM}]  {text}[/]")

    def progress(self, text: str) -> None:
        self.write(f"[{theme.HOT}]• {text}[/]")

    def info(self, text: str) -> None:
        self.write(f"[{theme.INFO}]→ {text}[/]")

    def success(self, text: str) -> None:
        self.write(f"[{theme.SUCCESS}]✓ {text}[/]")

    def warning(self, text: str) -> None:
        self.write(f"[{theme.WARNING}]! {text}[/]")

    def error(self, text: str) -> None:
        self.write(f"[{theme.ERROR}]x {text}[/]")

    def user_input(self, text: str) -> None:
        self.write(f"[{theme.IVORY}]$ {text}[/]")

    def answer(self, text: str) -> None:
        self.write(f"[{theme.IVORY}]{text}[/]")
