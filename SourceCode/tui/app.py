from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header
from textual.events import Mount
from textual import on
from textual.widgets import Input

from SourceCode.tui.commands.dispatcher import CommandDispatcher, DispatcherState
from SourceCode.tui.theme import THEME_CSS, theme
from SourceCode.tui.widgets.command_palette import CommandPalette
from SourceCode.tui.widgets.reasoning_stream import ReasoningStream
from SourceCode.tui.fox import random_fox


class FoxforgeCodeApp(App[None]):
    TITLE = "Foxforge-code"
    SUB_TITLE = "local-only terminal coding assistant"
    CSS = THEME_CSS
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_stream", "Clear"),
    ]

    def __init__(self, repo_root: Path | None = None) -> None:
        super().__init__()
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.dispatcher = CommandDispatcher(self.repo_root)
        self.state = DispatcherState(active_project_slug="", user_id="")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="body"):
            yield ReasoningStream(id="stream", wrap=True, highlight=True, markup=True)
            yield CommandPalette()
        yield Footer()

    def on_mount(self, event: Mount) -> None:
        _ = event
        stream = self.query_one(ReasoningStream)
        stream.info("Welcome to Foxforge-code")
        stream.answer(f"[bold {theme.PRIMARY}]" + random_fox() + "[/]")
        stream.answer("Use /new greenfield <name> <description> to start, or /help for commands.")

    @on(Input.Submitted, "#command-line")
    def handle_command(self, event: Input.Submitted) -> None:
        text = str(event.value or "").strip()
        event.input.value = ""
        if not text:
            return

        stream = self.query_one(ReasoningStream)
        stream.user_input(text)
        out = self.dispatcher.dispatch(text, self.state)
        if out.active_project_slug:
            self.state.active_project_slug = out.active_project_slug

        if out.text == "QUIT":
            self.exit()
            return

        if out.error:
            stream.error(out.text)
        else:
            stream.answer(out.text)

    def action_clear_stream(self) -> None:
        stream = self.query_one(ReasoningStream)
        stream.clear()
        stream.info("Stream cleared.")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    app = FoxforgeCodeApp(repo_root=repo_root)
    app.run()


if __name__ == "__main__":
    main()
