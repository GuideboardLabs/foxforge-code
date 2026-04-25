from __future__ import annotations

from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.events import Mount
from textual.widgets import Header, Input

from SourceCode.tui.commands.dispatcher import (
    CommandDispatcher,
    CommandOutcome,
    DispatcherState,
)
from SourceCode.tui.theme import THEME_CSS
from SourceCode.tui.widgets.command_palette import CommandPalette
from SourceCode.tui.widgets.command_suggester import CommandSuggester
from SourceCode.tui.widgets.fox_banner import FoxBanner
from SourceCode.tui.widgets.project_footer import ProjectFooter
from SourceCode.tui.widgets.reasoning_stream import ReasoningStream
from SourceCode.tui.widgets.thinking_indicator import ThinkingIndicator
from SourceCode.shared_tools import intent_verifier


def _task_for(text: str) -> str:
    t = text.strip().lower()
    if t.startswith("/forage"):
        return "forage"
    if t.startswith("/plan"):
        return "plan"
    if t.startswith("/build") or t.startswith("/execute") or t.startswith("/new"):
        return "build"
    return "msg"


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
        intent_verifier.init(self.repo_root)
        self._suggester = CommandSuggester(self.repo_root)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="body"):
            yield FoxBanner()
            yield ReasoningStream(id="stream", wrap=True, highlight=True, markup=True)
            yield ThinkingIndicator()
            yield CommandPalette(suggester=self._suggester)
        yield ProjectFooter()

    def on_mount(self, event: Mount) -> None:
        _ = event
        stream = self.query_one(ReasoningStream)
        stream.info("Welcome to Foxforge-code")
        stream.answer("Use /new greenfield <name> <description> to start, or /help for commands.")
        intent_verifier.prime()

    @on(Input.Submitted, "#command-line")
    def handle_command(self, event: Input.Submitted) -> None:
        text = str(event.value or "").strip()
        event.input.value = ""
        if not text:
            return
        stream = self.query_one(ReasoningStream)
        stream.user_input(text)
        task = _task_for(text)
        self.query_one(ThinkingIndicator).start(task=task)
        self._run_dispatch(text)

    @work(thread=True, exclusive=False)
    def _run_dispatch(self, text: str) -> None:
        ok, clarification = intent_verifier.verify(
            text,
            project_slug=self.state.active_project_slug,
        )
        if not ok and clarification:
            self.call_from_thread(self._apply_outcome, CommandOutcome(clarification, error=False, active_project_slug=self.state.active_project_slug))
            return

        def _progress(label: str) -> None:
            self.call_from_thread(self._stream_progress, label)

        out = self.dispatcher.dispatch(text, self.state, progress_fn=_progress)
        self.call_from_thread(self._apply_outcome, out)

    def _stream_progress(self, label: str) -> None:
        self.query_one(ReasoningStream).progress(label)
        self.query_one(ThinkingIndicator).update_label(label)

    def _apply_outcome(self, out: CommandOutcome) -> None:
        self.query_one(ThinkingIndicator).stop()
        if out.active_project_slug:
            self.state.active_project_slug = out.active_project_slug
            footer = self.query_one(ProjectFooter)
            footer.project_slug = out.active_project_slug
            try:
                proj = self.dispatcher.project_engine.get_by_slug(out.active_project_slug)
                footer.project_name = proj.get("name", "") if proj else ""
            except Exception:
                footer.project_name = ""
        if out.text == "QUIT":
            self.exit()
            return
        stream = self.query_one(ReasoningStream)
        if out.error:
            stream.error(out.text)
        elif out.render_md:
            stream.render_markdown(out.text)
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
