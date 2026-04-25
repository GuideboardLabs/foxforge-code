from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
from textual.widgets import RichLog

from SourceCode.tui.theme import theme

_PATH_RE = re.compile(r"(/(?:[a-zA-Z0-9_.\-/])+\.(?:[a-zA-Z0-9]{1,6}))")

_GRIP = str(Path(__file__).resolve().parents[3] / ".venv" / "bin" / "grip")


def _grip_port(path: str) -> int:
    """Launch grip for a markdown file on a deterministic port and return that port."""
    import hashlib
    port = 6200 + (int(hashlib.md5(path.encode()).hexdigest(), 16) % 800)
    try:
        subprocess.Popen(
            [_GRIP, path, str(port), "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    return port


def _highlight_paths(text: str) -> Text:
    """Return a Rich Text with file paths visually highlighted.

    Textual strips OSC 8 hyperlink sequences so we can't make true clickable
    links inside the TUI. Instead, paths are rendered in a distinct colour so
    they're easy to spot. VS Code's native terminal link provider will detect
    absolute paths in the raw output and make them Ctrl+Clickable on its own.
    """
    result = Text()
    last = 0
    for m in _PATH_RE.finditer(text):
        start, end = m.span()
        if start > last:
            result.append(text[last:start])
        path = m.group(0)
        if os.path.exists(path):
            result.append(path, style=f"underline {theme.AMBER}")
        else:
            result.append(path)
        last = end
    if last < len(text):
        result.append(text[last:])
    return result


class ReasoningStream(RichLog):
    def on_mount(self) -> None:
        self.write(f"[bold {theme.PRIMARY}]Foxforge-code reasoning stream ready.[/]")

    def thinking(self, text: str) -> None:
        self.write(f"[italic {theme.DIM}]  {text}[/]")

    def progress(self, text: str) -> None:
        line = Text.assemble(
            Text(f"• ", style=theme.HOT),
            _highlight_paths(text),
        )
        self.write(line)

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
        line = Text.assemble(
            Text("", style=theme.IVORY),
            _highlight_paths(text),
        )
        self.write(line)

    def render_markdown(self, text: str) -> None:
        self.write(Rule(style=theme.DIM))
        self.write(Markdown(text, hyperlinks=True))
        self.write(Rule(style=theme.DIM))
