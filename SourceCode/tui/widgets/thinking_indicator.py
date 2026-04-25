from __future__ import annotations

import random
import time

from rich.text import Text
from textual.timer import Timer
from textual.widget import Widget

from SourceCode.tui.theme import theme
from SourceCode.tui.indicator_phrases import (
    TASK_PHRASES,
    MSG_PHRASES,
    PHRASE_INTERVAL_MIN,
    PHRASE_INTERVAL_MAX,
)


class ThinkingIndicator(Widget):
    DEFAULT_CSS = """
    ThinkingIndicator {
        height: 1;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__(id="thinking-indicator")
        self._timer: Timer | None = None
        self._active = False
        self._task = "msg"
        self._label = "thinking"
        self._dots = 0
        self._started_at: float = 0.0
        self._phrase_idx = 0
        self._next_phrase_at: float = 0.0

    def render(self) -> Text:
        if not self._active:
            return Text(" ")

        dots = "." * self._dots
        pad = " " * (3 - self._dots)
        left = f"  {self._label}{dots}{pad}"

        elapsed = int(time.monotonic() - self._started_at)
        right = f"{elapsed}s"

        try:
            width = self.size.width
        except Exception:
            width = 80

        gap = max(1, width - len(left) - len(right))
        line = Text()
        line.append(left, style=f"italic {theme.DIM}")
        line.append(" " * gap)
        line.append(right, style=f"bold {theme.DIM}")
        return line

    def start(self, label: str = "", task: str = "msg") -> None:
        self._active = True
        self._task = task
        self._phrase_idx = 0
        self._started_at = time.monotonic()
        self._next_phrase_at = self._started_at + random.randint(PHRASE_INTERVAL_MIN, PHRASE_INTERVAL_MAX)
        phrases = TASK_PHRASES.get(task, MSG_PHRASES)
        self._label = label or phrases[0]
        self._dots = 0
        self.refresh()
        if self._timer is None:
            self._timer = self.set_interval(0.35, self._tick)

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._active = False
        self.refresh()

    def update_label(self, label: str) -> None:
        if self._active:
            self._label = label
            self.refresh()

    def _tick(self) -> None:
        self._dots = (self._dots + 1) % 4
        now = time.monotonic()
        if now >= self._next_phrase_at:
            phrases = TASK_PHRASES.get(self._task, MSG_PHRASES)
            self._phrase_idx = (self._phrase_idx + 1) % len(phrases)
            self._label = phrases[self._phrase_idx]
            self._next_phrase_at = now + random.randint(PHRASE_INTERVAL_MIN, PHRASE_INTERVAL_MAX)
        self.refresh()
