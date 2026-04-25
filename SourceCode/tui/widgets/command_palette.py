from __future__ import annotations

from textual import events
from textual.suggester import Suggester
from textual.widgets import Input

from SourceCode.tui.widgets.command_suggester import CommandSuggester


class CommandPalette(Input):
    def __init__(self, suggester: Suggester | None = None) -> None:
        super().__init__(
            placeholder="Type a command (try /help)",
            id="command-line",
            suggester=suggester,
        )
        self._suggester_ref: CommandSuggester | None = (
            suggester if isinstance(suggester, CommandSuggester) else None
        )
        self._cycle_seed: str = ""
        self._cycle_matches: list[str] = []
        self._cycle_idx: int = -1
        self._cycling: bool = False

    async def on_key(self, event: events.Key) -> None:
        if event.key in ("up", "down"):
            event.prevent_default()
            event.stop()
            await self._cycle(forward=(event.key == "down"))
        elif event.key not in ("tab", "right"):
            # Any key other than accept-completion resets the cycle
            self._cycling = False
            self._cycle_idx = -1
            self._cycle_matches = []

    async def _cycle(self, forward: bool) -> None:
        if not self._suggester_ref:
            return

        if not self._cycling:
            self._cycle_seed = self.value
            self._cycle_matches = self._suggester_ref.get_all_suggestions(self._cycle_seed)
            self._cycling = True
            self._cycle_idx = -1

        if not self._cycle_matches:
            return

        if forward:
            self._cycle_idx = (self._cycle_idx + 1) % len(self._cycle_matches)
        else:
            self._cycle_idx = (self._cycle_idx - 1) % len(self._cycle_matches)

        self.value = self._cycle_matches[self._cycle_idx]
        self.cursor_position = len(self.value)
