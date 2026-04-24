from __future__ import annotations

from textual.widgets import Input


class CommandPalette(Input):
    def __init__(self) -> None:
        super().__init__(placeholder="Type a command (try /help)", id="command-line")
