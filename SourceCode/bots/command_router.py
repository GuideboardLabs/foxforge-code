from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

try:
    from tui.commands.dispatcher import CommandDispatcher, DispatcherState
except ImportError:  # pragma: no cover - package-style fallback
    from SourceCode.tui.commands.dispatcher import CommandDispatcher, DispatcherState


@dataclass(slots=True)
class RoutedReply:
    text: str
    error: bool
    active_project: str


class BotCommandRouter:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)
        self.dispatcher = CommandDispatcher(self.repo_root)
        self._states: dict[str, DispatcherState] = {}
        self._lock = Lock()

    def _state_key(self, platform: str, user: str) -> str:
        return f"{platform}:{user}"

    def _get_state(self, platform: str, user: str) -> DispatcherState:
        key = self._state_key(platform, user)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = DispatcherState(active_project_slug="", user_id=user)
                self._states[key] = state
            return state

    def dispatch(self, *, platform: str, user: str, project: str, text: str) -> RoutedReply:
        state = self._get_state(platform, user)
        if project:
            state.active_project_slug = project
        out = self.dispatcher.dispatch(text, state)
        return RoutedReply(
            text=out.text,
            error=out.error,
            active_project=state.active_project_slug,
        )


def chunk_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    rest = text
    while rest:
        if len(rest) <= limit:
            chunks.append(rest)
            break
        split_at = rest.rfind("\n\n", 0, limit)
        if split_at == -1:
            split_at = rest.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(rest[:split_at].strip())
        rest = rest[split_at:].strip()
    return [c for c in chunks if c]


__all__ = ["BotCommandRouter", "RoutedReply", "chunk_text"]
