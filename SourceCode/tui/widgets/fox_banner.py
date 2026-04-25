from __future__ import annotations

import random
from enum import Enum, auto

from rich.text import Text
from textual.widget import Widget

from SourceCode.tui.fox import FOX_BASE, FOX_HOP_UP, FOX_MIRROR, FOX_WAG_FRAMES


COLOR_PALETTE: tuple[str, ...] = (
    "#ffcf92",
    "#ffbb66",
    "#ff9e3d",
    "#ff7a1a",
    "#d9580a",
)

TICK_MS = 150
COLOR_EVERY_N_TICKS = 3

_TICK_S = TICK_MS / 1000
WAG_CHANCE = _TICK_S / 8
HOP_CHANCE = _TICK_S / 20
TURN_CHANCE = _TICK_S / 30

_WAG_STEPS = 12
_HOP_PATTERN: tuple[bool, ...] = (True, True, False, False, True, True, False, False)
_TURN_STEPS = 20


class _Behavior(Enum):
    IDLE = auto()
    WAG = auto()
    HOP = auto()
    TURN = auto()


class FoxBanner(Widget):
    DEFAULT_CSS = """
    FoxBanner {
        height: 11;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__(id="fox-banner")
        self._tick = 0
        self._color_idx = 0
        self._color_dir = 1
        self._behavior = _Behavior.IDLE
        self._behavior_step = 0
        self._hop_up = False

    def on_mount(self) -> None:
        self.set_interval(TICK_MS / 1000, self._tick_once)

    def render(self) -> Text:
        frame = self._current_frame()
        padded = frame + "\n " if self._hop_up else " \n" + frame
        color = COLOR_PALETTE[self._color_idx]
        return Text(padded, style=f"bold {color}")

    def _tick_once(self) -> None:
        self._tick += 1
        if self._tick % COLOR_EVERY_N_TICKS == 0:
            self._advance_color()
        self._advance_behavior()
        self.refresh()

    def _advance_color(self) -> None:
        nxt = self._color_idx + self._color_dir
        if nxt < 0 or nxt >= len(COLOR_PALETTE):
            self._color_dir *= -1
            nxt = self._color_idx + self._color_dir
        self._color_idx = nxt

    def _advance_behavior(self) -> None:
        if self._behavior == _Behavior.IDLE:
            r = random.random()
            if r < TURN_CHANCE:
                self._behavior = _Behavior.TURN
                self._behavior_step = 0
            elif r < TURN_CHANCE + HOP_CHANCE:
                self._behavior = _Behavior.HOP
                self._behavior_step = 0
            elif r < TURN_CHANCE + HOP_CHANCE + WAG_CHANCE:
                self._behavior = _Behavior.WAG
                self._behavior_step = 0
            self._hop_up = False
            return

        if self._behavior == _Behavior.WAG:
            if self._behavior_step >= _WAG_STEPS:
                self._behavior = _Behavior.IDLE
            else:
                self._behavior_step += 1
        elif self._behavior == _Behavior.HOP:
            if self._behavior_step >= len(_HOP_PATTERN):
                self._behavior = _Behavior.IDLE
                self._hop_up = False
            else:
                self._hop_up = _HOP_PATTERN[self._behavior_step]
                self._behavior_step += 1
        elif self._behavior == _Behavior.TURN:
            if self._behavior_step >= _TURN_STEPS:
                self._behavior = _Behavior.IDLE
            else:
                self._behavior_step += 1

    def _current_frame(self) -> str:
        if self._behavior == _Behavior.TURN:
            return FOX_MIRROR
        if self._behavior == _Behavior.WAG:
            return FOX_WAG_FRAMES[(self._behavior_step // 2) % len(FOX_WAG_FRAMES)]
        if self._behavior == _Behavior.HOP and self._hop_up:
            return FOX_HOP_UP
        return FOX_BASE
