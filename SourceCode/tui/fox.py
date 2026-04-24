from __future__ import annotations

import random

FOX_VARIANTS: tuple[str, ...] = (
    r"""
   🭏   🭏
  ▄██▄▄██
 ▄███████▄
 ███▀ ▀███
 ███   ███
 ██████████
 ████████████\
 ████████████  ███
 ███   ███ ██  █████
███   ███ ███   ████
""".strip("\n"),
)


def random_fox() -> str:
    return random.choice(FOX_VARIANTS)


__all__ = ["FOX_VARIANTS", "random_fox"]
