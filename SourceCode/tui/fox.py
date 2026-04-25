from __future__ import annotations


_FOX_BASE_RAW = r"""
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
""".strip("\n")


_FOX_HOP_UP_RAW = r"""
   🭏   🭏
  ▄██▄▄██
 ▄███████▄
 ███▀ ▀███
 ███   ███
 ██████████
 ████████████/ ███
 ████████████  █████
 ███   ███ ██   ████
███   ███ ███
""".strip("\n")


def _with_tail_tip(frame: str, tip: str) -> str:
    lines = frame.split("\n")
    lines[6] = lines[6][:-1] + tip
    return "\n".join(lines)


def _with_tail_shift(frame: str, shift: int) -> str:
    lines = frame.split("\n")
    for i in (7, 8, 9):
        line = lines[i]
        last_space = line.rfind(" ")
        if last_space == -1:
            continue
        body = line[: last_space + 1]
        tail = line[last_space + 1 :]
        if shift > 0:
            body = body + (" " * shift)
        elif shift < 0:
            remove = -shift
            available = len(body) - len(body.rstrip()) - 1
            actual = min(remove, available)
            body = body[: len(body) - actual]
        lines[i] = body + tail
    return "\n".join(lines)


def _mirror(frame: str) -> str:
    lines = frame.split("\n")
    width = max(len(line) for line in lines)
    flip = str.maketrans("\\/", "/\\")
    return "\n".join(line.ljust(width)[::-1].translate(flip) for line in lines)


def _pad_frames(*frames: str) -> tuple[str, ...]:
    max_width = max(
        len(line) for frame in frames for line in frame.split("\n")
    )
    return tuple(
        "\n".join(line.ljust(max_width) for line in frame.split("\n"))
        for frame in frames
    )


_WAG_RIGHT_RAW = _with_tail_shift(_with_tail_tip(_FOX_BASE_RAW, "\\"), 1)
_WAG_LEFT_RAW = _with_tail_shift(_with_tail_tip(_FOX_BASE_RAW, "|"), -1)
_FOX_MIRROR_RAW = _mirror(_FOX_BASE_RAW)

FOX_BASE, FOX_HOP_UP, FOX_MIRROR, _WAG_RIGHT, _WAG_LEFT = _pad_frames(
    _FOX_BASE_RAW,
    _FOX_HOP_UP_RAW,
    _FOX_MIRROR_RAW,
    _WAG_RIGHT_RAW,
    _WAG_LEFT_RAW,
)


FOX_WAG_FRAMES: tuple[str, ...] = (_WAG_RIGHT, FOX_BASE, _WAG_LEFT, FOX_BASE)

FOX_VARIANTS: tuple[str, ...] = (FOX_BASE,)


def random_fox() -> str:
    return FOX_BASE


__all__ = [
    "FOX_BASE",
    "FOX_HOP_UP",
    "FOX_MIRROR",
    "FOX_WAG_FRAMES",
    "FOX_VARIANTS",
    "random_fox",
]
