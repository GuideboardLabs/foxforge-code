from __future__ import annotations

import hashlib
from typing import Iterable


MAKE_TYPES: tuple[str, ...] = (
    "tool",
    "web_app",
    "desktop_app",
)

_MAKE_TYPES_SET = set(MAKE_TYPES)


def normalize_make_type(value: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in _MAKE_TYPES_SET else ""


def list_make_types() -> list[str]:
    return list(MAKE_TYPES)


def validate_make_types(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        item = normalize_make_type(raw)
        if item and item not in out:
            out.append(item)
    return out


def make_types_prompt_fragment() -> str:
    return ", ".join(MAKE_TYPES)


def make_types_hash() -> str:
    joined = "|".join(MAKE_TYPES)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
