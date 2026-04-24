from __future__ import annotations

from dataclasses import dataclass


THEME: dict[str, str] = {
    "BACKGROUND": "#0f0b08",
    "PRIMARY": "#ff7a1a",
    "HOT": "#ff9e3d",
    "AMBER": "#ffbb66",
    "DIM": "#c7a188",
    "IVORY": "#f5ead8",
    "SUCCESS": "#4ade80",
    "ERROR": "#ef4444",
    "WARNING": "#facc15",
    "INFO": "#22d3ee",
    "PLAN": "#e879f9",
}


@dataclass(frozen=True)
class Theme:
    BACKGROUND: str = THEME["BACKGROUND"]
    PRIMARY: str = THEME["PRIMARY"]
    HOT: str = THEME["HOT"]
    AMBER: str = THEME["AMBER"]
    DIM: str = THEME["DIM"]
    IVORY: str = THEME["IVORY"]
    SUCCESS: str = THEME["SUCCESS"]
    ERROR: str = THEME["ERROR"]
    WARNING: str = THEME["WARNING"]
    INFO: str = THEME["INFO"]
    PLAN: str = THEME["PLAN"]


THEME_CSS = f"""
Screen {{
    background: {THEME['BACKGROUND']};
    color: {THEME['IVORY']};
}}

Header {{
    background: {THEME['PRIMARY']};
    color: {THEME['BACKGROUND']};
}}

Footer {{
    background: {THEME['BACKGROUND']};
    color: {THEME['AMBER']};
}}

#body {{
    layout: vertical;
    height: 1fr;
}}

#stream {{
    border: round {THEME['PRIMARY']};
    color: {THEME['IVORY']};
    background: {THEME['BACKGROUND']};
    height: 1fr;
    padding: 1;
}}

#command-line {{
    border: round {THEME['HOT']};
    background: {THEME['BACKGROUND']};
    color: {THEME['IVORY']};
}}
"""


theme = Theme()

__all__ = ["THEME", "Theme", "THEME_CSS", "theme"]
