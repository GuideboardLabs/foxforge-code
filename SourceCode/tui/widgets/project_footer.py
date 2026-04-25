from __future__ import annotations

from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget

from SourceCode.tui.theme import theme


class ProjectFooter(Widget):
    DEFAULT_CSS = """
    ProjectFooter {
        height: 1;
        dock: bottom;
        background: $panel;
        padding: 0;
    }
    """

    project_slug: reactive[str] = reactive("")
    project_name: reactive[str] = reactive("")

    def render(self) -> Text:
        left = Text()
        left.append(" ^l ", style=f"bold {theme.BACKGROUND} on {theme.PRIMARY}")
        left.append(" Clear ", style=f"{theme.IVORY} on {theme.BACKGROUND}")

        right = Text()
        right.append(" ^p ", style=f"bold {theme.BACKGROUND} on {theme.PRIMARY}")
        right.append(" palette ", style=f"{theme.IVORY} on {theme.BACKGROUND}")

        center = Text()
        if self.project_slug:
            display = self.project_name or self.project_slug
            center.append(f" {display} ", style=f"bold {theme.PRIMARY}")
            center.append(f"[{self.project_slug}] ", style=f"{theme.DIM}")
        else:
            center.append(" no project ", style=f"{theme.DIM}")

        left_str = left.plain
        right_str = right.plain
        center_str = center.plain

        try:
            width = self.size.width
        except Exception:
            width = 80

        padding_total = width - len(left_str) - len(center_str) - len(right_str)
        pad_left = padding_total // 2
        pad_right = padding_total - pad_left

        line = Text()
        line.append_text(left)
        line.append(" " * max(0, pad_left), style=f"on {theme.BACKGROUND}")
        line.append_text(center)
        line.append(" " * max(0, pad_right), style=f"on {theme.BACKGROUND}")
        line.append_text(right)
        return line
