from __future__ import annotations

from textual.widgets import Static

from SourceCode.tui.theme import theme


class PlanCard(Static):
    def set_plan(self, *, plan_id: str, created_at: str, stale: bool, body_md: str) -> None:
        state = "STALE" if stale else "READY"
        state_color = theme.ERROR if stale else theme.SUCCESS
        text = "\n".join(
            [
                f"[{theme.PLAN}]Plan: {plan_id}[/]",
                f"[{theme.AMBER}]Created: {created_at}[/]",
                f"[{state_color}]State: {state}[/]",
                "",
                body_md.strip() or "(empty)",
            ]
        )
        self.update(text)
