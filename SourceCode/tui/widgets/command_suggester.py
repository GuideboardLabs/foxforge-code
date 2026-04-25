from __future__ import annotations

import sys
from pathlib import Path

from textual.suggester import Suggester

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "SourceCode"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from shared_tools.project_engine import ProjectEngine

_COMMANDS = [
    "/build",
    "/execute --plan latest",
    "/forage ",
    "/git commit",
    "/git init",
    "/git push",
    "/git status",
    "/help",
    "/models",
    "/models set ",
    "/msg ",
    "/new greenfield ",
    "/new import ",
    "/open ",
    "/plan",
    "/plan --refresh latest",
    "/projects",
    "/quit",
    "/stack change ",
    "/stack save ",
    "/stack show",
    "/system",
    "/view",
    "/view --summary",
    "/view --raw",
    "/view --fancy",
    "/view --summary --fancy",
    "/view --raw --fancy",
]

_MODEL_ROLES = [
    "orchestrator_reasoning", "reynard_layer", "intent_confirmer",
    "chat_routing_gate", "embeddings", "research_market_analyst",
    "research_technical", "research_risk", "research_execution_planner",
    "synthesis_default", "synthesis_premium", "make_tool_architect",
    "make_tool_implementer", "make_webapp_architect", "make_webapp_implementer",
    "make_desktop_architect", "make_desktop_implementer",
    "plan_deep", "plan_shallow", "execute_editor", "stack_recommender",
]


class CommandSuggester(Suggester):
    def __init__(self, repo_root: Path) -> None:
        super().__init__(use_cache=False, case_sensitive=False)
        self._engine = ProjectEngine(repo_root)

    def _project_slugs(self) -> list[str]:
        try:
            return [p["slug"] for p in self._engine.list_projects() if p.get("slug")]
        except Exception:
            return []

    def get_all_suggestions(self, value: str) -> list[str]:
        v = value.lower()

        if v.startswith("/open "):
            typed = value[6:]
            return ["/open " + s for s in self._project_slugs() if s.lower().startswith(typed.lower())]

        if v.startswith("/models set "):
            typed = value[12:]
            return ["/models set " + r for r in _MODEL_ROLES if r.lower().startswith(typed.lower())]

        if v.startswith("/stack change "):
            backends = ["fastapi", "flask", "django", "hono", "express", "nextjs-api", "avalonia"]
            typed = value[14:]
            if " " not in typed:
                return ["/stack change " + b for b in backends if b.startswith(typed.lower())]

        return [cmd for cmd in _COMMANDS if cmd.lower().startswith(v) and cmd.lower() != v]

    async def get_suggestion(self, value: str) -> str | None:
        if not value:
            return None

        v = value.lower()

        # /open <slug>
        if v.startswith("/open "):
            typed = value[6:]
            for slug in self._project_slugs():
                if slug.lower().startswith(typed.lower()):
                    return "/open " + slug

        # /models set <role>
        if v.startswith("/models set "):
            typed = value[12:]
            for role in _MODEL_ROLES:
                if role.lower().startswith(typed.lower()):
                    return "/models set " + role

        # /stack change <backend> ...  — suggest backends after "change "
        if v.startswith("/stack change "):
            backends = ["fastapi", "flask", "django", "hono", "express", "nextjs-api", "avalonia"]
            typed = value[14:]
            if " " not in typed:
                for b in backends:
                    if b.startswith(typed.lower()):
                        return "/stack change " + b

        # Plain command completion
        for cmd in _COMMANDS:
            if cmd.lower().startswith(v) and cmd.lower() != v:
                # Preserve the user's capitalisation prefix + suggest rest
                return value + cmd[len(value):]

        return None
