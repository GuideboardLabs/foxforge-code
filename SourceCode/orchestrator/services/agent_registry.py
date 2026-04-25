from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from infra.tools import ToolRegistry

from agents_make.desktop_pool import run_desktop_pool
from agents_make.stack_builder_pool import run_stack_builder_pool
from agents_research.deep_researcher import run_research_pool
from agents_tool.tool_pool import run_tool_pool

from .agent_contracts import AgentCapability, AgentTask, BaseAgentExecutor
from .result_types import WorkerResult


class ResearchPoolAgent(BaseAgentExecutor):
    capability = AgentCapability(
        lane="research",
        supports_progress=True,
        supports_cancellation=True,
        supports_history=True,
        description="Runs the deep research pool against project and web context.",
    )

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        bus = tools.require("bus")
        result = run_research_pool(
            task.prompt,
            task.repo_root,
            task.project_slug,
            bus,
            web_context=str(task.context.get("web_context", "") or ""),
            prior_messages=task.history,
            cancel_checker=task.cancel_checker,
            pause_checker=task.pause_checker,
            yield_checker=task.yield_checker,
            progress_callback=task.progress_callback,
            topic_type=str(task.context.get("topic_type", "general") or "general"),
            project_context=str(task.context.get("project_context", "") or ""),
        )
        return WorkerResult.from_legacy("research", result)


class AppPoolAgent(BaseAgentExecutor):
    capability = AgentCapability(
        lane="make_app",
        supports_progress=True,
        supports_cancellation=True,
        description="Builds stack-driven coding project scaffolds.",
    )

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        bus = tools.require("bus")
        result = run_stack_builder_pool(
            question=task.prompt,
            repo_root=task.repo_root,
            project_slug=task.project_slug,
            bus=bus,
            research_context=str(task.context.get("research_context", "") or ""),
            plan_text=str(task.context.get("plan_text", "") or ""),
            stack=dict(task.context.get("stack") or {}),
            workspace_path=str(task.context.get("workspace_path", "") or ""),
            cancel_checker=task.cancel_checker,
            progress_callback=task.progress_callback,
        )
        return WorkerResult.from_legacy("make_app", result)


class ToolPoolAgent(BaseAgentExecutor):
    capability = AgentCapability(
        lane="make_tool",
        supports_progress=True,
        supports_cancellation=True,
        supports_history=True,
        description="Builds runnable tool scripts with fix loops.",
    )

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        bus = tools.require("bus")
        result = run_tool_pool(
            question=task.prompt,
            repo_root=task.repo_root,
            project_slug=task.project_slug,
            bus=bus,
            research_context=str(task.context.get("research_context", "") or ""),
            prior_messages=task.history,
            cancel_checker=task.cancel_checker,
            progress_callback=task.progress_callback,
        )
        return WorkerResult.from_legacy("make_tool", result)


class DesktopPoolAgent(BaseAgentExecutor):
    capability = AgentCapability(
        lane="make_desktop_app",
        supports_progress=True,
        supports_cancellation=True,
        description="Builds .NET 8 + Avalonia desktop app scaffold.",
    )

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        bus = tools.require("bus")
        result = run_desktop_pool(
            question=task.prompt,
            repo_root=task.repo_root,
            project_slug=task.project_slug,
            bus=bus,
            research_context=str(task.context.get("research_context", "") or ""),
            cancel_checker=task.cancel_checker,
            progress_callback=task.progress_callback,
        )
        return WorkerResult.from_legacy("make_desktop_app", result)


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, BaseAgentExecutor] = {}

    def register(self, lane: str, executor: BaseAgentExecutor) -> BaseAgentExecutor:
        key = str(lane or "").strip()
        if not key:
            raise ValueError("Agent lane must be non-empty.")
        self._agents[key] = executor
        return executor

    def get(self, lane: str) -> BaseAgentExecutor | None:
        return self._agents.get(str(lane or "").strip())

    def require(self, lane: str) -> BaseAgentExecutor:
        key = str(lane or "").strip()
        agent = self.get(key)
        if agent is None:
            raise KeyError(f"No agent registered for lane '{key}'.")
        return agent

    def lanes(self) -> list[str]:
        return sorted(self._agents.keys())

    def describe(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for lane in self.lanes():
            cap = self._agents[lane].capability
            rows.append(
                {
                    "lane": lane,
                    "supports_progress": cap.supports_progress,
                    "supports_cancellation": cap.supports_cancellation,
                    "supports_history": cap.supports_history,
                    "produces_artifacts": cap.produces_artifacts,
                    "description": cap.description,
                }
            )
        return rows


@dataclass(slots=True)
class OrchestratorRegistries:
    tools: ToolRegistry
    agents: AgentRegistry


def build_default_agent_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register("research", ResearchPoolAgent())
    registry.register("project", ResearchPoolAgent())
    registry.register("make_app", AppPoolAgent())
    registry.register("ui", AppPoolAgent())
    registry.register("make_tool", ToolPoolAgent())
    registry.register("make_desktop_app", DesktopPoolAgent())
    return registry


__all__ = [
    "AgentRegistry",
    "AppPoolAgent",
    "BaseAgentExecutor",
    "DesktopPoolAgent",
    "OrchestratorRegistries",
    "ResearchPoolAgent",
    "ToolPoolAgent",
    "build_default_agent_registry",
]
