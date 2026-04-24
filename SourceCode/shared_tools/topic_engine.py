"""Backward-compatible shim around ProjectEngine.

Foxforge-code now uses project semantics. This module remains only to avoid
breaking legacy imports while callers migrate to `project_engine`.
"""

from __future__ import annotations

from typing import Any

from shared_tools.project_engine import ProjectEngine, ProjectType


VALID_TOPIC_TYPES: frozenset[str] = frozenset({
    ProjectType.PYTHON_WEBAPP.value,
    ProjectType.NODE_WEBAPP.value,
    ProjectType.CLI_TOOL.value,
    ProjectType.DESKTOP_APP.value,
    ProjectType.UNKNOWN.value,
})


class TopicEngine(ProjectEngine):
    def list_topics(self, parent_id: str = "") -> list[dict[str, Any]]:
        _ = parent_id
        return self.list_projects()

    def get_topic(self, topic_id: str) -> dict[str, Any] | None:
        return self.get_project(topic_id)

    def create_topic(
        self,
        name: str,
        type: str,
        description: str,
        seed_question: str,
        parent_id: str = "",
    ) -> dict[str, Any]:
        _ = (type, seed_question, parent_id)
        return self.create_project(name=name, description=description, stack={}, workspace_path="")

    def update_topic(self, topic_id: str, **fields: Any) -> dict[str, Any] | None:
        mapped: dict[str, Any] = {}
        if "name" in fields:
            mapped["name"] = fields["name"]
        if "description" in fields:
            mapped["description"] = fields["description"]
        return self.update_project(topic_id, **mapped)

    def delete_topic(self, topic_id: str) -> bool:
        return self.delete_project(topic_id)


__all__ = ["TopicEngine", "VALID_TOPIC_TYPES"]
