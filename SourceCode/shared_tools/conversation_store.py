from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_title(text: str) -> str:
    compact = " ".join(str(text or "").strip().split())
    if not compact:
        return "Project Chat"
    return compact if len(compact) <= 64 else f"{compact[:61]}..."


def _clean_project(text: str) -> str:
    compact = "-".join(str(text or "").strip().split())
    return compact.lower() or "general"


def _atomic_write_text(path: Path, content: str) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


class ConversationStore:
    def __init__(self, repo_root: Path, user_id: str | None = None) -> None:
        uid = str(user_id or "").strip()
        if uid:
            self.root = repo_root / "Runtime" / "users" / uid / "conversations"
        else:
            self.root = repo_root / "Runtime" / "conversations"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()

    def _path_for(self, conversation_id: str) -> Path:
        return self.root / f"{conversation_id}.json"

    def _load(self, conversation_id: str) -> dict[str, Any] | None:
        path = self._path_for(conversation_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _save(self, data: dict[str, Any]) -> None:
        path = self._path_for(str(data["id"]))
        _atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=True))

    @staticmethod
    def _read_index(messages: list[dict[str, Any]], last_read_message_id: str) -> int:
        needle = str(last_read_message_id or "").strip()
        if not needle:
            return -1
        for idx in range(len(messages) - 1, -1, -1):
            if str(messages[idx].get("id", "")).strip() == needle:
                return idx
        return -1

    @classmethod
    def _assistant_unread_count(cls, data: dict[str, Any]) -> int:
        messages = data.get("messages") if isinstance(data.get("messages"), list) else []
        read_idx = cls._read_index(messages, str(data.get("last_read_message_id", "")).strip())
        unread = 0
        for idx, row in enumerate(messages):
            if idx <= read_idx:
                continue
            if str((row or {}).get("role", "")).strip().lower() == "assistant":
                unread += 1
        return unread

    @classmethod
    def _decorate(cls, data: dict[str, Any]) -> dict[str, Any]:
        unread_count = cls._assistant_unread_count(data)
        data["last_read_message_id"] = str(data.get("last_read_message_id", "")).strip()
        data["unread_count"] = unread_count
        data["has_unread"] = unread_count > 0
        data["project"] = _clean_project(data.get("project", "general"))
        # stripped legacy fields for project-only TUI
        data.pop("topic_id", None)
        data.pop("path", None)
        data.pop("title_manually_set", None)
        data.pop("image_style", None)
        data.pop("selected_loras", None)
        return data

    def get_or_create_for_project(self, project_slug: str, *, title: str = "Project Chat") -> dict[str, Any]:
        project = _clean_project(project_slug)
        with self.lock:
            for path in self.root.glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if _clean_project(payload.get("project", "general")) == project:
                    return self._decorate(payload)

            now = _now_iso()
            data = {
                "id": uuid.uuid4().hex[:12],
                "title": _clean_title(title),
                "project": project,
                "created_at": now,
                "updated_at": now,
                "summary": "",
                "messages": [],
                "last_read_message_id": "",
            }
            self._save(data)
            return self._decorate(data)

    def create(
        self,
        title: str = "Project Chat",
        project: str = "general",
        topic_id: str = "",
        path: str = "",
    ) -> dict[str, Any]:
        _ = (topic_id, path)
        return self.get_or_create_for_project(project_slug=project, title=title)

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        with self.lock:
            data = self._load(conversation_id)
            if data is None:
                return None
            return self._decorate(data)

    def rename(self, conversation_id: str, title: str, *, manual: bool = True) -> dict[str, Any] | None:
        _ = manual
        with self.lock:
            data = self._load(conversation_id)
            if data is None:
                return None
            data["title"] = _clean_title(title)
            data["updated_at"] = _now_iso()
            self._save(data)
            return self._decorate(data)

    def set_project(self, conversation_id: str, project: str) -> dict[str, Any] | None:
        with self.lock:
            data = self._load(conversation_id)
            if data is None:
                return None
            data["project"] = _clean_project(project)
            data["updated_at"] = _now_iso()
            self._save(data)
            return self._decorate(data)

    def set_path(self, conversation_id: str, path: str) -> dict[str, Any] | None:
        _ = path
        with self.lock:
            data = self._load(conversation_id)
            if data is None:
                return None
            return self._decorate(data)

    def set_topic(self, conversation_id: str, topic_id: str) -> dict[str, Any] | None:
        _ = topic_id
        with self.lock:
            data = self._load(conversation_id)
            if data is None:
                return None
            return self._decorate(data)

    def set_image_preferences(
        self,
        conversation_id: str,
        *,
        image_style: str | None = None,
        selected_loras: list[str] | None = None,
    ) -> dict[str, Any] | None:
        _ = (image_style, selected_loras)
        with self.lock:
            data = self._load(conversation_id)
            if data is None:
                return None
            return self._decorate(data)

    def delete(self, conversation_id: str) -> bool:
        with self.lock:
            path = self._path_for(conversation_id)
            if not path.exists():
                return False
            path.unlink(missing_ok=True)
            return True

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        mode: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        foraging: bool | None = None,
        building: bool | None = None,
        request_id: str | None = None,
        meta: dict[str, Any] | None = None,
        reply_to: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self.lock:
            data = self._load(conversation_id)
            if data is None:
                return None

            message = {
                "id": uuid.uuid4().hex[:10],
                "role": role,
                "content": content,
                "ts": _now_iso(),
            }
            if mode:
                message["mode"] = str(mode).strip().lower()
            if foraging is not None:
                message["foraging"] = bool(foraging)
            if building is not None:
                message["building"] = bool(building)
            if request_id:
                message["request_id"] = str(request_id).strip()
            if attachments:
                message["attachments"] = [row for row in attachments if isinstance(row, dict)]
            if meta and isinstance(meta, dict):
                message["meta"] = {k: v for k, v in meta.items() if v is not None}
            if reply_to and isinstance(reply_to, dict):
                message["reply_to"] = {
                    "id": str(reply_to.get("id", "")).strip(),
                    "role": str(reply_to.get("role", "")).strip(),
                    "excerpt": str(reply_to.get("excerpt", ""))[:300].strip(),
                }

            data.setdefault("messages", []).append(message)
            data["updated_at"] = _now_iso()
            self._save(data)
            return message

    def replace_messages(
        self,
        conversation_id: str,
        messages: list[dict[str, Any]] | None,
        *,
        summary: str | None = None,
        last_read_message_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self.lock:
            data = self._load(conversation_id)
            if data is None:
                return None
            safe_messages = [json.loads(json.dumps(row, ensure_ascii=True)) for row in (messages or []) if isinstance(row, dict)]
            data["messages"] = safe_messages
            if summary is not None:
                data["summary"] = str(summary).strip()[:600]
            if last_read_message_id is not None:
                data["last_read_message_id"] = str(last_read_message_id or "").strip()
            data["updated_at"] = _now_iso()
            self._save(data)
            return self._decorate(data)

    def mark_read(self, conversation_id: str) -> dict[str, Any] | None:
        with self.lock:
            data = self._load(conversation_id)
            if data is None:
                return None
            messages = data.get("messages") if isinstance(data.get("messages"), list) else []
            data["last_read_message_id"] = str(messages[-1].get("id", "")).strip() if messages else ""
            self._save(data)
            return self._decorate(data)

    def update_summary(self, conversation_id: str, summary: str) -> None:
        with self.lock:
            data = self._load(conversation_id)
            if data is None:
                return
            data["summary"] = str(summary).strip()[:600]
            data["updated_at"] = _now_iso()
            self._save(data)

    def get_summary(self, conversation_id: str) -> str:
        with self.lock:
            data = self._load(conversation_id)
            if data is None:
                return ""
            return str(data.get("summary", ""))

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            items: list[dict[str, Any]] = []
            for path in self.root.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                messages = data.get("messages") or []
                last_preview = ""
                if messages:
                    compact = " ".join(str(messages[-1].get("content", "")).split())
                    last_preview = compact[:90]
                unread_count = self._assistant_unread_count(data)
                items.append(
                    {
                        "id": data.get("id", path.stem),
                        "title": data.get("title", "Project Chat"),
                        "project": _clean_project(data.get("project", "general")),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "message_count": len(messages),
                        "last_preview": last_preview,
                        "summary": str(data.get("summary", ""))[:200],
                        "last_read_message_id": str(data.get("last_read_message_id", "")).strip(),
                        "unread_count": unread_count,
                        "has_unread": unread_count > 0,
                    }
                )
            items.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
            return items
