from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from bots.command_router import BotCommandRouter, chunk_text

LOGGER = logging.getLogger(__name__)


class SlackBot(threading.Thread):
    def __init__(self, repo_root: Path, bot_token: str, app_token: str, signing_secret: str = "") -> None:
        super().__init__(name="foxforge-slack-bot", daemon=True)
        self._repo_root = Path(repo_root)
        self._bot_token = str(bot_token or "").strip()
        self._app_token = str(app_token or "").strip()
        self._signing_secret = str(signing_secret or "").strip()
        self._router = BotCommandRouter(self._repo_root)
        self._user_store: Any = None

    def run(self) -> None:
        try:
            from slack_bolt import App
            from slack_bolt.adapter.socket_mode import SocketModeHandler
        except ImportError:
            LOGGER.error("slack-bolt is not installed. Install with: pip install 'slack-bolt>=1.18'")
            return

        if not self._bot_token or not self._app_token:
            LOGGER.error("Slack bot/app token missing; Slack bot will not start.")
            return

        from bots.bot_user_store import BotUserStore
        self._user_store = BotUserStore(self._repo_root)

        app = App(token=self._bot_token, signing_secret=self._signing_secret or None)

        @app.event("message")
        def _handle_message(event: dict[str, Any], say, client, logger) -> None:
            _ = logger
            user_id = str(event.get("user", "")).strip()
            if not user_id:
                return
            text = str(event.get("text", "")).strip()
            if not text:
                return
            channel = str(event.get("channel", "")).strip()
            ts = str(event.get("ts", "")).strip()

            mapping = self._resolve_mapping(user_id)
            if mapping is None:
                say(text="Sorry, you're not authorized to use this bot.")
                return

            platform_user = str(mapping.get("platform_user_id", user_id)).strip() or user_id
            active_project = str(mapping.get("active_project", "general")).strip() or "general"
            routed = self._router.dispatch(platform="slack", user=platform_user, project=active_project, text=text)

            body = routed.text or ""
            if len(body) <= 3800:
                say(text=body)
                return

            chunks = chunk_text(body, 3800)
            first = chunks[0] if chunks else body[:3800]
            posted = client.chat_postMessage(channel=channel, text=first)
            message_ts = posted.get("ts") or ts
            assembled = first
            for part in chunks[1:]:
                assembled = f"{assembled}\n\n{part}"
                if len(assembled) > 39000:
                    assembled = assembled[-39000:]
                client.chat_update(channel=channel, ts=message_ts, text=assembled)

        LOGGER.info("Slack bot starting in Socket Mode.")
        handler = SocketModeHandler(app, self._app_token)
        handler.start()

    def _resolve_mapping(self, user_id: str) -> dict[str, Any] | None:
        mapping = self._user_store.get_mapping("slack", user_id)
        if mapping:
            return mapping

        owner_id = self._get_owner_uid()
        if not owner_id:
            return None

        from shared_tools.conversation_store import ConversationStore
        store = ConversationStore(self._repo_root, owner_id)
        conv = store.get_or_create_for_project("general", title="Slack")
        return self._user_store.create_mapping(
            platform="slack",
            platform_user_id=user_id,
            platform_username=user_id,
            foxforge_user_id=owner_id,
            conversation_id=str(conv.get("id", "")),
        )

    def _get_owner_uid(self) -> str | None:
        try:
            from shared_tools.family_auth import FamilyAuthStore
            profiles = FamilyAuthStore(self._repo_root).list_profiles()
            owner = next((p for p in profiles if p.get("is_owner")), None)
            return owner["id"] if owner else (profiles[0]["id"] if profiles else None)
        except Exception:
            return None


__all__ = ["SlackBot"]
