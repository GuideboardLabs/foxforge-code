from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Any

from bots.command_router import BotCommandRouter, chunk_text

LOGGER = logging.getLogger(__name__)

_HISTORY_LIMIT = 20
_PENDING_MAX = 20

# Module-level store of recent unauthorized DM user IDs, read by the API
_pending_unauthorized: deque[dict[str, str]] = deque(maxlen=_PENDING_MAX)
_pending_lock = threading.Lock()


def get_pending_unauthorized() -> list[dict[str, str]]:
    with _pending_lock:
        return list(_pending_unauthorized)


class DiscordBot(threading.Thread):
    def __init__(self, repo_root: Path, token: str) -> None:
        super().__init__(name="foxforge-discord-bot", daemon=True)
        self._repo_root = repo_root
        self._token = token
        self._user_store: Any = None
        self._router = BotCommandRouter(self._repo_root)

    def run(self) -> None:
        try:
            import discord
        except ImportError:
            LOGGER.error(
                "discord.py is not installed. "
                "Install it with: pip install 'discord.py>=2.0'"
            )
            return

        from bots.bot_user_store import BotUserStore
        self._user_store = BotUserStore(self._repo_root)

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)

        @client.event
        async def on_ready() -> None:
            LOGGER.info("Discord bot connected as %s.", client.user)

        @client.event
        async def on_message(message: discord.Message) -> None:
            if message.author.bot:
                return
            is_dm = isinstance(message.channel, discord.DMChannel)
            is_mentioned = client.user in (message.mentions or [])
            if not is_dm and not is_mentioned:
                return

            text = message.content
            if client.user:
                text = (
                    text.replace(f"<@{client.user.id}>", "")
                        .replace(f"<@!{client.user.id}>", "")
                        .strip()
                )
            if not text:
                return

            loop = asyncio.get_running_loop()
            async with message.channel.typing():
                try:
                    reply = await loop.run_in_executor(
                        None, self._handle_message_sync, message, text, is_dm
                    )
                except Exception:
                    LOGGER.exception("Orchestrator error for Discord message.")
                    reply = "Sorry, something went wrong processing your message."

            if reply:
                for chunk in _chunk_text(reply, 2000):
                    await message.channel.send(chunk)

        asyncio.run(client.start(self._token))

    def _handle_message_sync(self, message: Any, text: str, is_dm: bool) -> str:
        if is_dm:
            mapping = self._get_or_create_dm(message)
            if mapping is None:
                return "Sorry, I couldn't set up your conversation. No owner profile found."
        else:
            mapping = self._get_or_create_guild(message)
            if mapping is None:
                LOGGER.error(
                    "Could not resolve guild mapping for guild %s.", message.guild.id
                )
                return ""

        try:
            return self._run_orchestrator(mapping, text)
        except Exception:
            LOGGER.exception("Orchestrator error for Discord message.")
            return "Sorry, something went wrong processing your message."

    def _get_or_create_dm(self, message: Any) -> dict[str, Any] | None:
        """For DMs: auto-create a mapping on first contact, tied to the owner profile."""
        user_id = str(message.author.id)
        username = str(message.author.name)

        mapping = self._user_store.get_mapping("discord", user_id)
        if mapping:
            return mapping

        owner_uid = self._get_owner_uid()
        if not owner_uid:
            LOGGER.error("No owner profile found — cannot auto-create Discord DM mapping.")
            return None

        from shared_tools.conversation_store import ConversationStore
        store = ConversationStore(self._repo_root, owner_uid)
        conv = store.create(title=f"Discord DM — {username}", project="general")

        LOGGER.info("Auto-created Discord DM mapping for user %s (%s).", username, user_id)
        return self._user_store.create_mapping(
            platform="discord",
            platform_user_id=user_id,
            platform_username=username,
            foxforge_user_id=owner_uid,
            conversation_id=conv["id"],
        )

    def _get_or_create_guild(self, message: Any) -> dict[str, Any] | None:
        """For server messages: one shared conversation per guild, auto-created."""
        guild_id = str(message.guild.id)
        guild_name = str(message.guild.name)
        platform_user_id = f"guild_{guild_id}"

        mapping = self._user_store.get_mapping("discord_guild", platform_user_id)
        if mapping:
            return mapping

        owner_uid = self._get_owner_uid()
        if not owner_uid:
            return None

        from shared_tools.conversation_store import ConversationStore
        store = ConversationStore(self._repo_root, owner_uid)
        conv = store.create(title=f"Discord — {guild_name}", project="general")

        return self._user_store.create_mapping(
            platform="discord_guild",
            platform_user_id=platform_user_id,
            platform_username=guild_name,
            foxforge_user_id=owner_uid,
            conversation_id=conv["id"],
        )

    def _get_owner_uid(self) -> str | None:
        try:
            from shared_tools.family_auth import FamilyAuthStore
            profiles = FamilyAuthStore(self._repo_root).list_profiles()
            owner = next((p for p in profiles if p.get("is_owner")), None)
            return owner["id"] if owner else (profiles[0]["id"] if profiles else None)
        except Exception:
            return None

    def _build_persona(self) -> str:
        """Build Discord persona override with a kind default voice."""
        try:
            from bots.bot_config import load_bot_config
            cfg = load_bot_config(self._repo_root).get("discord", {})
            name = str(cfg.get("persona_name", "") or "").strip() or "Reynard"
            notes = str(cfg.get("persona_notes", "") or "").strip()
        except Exception:
            name = "Reynard"
            notes = ""
        parts: list[str] = []
        parts.append(f"Your name is {name}. Respond to {name} as direct address.")
        parts.append(
            "You are the Discord-facing voice of Foxforge. "
            "Tone: warm, supportive, grounded, and lightly playful."
        )
        parts.append(
            "Be kind and respectful. Avoid snark, sarcasm, mockery, contempt, and condescending replies."
        )
        parts.append("Stay natural and conversational, not robotic or helpdesk-like.")
        if notes:
            parts.append(f"Owner notes: {notes}")
        parts.append(
            "If any owner notes conflict with kindness/respect, keep the response kind and non-snarky."
        )
        parts.append(
            "Just talking. Not performing assistance. "
            "No disclaimers. No 'as an AI' framing. No robotic helpdesk energy."
        )
        return " ".join(parts)

    def _run_orchestrator(self, mapping: dict[str, Any], text: str) -> str:
        platform_user = str(mapping.get("platform_user_id", "")).strip() or str(mapping.get("foxforge_user_id", "")).strip() or "discord-user"
        active_project = str(mapping.get("active_project", "general")).strip() or "general"
        routed = self._router.dispatch(platform="discord", user=platform_user, project=active_project, text=text)
        return routed.text or ""


def _chunk_text(text: str, limit: int) -> list[str]:
    return chunk_text(text, limit)
