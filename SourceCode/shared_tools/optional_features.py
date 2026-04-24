from __future__ import annotations

import importlib.util
import shutil
from typing import Any


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def optional_feature_status() -> dict[str, dict[str, Any]]:
    pdf_ready = _has_module("fitz")
    docx_ready = _has_module("docx")
    pillow_ready = _has_module("PIL")
    pytesseract_ready = _has_module("pytesseract")
    discord_ready = _has_module("discord")
    slack_ready = _has_module("slack_bolt")
    tesseract_ready = shutil.which("tesseract") is not None

    document_missing: list[str] = []
    if not pdf_ready:
        document_missing.append("PyMuPDF")
    if not docx_ready:
        document_missing.append("python-docx")

    ocr_missing: list[str] = []
    if not pillow_ready:
        ocr_missing.append("Pillow")
    if not pytesseract_ready:
        ocr_missing.append("pytesseract")
    if not tesseract_ready:
        ocr_missing.append("tesseract binary")

    discord_missing: list[str] = []
    if not discord_ready:
        discord_missing.append("discord.py")
    slack_missing: list[str] = []
    if not slack_ready:
        slack_missing.append("slack-bolt")

    return {
        "document_extraction": {
            "available": not document_missing,
            "missing": document_missing,
            "install_hint": "pip install -r requirements-optional-docs.txt",
        },
        "image_ocr": {
            "available": not ocr_missing,
            "missing": ocr_missing,
            "install_hint": "pip install -r requirements-optional-docs.txt",
        },
        "discord_bot": {
            "available": not discord_missing,
            "missing": discord_missing,
            "install_hint": "pip install -r requirements-optional-bots.txt",
        },
        "telegram_bot": {
            "available": True,
            "missing": [],
            "install_hint": "",
        },
        "slack_bot": {
            "available": not slack_missing,
            "missing": slack_missing,
            "install_hint": "pip install slack-bolt>=1.18",
        },
    }


def feature_warning(feature_key: str) -> str:
    row = optional_feature_status().get(feature_key, {})
    missing = row.get("missing") or []
    if not missing:
        return ""
    install_hint = str(row.get("install_hint", "")).strip()
    parts = [f"missing {', '.join(str(item) for item in missing)}"]
    if install_hint:
        parts.append(f"install with `{install_hint}`")
    return "; ".join(parts)
