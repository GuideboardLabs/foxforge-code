from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ConfigIni:
    path: Path
    parser: ConfigParser

    def section(self, name: str) -> dict[str, str]:
        key = str(name or "").strip()
        if not key:
            return {}
        if not self.parser.has_section(key):
            return {}
        return {k: v for k, v in self.parser.items(key)}

    def get(self, section: str, key: str, *, fallback: str | None = None) -> str:
        if self.parser.has_option(section, key):
            return self.parser.get(section, key)
        if fallback is None:
            raise KeyError(f"Missing config key [{section}] {key}")
        return fallback

    def get_model(self, role: str) -> str:
        model_key = str(role or "").strip()
        if not model_key:
            raise KeyError("Model role is required.")
        if not self.parser.has_section("models"):
            raise KeyError("Missing [models] section in config.ini")
        if not self.parser.has_option("models", model_key):
            raise KeyError(f"Missing model role in [models]: {model_key}")
        model = self.parser.get("models", model_key).strip()
        if not model:
            raise KeyError(f"Empty model assignment in [models]: {model_key}")
        return model

    def set_model(self, role: str, model_name: str) -> None:
        model_key = str(role or "").strip()
        model_value = str(model_name or "").strip()
        if not model_key or not model_value:
            raise ValueError("Both role and model name are required.")
        if not self.parser.has_section("models"):
            self.parser.add_section("models")
        self.parser.set("models", model_key, model_value)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            self.parser.write(handle)


def load_config(repo_root: str | Path) -> ConfigIni:
    root = Path(repo_root)
    path = root / "config.ini"
    if not path.exists():
        raise FileNotFoundError(f"Missing config.ini at {path}")

    parser = ConfigParser()
    parser.optionxform = str
    parser.read(path, encoding="utf-8")
    return ConfigIni(path=path, parser=parser)


__all__ = ["ConfigIni", "load_config"]
