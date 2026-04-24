from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from shared_tools.config_ini import load_config


class GitServiceError(RuntimeError):
    pass


def _run_git(path: Path, args: list[str], *, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    completed = subprocess.run(["git", *args], cwd=path, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        raise GitServiceError((completed.stderr or completed.stdout or "git command failed").strip())
    return completed


def init(path: str | Path) -> dict[str, Any]:
    repo = Path(path).expanduser().resolve()
    repo.mkdir(parents=True, exist_ok=True)
    _run_git(repo, ["init"])
    return {"ok": True, "message": "Initialized git repository.", "path": str(repo)}


def status(path: str | Path) -> dict[str, Any]:
    repo = Path(path).expanduser().resolve()
    out = _run_git(repo, ["status", "--porcelain=v1", "--branch"]).stdout
    lines = [line for line in out.splitlines() if line.strip()]
    return {"ok": True, "lines": lines, "raw": out}


def commit(path: str | Path, message: str, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    repo = Path(path).expanduser().resolve()
    msg = str(message or "").strip()
    if not msg:
        raise GitServiceError("Commit message is required.")

    env_extra: dict[str, str] = {}
    if repo_root is not None:
        cfg = load_config(repo_root)
        env_extra["GIT_AUTHOR_NAME"] = cfg.get("integrations.git", "commit_author_name", fallback="Foxforge-code")
        env_extra["GIT_AUTHOR_EMAIL"] = cfg.get("integrations.git", "commit_author_email", fallback="foxforge-code@localhost")
        env_extra["GIT_COMMITTER_NAME"] = env_extra["GIT_AUTHOR_NAME"]
        env_extra["GIT_COMMITTER_EMAIL"] = env_extra["GIT_AUTHOR_EMAIL"]

    _run_git(repo, ["add", "-A"], env_extra=env_extra)
    _run_git(repo, ["commit", "-m", msg], env_extra=env_extra)
    return {"ok": True, "message": "Commit created.", "path": str(repo)}


def push(path: str | Path, *, remote: str = "origin", branch: str | None = None) -> dict[str, Any]:
    repo = Path(path).expanduser().resolve()
    if not branch:
        branch = _run_git(repo, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    _run_git(repo, ["push", remote, branch])
    return {"ok": True, "message": f"Pushed {branch} to {remote}.", "path": str(repo)}


def suggest_commit_message(path: str | Path, recent_plan: str = "") -> str:
    repo = Path(path).expanduser().resolve()
    diff = _run_git(repo, ["diff", "--staged", "--name-only"]).stdout.strip()
    files = [row for row in diff.splitlines() if row.strip()]
    stem = "update"
    if files:
        stem = f"update {files[0]}"
        if len(files) > 1:
            stem = f"update {files[0]} and {len(files)-1} more"
    if recent_plan.strip():
        return f"{stem}: {recent_plan.strip()[:90]}"
    return stem


__all__ = ["GitServiceError", "init", "status", "commit", "push", "suggest_commit_message"]
