from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class ProjectImportError(RuntimeError):
    pass


def _ensure_empty_or_confirmed(target_dir: Path, *, overwrite: bool = False) -> None:
    if not target_dir.exists():
        return
    if not target_dir.is_dir():
        raise ProjectImportError(f"Target exists and is not a directory: {target_dir}")
    if any(target_dir.iterdir()) and not overwrite:
        raise ProjectImportError(f"Target directory is not empty: {target_dir}")


def import_from_git(url: str, target_dir: str | Path, *, overwrite: bool = False) -> Path:
    target = Path(target_dir).expanduser().resolve()
    _ensure_empty_or_confirmed(target, overwrite=overwrite)
    if target.exists() and overwrite:
        shutil.rmtree(target)

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", str(url), str(target)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ProjectImportError((completed.stderr or completed.stdout or "git clone failed").strip())
    return target


def import_from_path(source_path: str | Path, target_dir: str | Path, mode: str = "attach", *, overwrite: bool = False) -> Path:
    src = Path(source_path).expanduser().resolve()
    if not src.exists() or not src.is_dir():
        raise ProjectImportError(f"Source directory does not exist: {src}")

    normalized_mode = str(mode or "attach").strip().lower()
    if normalized_mode == "attach":
        return src
    if normalized_mode != "copy":
        raise ProjectImportError("mode must be either 'attach' or 'copy'.")

    dst = Path(target_dir).expanduser().resolve()
    _ensure_empty_or_confirmed(dst, overwrite=overwrite)
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


__all__ = ["ProjectImportError", "import_from_git", "import_from_path"]
