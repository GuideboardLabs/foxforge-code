# Stack patterns — Python CLI + JSON storage

## Directory structure

```
src/
  {{project_slug}}/
    __init__.py
    cli.py           # CLI entry points (Click/Typer commands)
    core.py          # Business logic — no CLI concerns here
    store.py         # JSON read/write — all file I/O in one place
    models.py        # Dataclasses or Pydantic models for data structures
tests/
  test_core.py
  test_store.py
data/                # Runtime JSON files (gitignored)
pyproject.toml       # or setup.py / requirements.txt
```

## CLI entry pattern (Typer)

```python
# src/{{project_slug}}/cli.py
import typer
from {{project_slug}} import core

app = typer.Typer()

@app.command()
def add(name: str, description: str = ""):
    """Add a new item."""
    item = core.create_item(name=name, description=description)
    typer.echo(f"Created: {item['id']} — {item['name']}")

@app.command()
def list():
    """List all items."""
    items = core.list_items()
    for item in items:
        typer.echo(f"{item['id']}: {item['name']}")

if __name__ == "__main__":
    app()
```

## Separation: CLI from logic

`cli.py` handles input/output only. `core.py` contains all logic. `store.py` handles file I/O. This makes core and store fully testable without CLI.

## JSON store pattern

```python
# src/{{project_slug}}/store.py
import json
from pathlib import Path
from typing import Any

DATA_DIR = Path.home() / ".{{project_slug}}"

def _db_path() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "data.json"

def load() -> dict[str, Any]:
    path = _db_path()
    if not path.exists():
        return {"items": []}
    return json.loads(path.read_text(encoding="utf-8"))

def save(data: dict[str, Any]) -> None:
    _db_path().write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
```

Store data in `~/.{project_slug}/` not in the repo working directory.

## Naming conventions

- Commands: lowercase verbs — `add`, `list`, `remove`, `update`
- Core functions: `create_{resource}`, `get_{resource}`, `list_{resources}`, `delete_{resource}`
- Data keys in JSON: snake_case strings

## Common mistakes to avoid

- Do NOT put business logic in CLI commands — put it in `core.py`
- Do NOT write JSON files to the repo root — use `~/.{project_slug}/`
- Do NOT catch all exceptions silently — let errors surface so Typer can display them
- Do NOT use `print()` in `core.py` — return values, let CLI handle output
