# Foxforge-code

Local-only terminal coding assistant.

Foxforge-code is a TUI-first fork focused on coding workflows in the terminal:
- Project-centric workspace (`/new`, `/open`, `/projects`)
- Plan and execution loops (`/plan`, `/execute`, `/build`)
- Stack-aware scaffolding and imports (`/import` + detect mode)
- Local model routing through `config.ini` (Ollama-first)
- Optional Telegram, Discord, and Slack command parity

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp config.ini.example config.ini
./start_foxforge_code.sh --check
python3 -m SourceCode.tui.app
```

## Core commands

- `/new greenfield <name> <description>`
- `/new import <git-url|path>`
- `/projects`
- `/open <slug|id>`
- `/msg <text>`
- `/forage <query>`
- `/plan [prompt]`
- `/execute --plan <plan_id|latest>`
- `/build [prompt]`
- `/stack show|change|save`
- `/git init|status|commit|push`
- `/models`
- `/models set <role> <model>`
- `/system`

## Configuration

- `config.ini` is ignored by git and should contain local tokens and model assignments.
- `config.ini.example` is the tracked template.
- All model roles are resolved from `[models]` at runtime.

## Notes

- This fork is terminal-only; the legacy web GUI was removed.
- `Projects/` is intentionally empty on first run.
