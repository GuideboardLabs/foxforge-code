```
   🭏   🭏
  ▄██▄▄██
 ▄███████▄
 ███▀ ▀███
 ███   ███
 ██████████
 ████████████\
 ████████████  ███
 ███   ███ ██  █████
███   ███ ███   ████
```

# Foxforge-code

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Runtime](https://img.shields.io/badge/runtime-local--only-darkgreen)
![Interface](https://img.shields.io/badge/interface-TUI-orange)
![Status](https://img.shields.io/badge/status-experimental-yellow)

**A local-only terminal coding assistant. No API keys. No cloud. No subscriptions.**

Foxforge-code is a TUI-first fork of [Foxforge](https://github.com/GuideboardLabs/Foxforge) focused on coding workflows. It keeps the project-centric workspace and multi-agent loops, drops the web GUI, and runs entirely against local models through Ollama.

## Highlights

- Project-centric workspace — every session lives in a named project
- Plan / execute / build loops with stack-aware scaffolding
- Local model routing via `config.ini` (Ollama-first)
- Optional Telegram, Discord, and Slack command parity
- Zero remote API calls. Ever.

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp config.ini.example config.ini
./start_foxforge_code.sh --check
python3 -m SourceCode.tui.app
```

See [INSTALL_GUIDE.md](INSTALL_GUIDE.md) for the full setup, including Ollama and optional bot integrations.

## Core commands

### Projects

| Command | What it does |
|---|---|
| `/new greenfield <name> <description>` | Start a new project from scratch |
| `/new import <git-url\|path>` | Import an existing codebase |
| `/projects` | List known projects |
| `/open <slug\|id>` | Switch into a project |

### Work loops

| Command | What it does |
|---|---|
| `/msg <text>` | Chat with the active project |
| `/forage <query>` | Search project context |
| `/plan [prompt]` | Produce a plan |
| `/execute --plan <id\|latest>` | Run a plan |
| `/build [prompt]` | Plan + execute in one shot |

### System

| Command | What it does |
|---|---|
| `/stack show\|change\|save` | Inspect or change the detected stack |
| `/git init\|status\|commit\|push` | Project-scoped git ops |
| `/models` / `/models set <role> <model>` | View or reassign model roles |
| `/system` | Show repo root, paths, and runtime info |

## Configuration

- `config.ini` is gitignored — put local tokens and per-role model assignments here.
- `config.ini.example` is the tracked template; copy it and edit.
- All model roles resolve from `[models]` at runtime, so swapping a single model is a one-line change.

## Notes

- This fork is terminal-only; the legacy Flask web GUI was removed.
- `Projects/` is intentionally empty on first run and is gitignored per-project.
- Bots (Telegram / Discord / Slack) are optional — install with `pip install -r requirements-optional-bots.txt`.

## License

Service-Only Source-Available. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
