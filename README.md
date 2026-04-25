```
                       ███████╗ ██████╗ ██╗  ██╗███████╗ ██████╗ ██████╗  ██████╗ ███████╗
                       ██╔════╝██╔═══██╗╚██╗██╔╝██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
                       █████╗  ██║   ██║ ╚███╔╝ █████╗  ██║   ██║██████╔╝██║  ███╗█████╗
   🭏   🭏              ██╔══╝  ██║   ██║ ██╔██╗ ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
  ▄██▄▄██              ██║     ╚██████╔╝██╔╝ ██╗██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
 ▄███████▄             ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
 ███▀ ▀███
 ███   ███                                ██████╗ ██████╗ ██████╗ ███████╗
 ██████████                              ██╔════╝██╔═══██╗██╔══██╗██╔════╝
 ████████████\                           ██║     ██║   ██║██║  ██║█████╗
 ████████████  ███                       ██║     ██║   ██║██║  ██║██╔══╝
 ███   ███ ██  █████                     ╚██████╗╚██████╔╝██████╔╝███████╗
███   ███ ███   ████                      ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
```

# Foxforge-code

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Runtime](https://img.shields.io/badge/runtime-local--only-darkgreen)
![Interface](https://img.shields.io/badge/interface-TUI-orange)
![Status](https://img.shields.io/badge/status-experimental-yellow)

**A local-only terminal coding assistant. No API keys. No cloud. No subscriptions.**

Foxforge-code is a TUI-first fork of [Foxforge](https://github.com/GuideboardLabs/Foxforge) stripped down to coding workflows. It drops the web GUI, the general-purpose conversation layer, and the domain-specific research profiles — and replaces them with a direct coding assistant (Fox), a grounded web research pipeline (DeepFox), and a project workspace that tracks everything locally.

---

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp config.ini.example config.ini
./start_foxforge_code.sh --check
python3 -m SourceCode.tui.app
```

See [INSTALL_GUIDE.md](INSTALL_GUIDE.md) for Ollama setup and optional bot integrations.

---

## What's in the box

### Fox — the chat layer

`/msg` routes to Fox: a direct, no-personality coding assistant. Fox answers questions about the active project, suggests architecture, reviews code, and helps debug. It knows the project name, description, stack, and workspace path from the project record.

Fox does not search the web. If it doesn't have enough to answer, it says so and suggests `/forage <query>`.

### DeepFox — the research layer

`/forage` runs the full research pipeline under the hood. DeepFox is the orchestration layer — no personality, no editorializing, no injecting its own ideas into summaries. It reports what the evidence shows, flags gaps, and stops.

The pipeline:
1. **Persona-driven discovery** — 4 roles (Product Manager, Market Analyst, Project Manager, Tech Lead) each generate one search query based on the project context and research question. 4 parallel web crawls × 20 results each = 80 candidate sources.
2. **Diversity filtering** — each persona bucket is ranked by source tier × topical relevance × intra-bucket diversity. Top 8 per persona = 32 curated sources fed to the research pool.
3. **Research pool** — 4 parallel agents work the 32 sources: market analyst, technical researcher, risk researcher, execution planner.
4. **Synthesis** — a single unified narrative with [E] (evidence), [I] (inference), [S] (speculation) labels and tier-aware confidence.
5. **Skeptic pass** — adversarial critique of the synthesis before it's written to disk.
6. **Gap-fill** (if needed) — if agent confidence is low, 2 targeted agents run again on identified gaps and the synthesis is rebuilt.

Source tiers used throughout:
- **Tier 1** — official docs, language references, package registries, standards bodies, security advisories
- **Tier 2** — maintained repos, Stack Overflow, engineering blogs, established technical publishers
- **Tier 3** — community forums, tutorial sites, dev blogs, Medium posts
- **Tier 4** — cross-domain analogy sources; cannot support evidence claims

If most sources are tier 3/4, synthesis confidence is capped and a Source Quality Warning is added.

Research output is written to `Projects/<slug>/research_summaries/` and `Projects/<slug>/research_raw/`.

---

## TUI

The interface is built with [Textual](https://textual.textualize.io/). Key features:

- **Animated fox** in the banner — breathing color, occasional tail wag, hop, and turn. Phrases and intervals are configurable in `SourceCode/tui/indicator_phrases.py`.
- **Thinking indicator** — shows current pipeline stage and elapsed time. Label cycles through task-specific phrases (chat / forage / plan / build) every 5–7 seconds. During `/forage`, each research stage (agent running, web crawl, skeptic review, etc.) updates the label live.
- **Project footer** — shows `Name [slug]` at the bottom center, always visible.
- **Autocomplete** — `↑`/`↓` cycles through matching commands. `/open ` tab-completes your project slugs from the live database.
- **File path highlighting** — any existing file path in the reasoning stream is highlighted amber+underline. VS Code's terminal link provider makes them Ctrl+Clickable.

---

## Commands

### Projects

| Command | What it does |
|---|---|
| `/new greenfield <name> <description>` | Scaffold a new project. Stack is auto-detected from the description. |
| `/new import <git-url\|path> [--copy] [--name <name>]` | Import an existing codebase. |
| `/projects` | List all projects. |
| `/open <slug\|id>` | Switch active project. |

### Work

| Command | What it does |
|---|---|
| `/msg <text>` | Chat with Fox about the active project. No web access. |
| `/forage <query>` | Run the full web research pipeline. Progress streams to the TUI. |
| `/plan [prompt]` | Generate a markdown plan for the project. |
| `/execute --plan <id\|latest>` | Run a plan. |
| `/build [prompt]` | Plan + execute in one shot. |

### Research output

| Command | What it does |
|---|---|
| `/view` | Render the most recent research summary in the TUI. |
| `/view --summary` | Most recent research summary (timestamp-prefixed files only). |
| `/view --raw` | Most recent raw research notes. |
| `/view --fancy` | Open the file via `grip` in the browser for rendered markdown. |
| `/view --summary --fancy` | Summary in browser. |
| `/view --raw --fancy` | Raw notes in browser. |

### System

| Command | What it does |
|---|---|
| `/stack show` | Show the active project's current stack. |
| `/stack change <backend> <frontend> <db> <lang>` | Override the stack. |
| `/stack save <name>` | Save as a reusable custom stack. |
| `/git init\|status\|commit\|push` | Project-scoped git operations. |
| `/models` | List configured model roles. |
| `/models set <role> <model>` | Reassign a model role. |
| `/system` | Show repo root, paths, and runtime info. |
| `/help` | List all commands. |
| `/quit` | Exit. |

---

## Stack detection

`/new greenfield` detects the stack from your description:

| Description signals | Detected stack |
|---|---|
| `vue`, `react`, `svelte`, `htmx`, `angular` | Frontend set accordingly |
| `node`, `typescript`, `javascript`, `hono`, `next` | Hono + Svelte + SQLite |
| `django` | Django + htmx + Postgres |
| `flask` | Flask + htmx + SQLite |
| `desktop`, `.net`, `dotnet`, `avalonia` | Avalonia + none + none |
| `cli`, `terminal`, `tool`, `script`, `bot`, `daemon`, `simple` | No frontend |
| Default (web app) | FastAPI + htmx + SQLite |

Override anytime with `/stack change <backend> <frontend> <db> <lang>`.

Available stack components:
- **Backends**: `fastapi` `flask` `django` `express` `hono` `nextjs-api` `avalonia`
- **Frontends**: `none` `htmx` `react` `vue` `svelte` `angular`
- **Databases**: `sqlite` `postgres` `mongodb` `json-file`
- **Languages**: `python` `node` `dotnet` `go` `rust`

---

## Model configuration

All model roles are set in `config.ini` under `[models]`. Changes take effect on the next command — no restart needed.

Key roles:

| Role key | Default | Used for |
|---|---|---|
| `reynard_layer` | `qwen3:8b` | Fox chat responses (`/msg`) |
| `intent_confirmer` | `gemma3:4b` | Intent verification gate (skip list covers most commands) |
| `research_market_analyst` | `qwen3:8b` | Research pool agents |
| `research_technical` | `deepseek-r1:8b` | Research pool — technical agent |
| `synthesis_default` | `qwen3:8b` | Research synthesis |
| `synthesis_premium` | `qwen3:14b` | Escalated synthesis (skeptic-triggered) |
| `plan_deep` | `qwen2.5-coder:14b` | Deep plan generation |
| `plan_shallow` | `qwen2.5-coder:7b` | Quick plan generation |
| `execute_editor` | `qwen2.5-coder:14b` | Code execution / scaffolding |
| `embeddings` | `qwen3-embedding:4b` | Source diversity ranking |

The `parallel_agents` setting in the research pool defaults to 2 to stay within 8 GB VRAM. The 14B synthesis model is only used when the skeptic pass flags issues — it does not run on every forage.

---

## Customising the TUI

### Indicator phrases

`SourceCode/tui/indicator_phrases.py` contains all the phrase tuples shown in the thinking indicator. Each task type has 5 phrases that cycle every 5–7 seconds. Edit freely — change wording, remove phrases, add task types.

```python
FORAGE_PHRASES = (
    "foraging",
    "on the trail",
    "digging in",
    "casting the net",
    "running it down",
)
```

Interval bounds are `PHRASE_INTERVAL_MIN` / `PHRASE_INTERVAL_MAX` at the bottom of the file.

---

## Project layout

```
Projects/
  <slug>/
    research_summaries/   # Synthesis output, timestamped *.md
    research_raw/         # Full agent findings, timestamped *.md
    research_web_sources/ # Web source records
    plan/                 # Generated plans
    deliverables/         # Build output
    implementation/       # Scaffolded code
Runtime/
  conversations/          # Per-project conversation history
  projects/               # Project metadata DB
  memory/                 # Project and topic memory
SourceCode/
  tui/                    # Textual TUI app and widgets
  orchestrator/           # Main pipeline and Reynard/DeepFox logic
  agents_research/        # Research pool, synthesizer, source diversity
  agents_make/            # Stack builder, scaffolding pools
  shared_tools/           # Config, model routing, web research, stores
```

---

## Configuration

- `config.ini` is gitignored — put local tokens and per-role model assignments here.
- `config.ini.example` is the tracked template; copy and edit.
- `Projects/` is gitignored per-project on first run.
- Bots (Telegram / Discord / Slack) are optional: `pip install -r requirements-optional-bots.txt`.

---

## License

Service-Only Source-Available. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
