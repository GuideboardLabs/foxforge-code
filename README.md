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

Foxforge-code is a TUI-first fork of [Foxforge](https://github.com/GuideboardLabs/Foxforge) stripped down to coding workflows. It drops the web GUI and the general-purpose conversation layer, and replaces them with a direct coding assistant (Fox), a grounded web research pipeline (DeepFox), and a project workspace that tracks everything locally.

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

`/forage`, `/forage --domain`, and `/forage --intent <name>` run the full research pipeline under the hood. DeepFox is the orchestration layer — no personality, no editorializing, no injecting its own ideas into summaries. It reports what the evidence shows, flags gaps, and stops.

Both modes share the same pipeline skeleton. What differs is the _lens_ at each stage.

#### Technical mode — `/forage <query>` (default)

Use this when you want to understand how to build something: architectural tradeoffs, implementation patterns, libraries, risk, ecosystem maturity.

1. **Persona-driven discovery** — 4 roles (Product Manager, Market Analyst, Project Manager, Tech Lead) each generate one search query based on the project context and research question. 4 parallel web crawls × 20 results each = 80 candidate sources.
2. **Diversity filtering** — each persona bucket is ranked by source tier × topical relevance × intra-bucket diversity. Top 8 per persona = 32 curated sources fed to the research pool.
3. **Research pool** — role-routed agents work the sources via topic policy + intent policy:
   - Router inputs: `topic_type`, `research_intent`, query, project context, workspace knowledge
   - Role classes: `primary`, `advisory`, `adjudicator`
   - Gating: legal and quantitative roles are trigger-based, not unconditional
   - Animal-care routing is split from medical to avoid human-clinical drift
4. **Synthesis** → **Skeptic pass** → **Gap-fill** (if confidence is low)

#### Domain mode — `/forage --domain <query>`

Use this when you want to understand a subject area itself: what experts actually know, what users actually experience, and what authoritative resources define the field. This is the right mode for product research, content strategy, learning path design, or any question about a domain rather than an implementation.

The logic behind the split: technical sources (Stack Overflow, GitHub, engineering blogs) are noise for domain questions, and domain sources (practitioner communities, certification bodies, user forums) are noise for implementation questions. Sending the wrong agents at the wrong sources produces confident-sounding synthesis with the wrong signal.

1. **Persona-driven discovery** — 3 domain-specific roles generate queries oriented toward expert knowledge, user experience, and authoritative references rather than implementation.
2. **Diversity filtering** — same ranking as technical mode, but source tier calibration favors domain authorities over technical publishers.
3. **Research pool** — domain-intent roles prioritize:
   - `domain_practitioner_researcher`
   - `end_user_researcher`
   - `resource_scout` (advisory, claims filtered from evidence aggregation)
   - plus trigger-gated roles when relevant (safety, standards/certification, legal, quantitative)
4. **Synthesis** → **Skeptic pass** → **Gap-fill** (if confidence is low)

#### Intent-aware routing

`research_intent` is now first-class and threads through the stack:
- `general_research`
- `domain_foraging`
- `product_research`
- `technical_planning`
- `implementation_support`
- `market_research`
- `standards_research`
- `risk_research`
- `final_synthesis`

`/forage --domain` maps to `domain_foraging` by default. `/forage --intent <name>` lets you set intent explicitly.

#### Project-bound forage translation policy

In project-bound forage:
- Always translate findings to project context.
- Only translate into stack-specific implementation actions when intent is `technical_planning` or `implementation_support`.
- Otherwise technical notes are framed under **Possible Technical Implications** instead of direct Tech Lead actions.

#### Synthesis output (both modes)

The synthesizer produces a unified narrative — not a per-agent summary. Key findings are tagged:
- `[baseline]` — well-known consensus; compressed to one sentence
- `[insight]` — non-obvious, differentiating finding; given full treatment
- `[risk]` — specific failure mode or when-it-breaks scenario; given full treatment
- `[gap]` — no primary evidence found; stated explicitly

Required sections: **Executive Summary**, **Key Findings**, **Insights & Design Implications** (≥2 non-obvious items), **Failure Modes & Risks** (≥2 specific scenarios), **Differentiation Opportunity**, **Next Steps**.

Evidence labels: `[E]` (evidence — cite a source URL), `[I]` (inference), `[S]` (speculation). Convergence is reported as "N of M agents supported X" — when all agents cite the same source type, that's flagged as an echo, not independent validation.

#### Source tiers (both modes)

- **Tier 1** — official docs, language references, package registries, standards bodies, security advisories, technical specifications
- **Tier 2** — maintained repos, Stack Overflow, engineering blogs, established technical publishers, reputable practitioner communities
- **Tier 3** — community forums, tutorial sites, dev blogs, Medium posts, Reddit threads
- **Tier 4** — cross-domain analogy sources; cannot support evidence claims

If most sources are tier 3/4, synthesis confidence is capped and a Source Quality Warning is added.

Research output is written to `Projects/<slug>/research_summaries/` and `Projects/<slug>/research_raw/`.
Additional artifacts now include:
- `<summary>.primitives.json` for domain primitives (`milestones`, `success_criteria`, `failure_modes`, `measurement_dimensions`)
- `domain_summary` (domain forage runs)
- `project_summary` and `implementation_brief` (technical planning runs)

#### Quality controls and guardrails

- **Skeptic sidecar**: adversarial critique/revise pass after synthesis.
- Genericity/usefulness gate: scores artifacts and triggers at most one rewrite when outputs are too generic.
- Recommendation-strength labels on actions: `implement_now`, `prototype`, `design_option`, `future_experiment`, `weak_do_not_prioritize`, `reject`.
- **Public-content guardrail**: unsafe/public-content requests are filtered before synthesis output is finalized.
- Scaffold/documentation system references include **Canon v1** and optional **runtime smoke** validation paths.

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
| `/forage <query>` | Run the full research pipeline — technical mode. Targets implementation, architecture, and ecosystem questions. |
| `/forage --domain <query>` | Run the full research pipeline — domain mode. Targets expert knowledge, user experience, and authoritative resources rather than implementation. |
| `/forage --intent <name> <query>` | Run forage with explicit intent routing (for example `technical_planning`, `implementation_support`, `domain_foraging`). |
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

If a request is treated as a system-level stack decision, re-evaluate from a **Technical topic** before re-running research.

Current scaffolding includes system-fixed tracks for `Flask 3.x + Vue 3.5` and `.NET 8 + Avalonia` alongside other templates. These are referred to as `system-fixed` stacks in internal docs and tests.

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
| `research_market_analyst` | `qwen3:8b` | Research pool baseline lane config |
| `research_technical` | `deepseek-r1:8b` | Legacy technical-role compatibility |
| `research_critical_analyst` | `deepseek-r1:8b` | Adjudicator role (in-pool) |
| `research_critical_analyst_premium` | `hengwen/DeepSeek-R1-Distill-Qwen-14B:q4_k_m` | Premium escalation for critical analyst |
| `research_contrarian_red_team_premium` | `hengwen/DeepSeek-R1-Distill-Qwen-14B:q4_k_m` | Premium escalation for contrarian/red-team |
| `synthesis_default` | `qwen3:8b` | Research synthesis |
| `synthesis_premium` | `hengwen/DeepSeek-R1-Distill-Qwen-14B:q4_k_m` | Escalated synthesis (skeptic-triggered) |
| `research_genericity_gate` | `qwen3:8b` | Genericity/usefulness scoring and rewrite gating |
| `plan_deep` | `qwen3:14b` | Deep plan generation |
| `plan_shallow` | `qwen3:8b` | Quick plan generation |
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
