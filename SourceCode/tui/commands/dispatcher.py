from __future__ import annotations

import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "SourceCode"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from orchestrator.main import FoxforgeOrchestrator
from shared_tools.config_ini import load_config
from shared_tools.conversation_store import ConversationStore
from shared_tools.git_service import GitServiceError
from shared_tools import git_service
from shared_tools.plan_store import PlanStore
from shared_tools.project_engine import ProjectEngine
from shared_tools.project_importer import ProjectImportError, import_from_git, import_from_path
from shared_tools.stack_catalog import PREBUILT_STACKS, derive_project_type, save_custom_stack
from shared_tools.stack_recommender import recommend_detect, recommend_greenfield
from agents_make.stack_builder_pool import build as build_stack


_FORAGE_STAGE_LABELS: dict[str, str] = {
    "orchestrator_received":   "routing",
    "lane_routed":             "lane selected",
    "research_pool_started":   "agents starting",
    "research_agent_started":  "agent running",
    "research_agent_completed":"agent done",
    "web_research_started":    "web research starting",
    "web_source_discovered":   "source found",
    "persona_queries_planned": "persona queries planned",
    "web_stack_ready":         "web context ready",
    "numeric_validation_started": "checking numerics",
    "numeric_validation_completed": "numeric check done",
    "gap_fill_started":        "filling gaps",
    "gap_fill_completed":      "gaps filled",
    "skeptic_pass_started":    "skeptic reviewing",
    "skeptic_pass_completed":  "review done",
    "translation_chain_started": "project implications running",
    "translation_chain_completed": "project implications ready",
    "stack_builder_started":   "building output",
    "stack_builder_done":      "synthesis complete",
    "research_cancel_requested": "cancelling",
    "foraging_paused":         "paused",
}

_FOX_PERSONA = """\
You are Fox, the coding assistant for Foxforge-code.
The orchestration layer that runs research and builds behind you is called DeepFox.

You help developers build software. That is your only job here.

Style:
- Direct and concise. No preamble, no filler, no sign-off.
- Answer the question asked. If something needs a decision, say which option you'd pick and why in one sentence.
- When you don't know something, say so plainly.
- No personality performance — no wit, no warmth for its own sake, no theatrical responses.
- Short answers for short questions. Long answers only when the complexity earns it.
- Write code when code is the right answer. Explain when explanation is the right answer.

Scope:
- You help with planning, architecture, implementation, debugging, and code review for the active project.
- You do not fetch live data or search the web. If you don't have enough to answer, say so and suggest the user run /forage <query> to search for it.
- You do not do general chat, life advice, or anything outside of building software.
- If asked something outside that scope, redirect once, briefly.\
"""


@dataclass(slots=True)
class DispatcherState:
    active_project_slug: str = ""
    user_id: str = ""


@dataclass(slots=True)
class CommandOutcome:
    text: str
    error: bool = False
    active_project_slug: str = ""
    render_md: bool = False


class CommandDispatcher:
    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root)
        self.project_engine = ProjectEngine(self.repo_root)
        self.plan_store = PlanStore(self.repo_root)

    def dispatch(self, raw: str, state: DispatcherState, progress_fn=None) -> CommandOutcome:
        text = str(raw or "").strip()
        if not text:
            return CommandOutcome("", active_project_slug=state.active_project_slug)

        if not text.startswith("/"):
            return self._handle_msg(text, state)

        try:
            argv = shlex.split(text)
        except ValueError as exc:
            return CommandOutcome(f"Command parse error: {exc}", error=True, active_project_slug=state.active_project_slug)

        command = argv[0].lower()
        args = argv[1:]

        try:
            if command == "/help":
                return CommandOutcome(self._help_text(), active_project_slug=state.active_project_slug)
            if command == "/new":
                return self._cmd_new(args, state)
            if command == "/import":
                return self._cmd_import(args, state)
            if command == "/projects":
                return self._cmd_projects(state)
            if command == "/open":
                return self._cmd_open(args, state)
            if command == "/msg":
                return self._handle_msg(" ".join(args), state)
            if command == "/forage":
                return self._cmd_forage(args, state, progress_fn=progress_fn)
            if command == "/view":
                return self._cmd_view(args, state)
            if command == "/plan":
                return self._cmd_plan(args, state)
            if command == "/execute":
                return self._cmd_execute(args, state)
            if command == "/build":
                return self._cmd_build(args, state)
            if command == "/stack":
                return self._cmd_stack(args, state)
            if command == "/git":
                return self._cmd_git(args, state)
            if command == "/models":
                return self._cmd_models(args, state)
            if command == "/system":
                return self._cmd_system(state)
            if command in {"/pending", "/lessons", "/library", "/reflections", "/cards"}:
                return CommandOutcome(f"{command} is available as a read-only TUI panel in a future pass.", active_project_slug=state.active_project_slug)
            if command == "/quit":
                return CommandOutcome("QUIT", active_project_slug=state.active_project_slug)
        except Exception as exc:
            return CommandOutcome(f"Command failed: {exc}", error=True, active_project_slug=state.active_project_slug)

        return CommandOutcome(f"Unknown command: {command}. Use /help.", error=True, active_project_slug=state.active_project_slug)

    def _help_text(self) -> str:
        return "\n".join(
            [
                "Commands:",
                "/new greenfield <name> <description>",
                "/new import <git-url|path> [--copy] [--name <name>]",
                "/import <git-url|path>",
                "/projects",
                "/open <slug|id>",
                "/msg <text>",
                "/forage <query>",
                "/plan [prompt]",
                "/plan --refresh <plan_id|latest>",
                "/execute --plan <plan_id|latest>",
                "/build [prompt]",
                "/stack show|change|save",
                "/git init|status|commit|push",
                "/models",
                "/models set <role> <name>",
                "/system",
                "/quit",
            ]
        )

    def _require_active_project(self, state: DispatcherState) -> dict[str, Any]:
        slug = str(state.active_project_slug or "").strip()
        if not slug:
            raise RuntimeError("No active project. Use /new or /open first.")
        project = self.project_engine.get_by_slug(slug)
        if project is None:
            raise RuntimeError(f"Active project not found: {slug}")
        return project

    def _create_project_conversation(self, state: DispatcherState, slug: str) -> dict[str, Any]:
        store = ConversationStore(self.repo_root, user_id=state.user_id or None)
        return store.get_or_create_for_project(slug, title=f"{slug} chat")

    def _cmd_new(self, args: list[str], state: DispatcherState) -> CommandOutcome:
        if not args:
            return CommandOutcome(
                "Usage: /new greenfield <name> <description> | /new import <git-url|path> [--copy] [--name <name>]",
                error=True,
                active_project_slug=state.active_project_slug,
            )

        mode = args[0].lower()
        if mode == "import":
            return self._cmd_import(args[1:], state)

        if mode != "greenfield":
            return CommandOutcome("/new expects 'greenfield' or 'import'.", error=True, active_project_slug=state.active_project_slug)
        if len(args) < 3:
            return CommandOutcome("Usage: /new greenfield <name> <description>", error=True, active_project_slug=state.active_project_slug)

        name = args[1]
        description = " ".join(args[2:]).strip()
        rec = recommend_greenfield(description)
        stack = dict(rec.get("recommended") or {})

        default_root = load_config(self.repo_root).get("app", "default_project_root", fallback="./Projects")
        workspace = (self.repo_root / default_root / name.replace(" ", "-").lower()).resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        project = self.project_engine.create_project(
            name=name,
            description=description,
            stack=stack,
            workspace_path=str(workspace),
        )
        build_stack(project, "Initial scaffold from /new", repo_root=self.repo_root)
        self._create_project_conversation(state, project["slug"])
        self._seed_project_memory(project["slug"], description)
        state.active_project_slug = project["slug"]
        return CommandOutcome(
            "\n".join(
                [
                    f"Created project: {project['name']} ({project['slug']})",
                    f"Workspace: {project['workspace_path']}",
                    f"Stack: {json.dumps(project['stack'])}",
                    f"Type: {project['project_type']}",
                    f"Recommendation: {rec.get('rationale', '')}",
                ]
            ),
            active_project_slug=state.active_project_slug,
        )

    def _cmd_import(self, args: list[str], state: DispatcherState) -> CommandOutcome:
        if not args:
            return CommandOutcome("Usage: /import <git-url|path> [--copy] [--name <name>]", error=True, active_project_slug=state.active_project_slug)

        source = args[0]
        mode = "attach"
        name_override = ""
        idx = 1
        while idx < len(args):
            token = args[idx]
            if token == "--copy":
                mode = "copy"
                idx += 1
                continue
            if token == "--name" and idx + 1 < len(args):
                name_override = args[idx + 1]
                idx += 2
                continue
            idx += 1

        default_root = load_config(self.repo_root).get("app", "default_project_root", fallback="./Projects")
        is_git = source.startswith("http://") or source.startswith("https://") or source.startswith("git@") or source.endswith(".git")

        try:
            if is_git:
                repo_name = source.rstrip("/").split("/")[-1].replace(".git", "") or "imported"
                target = (self.repo_root / default_root / repo_name).resolve()
                workspace = import_from_git(source, target)
                display_name = name_override or repo_name
            else:
                src_path = Path(source).expanduser().resolve()
                display_name = name_override or src_path.name
                target = (self.repo_root / default_root / display_name).resolve()
                workspace = import_from_path(src_path, target, mode=mode)
        except ProjectImportError as exc:
            return CommandOutcome(f"Import failed: {exc}", error=True, active_project_slug=state.active_project_slug)

        rec = recommend_detect(workspace)
        stack = dict(rec.get("recommended") or {})
        project = self.project_engine.create_project(
            name=display_name,
            description=f"Imported workspace from {source}",
            stack=stack,
            workspace_path=str(workspace),
        )
        self._create_project_conversation(state, project["slug"])
        self._seed_project_memory(project["slug"], f"Imported workspace from {source}. Stack: {json.dumps(stack)}.")
        state.active_project_slug = project["slug"]

        return CommandOutcome(
            "\n".join(
                [
                    f"Imported project: {project['name']} ({project['slug']})",
                    f"Workspace: {project['workspace_path']}",
                    f"Detected stack: {json.dumps(project['stack'])}",
                    f"Signals: {', '.join(rec.get('signals', []))}",
                ]
            ),
            active_project_slug=state.active_project_slug,
        )

    def _cmd_projects(self, state: DispatcherState) -> CommandOutcome:
        rows = self.project_engine.list_projects()
        if not rows:
            return CommandOutcome("No projects yet.", active_project_slug=state.active_project_slug)
        lines = []
        for row in rows:
            active = "*" if row.get("slug") == state.active_project_slug else " "
            lines.append(f"{active} {row.get('slug')} | {row.get('project_type')} | {row.get('workspace_path')}")
        return CommandOutcome("\n".join(lines), active_project_slug=state.active_project_slug)

    def _cmd_open(self, args: list[str], state: DispatcherState) -> CommandOutcome:
        if not args:
            return CommandOutcome("Usage: /open <slug|id>", error=True, active_project_slug=state.active_project_slug)
        key = args[0]
        project = self.project_engine.get_by_slug(key) or self.project_engine.get_project(key)
        if project is None:
            return CommandOutcome(f"Project not found: {key}", error=True, active_project_slug=state.active_project_slug)
        state.active_project_slug = str(project.get("slug", ""))
        self._create_project_conversation(state, state.active_project_slug)
        return CommandOutcome(
            f"Opened project {project.get('name')} ({state.active_project_slug})",
            active_project_slug=state.active_project_slug,
        )

    def _seed_project_memory(self, slug: str, description: str) -> None:
        pass  # project_context_memory is dog-domain specific; context injected directly in _handle_msg

    def _project_context_block(self, slug: str) -> str | None:
        try:
            project = self.project_engine.get_by_slug(slug)
            if not project:
                return None
            lines = [f"Active coding project: {project.get('name', slug)}"]
            desc = str(project.get("description") or "").strip()
            if desc:
                lines.append(f"Description: {desc}")
            stack = project.get("stack")
            if stack:
                lines.append(f"Stack: {json.dumps(stack)}")
            workspace = str(project.get("workspace_path") or "").strip()
            if workspace:
                lines.append(f"Workspace: {workspace}")
            return "\n".join(lines)
        except Exception:
            return None

    def _handle_msg(self, text: str, state: DispatcherState) -> CommandOutcome:
        if not text.strip():
            return CommandOutcome("", active_project_slug=state.active_project_slug)
        project = state.active_project_slug or "general"
        store = ConversationStore(self.repo_root, user_id=state.user_id or None)
        conv = store.get_or_create_for_project(project, title=f"{project} chat")
        history = list(conv.get("messages") or [])[-20:]
        store.add_message(conv["id"], "user", text, mode="talk")

        context_block = self._project_context_block(project)
        orch = FoxforgeOrchestrator(self.repo_root)
        persona = _FOX_PERSONA
        if context_block:
            persona = persona + f"\n\nActive project:\n{context_block}"
        reply = orch.conversation_reply(text, history=history, project=project, persona_override=persona, disable_web=True)
        store.add_message(conv["id"], "assistant", reply, mode="talk")
        return CommandOutcome(reply, active_project_slug=state.active_project_slug)

    def _cmd_forage(self, args: list[str], state: DispatcherState, progress_fn=None) -> CommandOutcome:
        domain_mode = "--domain" in args
        clean_args = [a for a in args if a != "--domain"]
        prompt = " ".join(clean_args).strip()
        if not prompt:
            return CommandOutcome(
                "Usage: /forage <query>  |  /forage --domain <query>",
                error=True,
                active_project_slug=state.active_project_slug,
            )
        project = state.active_project_slug or "general"
        orch = FoxforgeOrchestrator(self.repo_root)
        orch.set_project(project)

        def _progress(stage: str, detail=None) -> None:
            if not callable(progress_fn):
                return
            label = _FORAGE_STAGE_LABELS.get(stage, stage.replace("_", " "))
            extra = ""
            if isinstance(detail, dict):
                agent = detail.get("agent", "")
                if agent and stage in ("research_agent_started", "research_agent_completed"):
                    name = agent.replace("_", " ")
                    done = detail.get("completed", "")
                    total = detail.get("total", "")
                    suffix = f" {done}/{total}" if done != "" and total != "" else ""
                    verb = "done" if stage == "research_agent_completed" else "running"
                    extra = f": {name} {verb}{suffix}"
                elif detail.get("agents_total"):
                    extra = f" ({detail['agents_total']} agents)"
                elif detail.get("url") or detail.get("source"):
                    url = detail.get("url") or detail.get("source")
                    extra = f": {str(url)[:80]}"
                elif detail.get("gap_queries"):
                    extra = f" ({len(detail['gap_queries'])} queries)"
            elif isinstance(detail, str) and detail.strip():
                extra = f": {detail.strip()[:100]}"
            progress_fn(f"{label}{extra}")

        forage_profile = "domain" if domain_mode else "technical"
        out = orch.handle_message(prompt, force_research=True, forage_profile=forage_profile, progress_callback=_progress)
        return CommandOutcome(str(out or "Research completed."), active_project_slug=state.active_project_slug)

    def _cmd_view(self, args: list[str], state: DispatcherState) -> CommandOutcome:
        slug = state.active_project_slug or "general"
        flags = {a.lstrip("-") for a in args if a.startswith("--")}
        fancy = "fancy" in flags

        project = self.project_engine.get_by_slug(slug)
        workspace = Path(project["workspace_path"]) if project else None

        if not workspace or not workspace.exists():
            return CommandOutcome("No active project workspace found.", error=True, active_project_slug=state.active_project_slug)

        if "summary" in flags or not flags or flags == {"fancy"}:
            # Real summaries match *_research_summary.md — exclude library index and critique files
            summaries = sorted(
                [
                    f for f in (workspace / "research_summaries").glob("*.md")
                    if f.name.endswith("_research_summary.md") and f.name[0].isdigit()
                ],
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            ) if (workspace / "research_summaries").exists() else []
            target = summaries[0] if summaries else None
            if not target:
                target = self._latest_file(workspace)
        elif "raw" in flags:
            # Real raws match *_research_raw.md — exclude library index files
            raws = sorted(
                [
                    f for f in (workspace / "research_raw").glob("*.md")
                    if f.name.endswith("_research_raw.md")
                ],
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            ) if (workspace / "research_raw").exists() else []
            target = raws[0] if raws else None
        else:
            target = self._latest_file(workspace)

        if not target:
            return CommandOutcome("No files found for this project.", error=True, active_project_slug=state.active_project_slug)

        if fancy:
            from SourceCode.tui.widgets.reasoning_stream import _grip_port
            import subprocess as _sp
            port = _grip_port(str(target))
            import time as _time; _time.sleep(0.8)  # let grip bind before browser hits it
            _sp.Popen(["xdg-open", f"http://localhost:{port}"],
                      stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
            return CommandOutcome(
                f"Opened {target.name} in browser → http://localhost:{port}",
                active_project_slug=state.active_project_slug,
            )

        # Render in TUI
        try:
            content = target.read_text(encoding="utf-8")
        except OSError as exc:
            return CommandOutcome(f"Could not read {target}: {exc}", error=True, active_project_slug=state.active_project_slug)

        return CommandOutcome(
            f"**{target.name}**\n\n{content}",
            active_project_slug=state.active_project_slug,
            render_md=True,
        )

    def _latest_file(self, workspace: Path) -> Path | None:
        """Most recently modified file anywhere in the project workspace."""
        try:
            files = [f for f in workspace.rglob("*") if f.is_file()]
            return max(files, key=lambda f: f.stat().st_mtime) if files else None
        except Exception:
            return None

    def _generate_plan_markdown(self, prompt: str, *, kind: str) -> str:
        cfg = load_config(self.repo_root)
        model_role = "plan_deep" if kind == "deep" else "plan_shallow"
        model = cfg.get_model(model_role)
        system = (
            "You are a coding planner. Produce concise markdown with sections: Objective, "
            "Steps, Risks, and Verification."
        )
        user = prompt.strip() or "Generate a plan for the active coding project."
        try:
            orch = FoxforgeOrchestrator(self.repo_root)
            text = orch.ollama.chat(model, system, user, temperature=0.2, num_ctx=8192, think=True)
            if text.strip():
                return text.strip()
        except Exception:
            pass
        return "\n".join([
            "# Plan",
            "",
            f"Prompt: {user}",
            "",
            "## Objective",
            "Implement the requested coding change safely.",
            "",
            "## Steps",
            "1. Inspect relevant files.",
            "2. Apply code edits.",
            "3. Run import/test checks.",
            "",
            "## Risks",
            "- Stale assumptions if workspace changed.",
            "",
            "## Verification",
            "- Run the relevant command and confirm output.",
        ])

    def _resolve_plan_ref(self, project_slug: str, ref: str) -> Any:
        key = str(ref or "latest").strip().lower() or "latest"
        if key == "latest":
            return self.plan_store.latest_plan(project_slug)
        return self.plan_store.get_plan(key)

    def _cmd_plan(self, args: list[str], state: DispatcherState) -> CommandOutcome:
        project = self._require_active_project(state)
        project_slug = str(project["slug"])
        workspace = str(project.get("workspace_path") or "")

        if args and args[0] == "--refresh":
            ref = args[1] if len(args) > 1 else "latest"
            old = self._resolve_plan_ref(project_slug, ref)
            if old is None:
                return CommandOutcome(f"Plan not found: {ref}", error=True, active_project_slug=state.active_project_slug)
            prompt = old.prompt or old.body_md[:300]
            body = self._generate_plan_markdown(prompt, kind=old.kind or "deep")
            new_id = self.plan_store.create_plan(project_slug, old.kind or "deep", prompt, body, workspace_path=workspace)
            self.plan_store.supersede(old.id, new_id)
            return CommandOutcome(f"Refreshed plan {old.id} -> {new_id}", active_project_slug=state.active_project_slug)

        prompt = " ".join(args).strip() or f"Plan next coding work for project {project_slug}."
        body = self._generate_plan_markdown(prompt, kind="deep")
        plan_id = self.plan_store.create_plan(project_slug, "deep", prompt, body, workspace_path=workspace)
        return CommandOutcome(f"Created plan {plan_id}\n\n{body}", active_project_slug=state.active_project_slug)

    def _cmd_execute(self, args: list[str], state: DispatcherState) -> CommandOutcome:
        project = self._require_active_project(state)
        project_slug = str(project["slug"])
        workspace = str(project.get("workspace_path") or "")

        ref = "latest"
        if args and args[0] == "--plan" and len(args) > 1:
            ref = args[1]
        plan = self._resolve_plan_ref(project_slug, ref)
        if plan is None:
            return CommandOutcome(f"Plan not found: {ref}", error=True, active_project_slug=state.active_project_slug)

        if self.plan_store.is_stale(plan, workspace):
            return CommandOutcome(
                f"Plan {plan.id} is stale. Run /plan --refresh {plan.id} or /plan --refresh latest.",
                error=True,
                active_project_slug=state.active_project_slug,
            )

        result = build_stack(project, plan.body_md, repo_root=self.repo_root)
        summary = f"{result.message}\nPath: {result.path}\nFiles: {len(result.files_written)}"
        self.plan_store.mark_executed(plan.id, summary)
        return CommandOutcome(
            "\n".join(
                [
                    f"Executed {plan.id}",
                    summary,
                    "Tip: run /git status and /git commit when ready.",
                ]
            ),
            active_project_slug=state.active_project_slug,
        )

    def _cmd_build(self, args: list[str], state: DispatcherState) -> CommandOutcome:
        project = self._require_active_project(state)
        project_slug = str(project["slug"])
        workspace = str(project.get("workspace_path") or "")
        prompt = " ".join(args).strip() or f"Build next increment for project {project_slug}."

        body = self._generate_plan_markdown(prompt, kind="shallow")
        plan_id = self.plan_store.create_plan(project_slug, "shallow", prompt, body, workspace_path=workspace)
        execute_out = self._cmd_execute(["--plan", plan_id], state)
        if execute_out.error:
            return execute_out
        return CommandOutcome(f"Created shallow plan {plan_id} and executed it.\n\n{execute_out.text}", active_project_slug=state.active_project_slug)

    def _cmd_stack(self, args: list[str], state: DispatcherState) -> CommandOutcome:
        project = self._require_active_project(state)
        if not args or args[0] == "show":
            stacks = [f"- {s.name}: {s.backend}/{s.frontend}/{s.database}/{s.language}" for s in PREBUILT_STACKS]
            return CommandOutcome(
                "\n".join(
                    [
                        f"Current stack: {json.dumps(project.get('stack', {}))}",
                        "Prebuilt stacks:",
                        *stacks,
                    ]
                ),
                active_project_slug=state.active_project_slug,
            )

        action = args[0]
        if action == "change":
            if len(args) < 5:
                return CommandOutcome("Usage: /stack change <backend> <frontend> <database> <language>", error=True, active_project_slug=state.active_project_slug)
            new_stack = {
                "backend": args[1].lower(),
                "frontend": args[2].lower(),
                "database": args[3].lower(),
                "language": args[4].lower(),
            }
            updated = self.project_engine.update_project(project["id"], stack=new_stack)
            return CommandOutcome(
                f"Updated stack: {json.dumps(updated.get('stack', {}))}\nType: {derive_project_type(updated.get('stack', {}))}",
                active_project_slug=state.active_project_slug,
            )

        if action == "save":
            if len(args) < 2:
                return CommandOutcome("Usage: /stack save <name>", error=True, active_project_slug=state.active_project_slug)
            saved = save_custom_stack(self.repo_root, name=args[1], notes=f"saved from {state.active_project_slug}", **dict(project.get("stack") or {}))
            return CommandOutcome(f"Saved custom stack {saved.get('name')}", active_project_slug=state.active_project_slug)

        return CommandOutcome("Usage: /stack show|change|save", error=True, active_project_slug=state.active_project_slug)

    def _project_workspace(self, state: DispatcherState) -> Path:
        project = self._require_active_project(state)
        workspace = Path(str(project.get("workspace_path") or "")).expanduser()
        if not workspace.is_absolute():
            workspace = (self.repo_root / workspace).resolve()
        return workspace

    def _cmd_git(self, args: list[str], state: DispatcherState) -> CommandOutcome:
        if not args:
            return CommandOutcome("Usage: /git init|status|commit|push", error=True, active_project_slug=state.active_project_slug)
        workspace = self._project_workspace(state)
        action = args[0]

        try:
            if action == "init":
                out = git_service.init(workspace)
                return CommandOutcome(out["message"], active_project_slug=state.active_project_slug)
            if action == "status":
                out = git_service.status(workspace)
                return CommandOutcome("\n".join(out["lines"]) or "clean", active_project_slug=state.active_project_slug)
            if action == "commit":
                message = " ".join(args[1:]).strip()
                if not message:
                    latest = self.plan_store.latest_plan(state.active_project_slug)
                    message = git_service.suggest_commit_message(workspace, recent_plan=latest.body_md if latest else "")
                out = git_service.commit(workspace, message, repo_root=self.repo_root)
                return CommandOutcome(out["message"], active_project_slug=state.active_project_slug)
            if action == "push":
                out = git_service.push(workspace)
                return CommandOutcome(out["message"], active_project_slug=state.active_project_slug)
        except GitServiceError as exc:
            return CommandOutcome(f"Git error: {exc}", error=True, active_project_slug=state.active_project_slug)

        return CommandOutcome("Usage: /git init|status|commit|push", error=True, active_project_slug=state.active_project_slug)

    def _cmd_models(self, args: list[str], state: DispatcherState) -> CommandOutcome:
        cfg = load_config(self.repo_root)
        if not args:
            rows = cfg.section("models")
            lines = [f"{k} = {v}" for k, v in sorted(rows.items())]
            return CommandOutcome("\n".join(lines), active_project_slug=state.active_project_slug)

        if args[0] == "set" and len(args) >= 3:
            role = args[1]
            model_name = " ".join(args[2:]).strip()
            cfg.set_model(role, model_name)
            cfg.save()
            return CommandOutcome(f"Updated model role {role} -> {model_name}", active_project_slug=state.active_project_slug)

        return CommandOutcome("Usage: /models or /models set <role> <name>", error=True, active_project_slug=state.active_project_slug)

    def _cmd_system(self, state: DispatcherState) -> CommandOutcome:
        cfg = load_config(self.repo_root)
        lines = [
            f"repo_root: {self.repo_root}",
            f"active_project: {state.active_project_slug or '(none)'}",
            f"ollama_base_url: {cfg.get('app', 'ollama_base_url', fallback='http://127.0.0.1:11434')}",
        ]
        models = cfg.section("models")
        lines.append(f"model_roles: {len(models)} configured")
        return CommandOutcome("\n".join(lines), active_project_slug=state.active_project_slug)


__all__ = ["CommandDispatcher", "CommandOutcome", "DispatcherState"]
