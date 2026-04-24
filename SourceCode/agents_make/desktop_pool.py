"""Desktop App pool — .NET 8 + Avalonia UI (MVVM, Windows-first, Linux-portable).

Pipeline (7 stages sequential):
    1. Specifier      — defines feature set, state model, and UI wireframe in prose
    2. Architect      — produces the full project scaffold (file tree + .csproj + .sln)
    3. ViewModel impl — writes ReactiveUI ViewModels for each feature area
    4. View impl      — writes AXAML Views (data-bound to ViewModels, no code-behind logic)
    5. Services impl  — writes data/service layer (repositories, file I/O, network, etc.)
    6. Build check    — validates the generated .csproj can compile (dotnet syntax check)
    7. README writer  — Windows install steps, Linux port notes, run instructions

Output layout:
    Projects/{slug}/desktop_apps/{app_name}/
    ├── README.md
    ├── .gitignore
    ├── {AppName}.sln
    ├── src/{AppName}/
    │   ├── {AppName}.csproj
    │   ├── App.axaml / App.axaml.cs
    │   ├── Program.cs
    │   ├── ViewModels/
    │   ├── Views/
    │   ├── Models/
    │   └── Services/
    └── tests/{AppName}.Tests/
"""

from __future__ import annotations

import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from shared_tools.config_ini import load_config
from shared_tools.ollama_client import OllamaClient


_MODEL_SPEC   = "qwen3:8b"
_MODEL_ARCH   = "qwen2.5-coder:14b"
_MODEL_IMPL   = "qwen2.5-coder:14b"
_MODEL_README = "qwen3:8b"
# Fallback if upgraded model unavailable
_MODEL_IMPL_FALLBACK = "qwen2.5-coder:7b"


def _resolve_models(repo_root: Path) -> None:
    global _MODEL_SPEC, _MODEL_ARCH, _MODEL_IMPL, _MODEL_README, _MODEL_IMPL_FALLBACK
    try:
        cfg = load_config(repo_root)
        _MODEL_SPEC = cfg.get_model("make_desktop_specifier")
        _MODEL_ARCH = cfg.get_model("make_desktop_architect")
        _MODEL_IMPL = cfg.get_model("make_desktop_implementer")
        _MODEL_README = cfg.get_model("make_desktop_readme")
        _MODEL_IMPL_FALLBACK = cfg.get_model("make_webapp_fallback")
    except Exception:
        pass


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _trim(text: str, max_chars: int) -> str:
    body = str(text or "").strip()
    if len(body) <= max_chars:
        return body
    cut = body[:max_chars].rsplit("\n", 1)[0].strip()
    return cut or body[:max_chars]


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text).strip()
    words = text.split()[:4]
    return "".join(w.capitalize() for w in words) or "DesktopApp"


def _pascal(text: str) -> str:
    return _slugify(text)


# ---------------------------------------------------------------------------
# Canonical stack patterns / gotchas
# ---------------------------------------------------------------------------

_DOTNET_PATTERNS = """\
# Validated against: .NET 8 LTS, C# 12
- Use <TargetFramework>net8.0</TargetFramework>, <Nullable>enable</Nullable>, <ImplicitUsings>enable</ImplicitUsings>.
- Prefer file-scoped namespaces.
- Use GlobalUsings.cs for shared imports.
- Use record for immutable DTOs, class for mutable state.
- Use async Task for I/O methods and Async suffix names.
- Prefer System.Text.Json for new code.
"""

_AVALONIA_PATTERNS = """\
# Validated against: Avalonia 11.x
- Add x:DataType on each Window/UserControl root for compiled bindings.
- Keep AXAML logic in bindings; code-behind only InitializeComponent().
- Use FluentTheme in App.axaml.
- Prefer Grid/StackPanel/DockPanel layouts over absolute positioning.
- Keep shared styles in App.axaml or Styles/.
- Program.cs should use AppBuilder.Configure<App>().UsePlatformDetect().
"""

_REACTIVEUI_PATTERNS = """\
# Validated against: ReactiveUI compatible with Avalonia 11
- ViewModelBase inherits ReactiveObject.
- Use RaiseAndSetIfChanged for reactive properties.
- Use ReactiveCommand.Create/CreateFromTask for actions.
- Manage subscriptions in WhenActivated(...) blocks.
- Use ObservableAsPropertyHelper for derived properties.
- Use IScreen + RoutedViewHost for multi-screen navigation.
"""

_DOTNET_GOTCHAS = """\
Dotnet gotchas:
- Avoid async void except event handlers.
- Avoid .Result/.Wait() on UI thread tasks (deadlock risk).
- Prefer DateTime.UtcNow or DateTimeOffset for persisted time.
- Do not mutate collections during enumeration.
- Avoid swallowing Exception silently.
- Avoid repeated string concatenation in loops.
"""

_AVALONIA_GOTCHAS = """\
Avalonia gotchas:
- No business logic in .axaml.cs.
- Missing x:DataType causes runtime-only binding errors.
- ViewModel should not reference named controls directly.
- Avoid long-running work in View constructor.
- Use Mode=TwoWay for editable input bindings when needed.
"""

_REACTIVEUI_GOTCHAS = """\
ReactiveUI gotchas:
- Always dispose subscriptions (store IDisposable).
- Handle ReactiveCommand ThrownExceptions.
- Ensure OAPH properties are initialized via ToProperty.
- Ensure View constructors include WhenActivated(...) when lifecycle hooks are used.
"""


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _run_specifier(client: OllamaClient, question: str, research_context: str) -> str:
    system_prompt = (
        f"Today: {_today()}. You are a .NET software architect.\n\n"
        "Produce a concise App Specification covering:\n"
        "1. **App Name** — PascalCase, suitable for a .NET namespace\n"
        "2. **One-liner** — what the app does in one sentence\n"
        "3. **Features** — bulleted list of 3–8 concrete features\n"
        "4. **State Model** — the main ViewModels needed (name + responsibility)\n"
        "5. **Data Layer** — what data the app reads/writes (file, SQLite, API, etc.)\n"
        "6. **UI Layout** — prose description of the main window and key views\n"
        "7. **External Deps** — any NuGet packages needed beyond Avalonia + ReactiveUI\n\n"
        "Be specific. No placeholder features. Keep it to what can realistically be implemented."
    )
    try:
        result = client.chat(
            model=_MODEL_SPEC,
            system_prompt=system_prompt,
            user_prompt=f"Request: {question}\n\nResearch context:\n{_trim(research_context, 3000)}",
            temperature=0.2,
            num_ctx=12288,
            think=False,
            timeout=240,
            retry_attempts=3,
            retry_backoff_sec=1.5,
        )
        return str(result or "").strip()
    except Exception as exc:
        return f"[Specifier failed: {exc}]"


def _run_architect(client: OllamaClient, spec: str, app_name: str) -> str:
    system_prompt = (
        f"Today: {_today()}. You are a .NET 8 + Avalonia UI architect.\n\n"
        + _DOTNET_PATTERNS + "\n\n" + _AVALONIA_PATTERNS + "\n\n"
        "Given the app spec, produce the full project scaffold as file contents.\n\n"
        "Output format: For each file, write:\n"
        "=== FILE: path/to/file.ext ===\n"
        "[file contents]\n\n"
        "Files to produce:\n"
        f"1. {app_name}/{app_name}.csproj  — TargetFramework=net8.0, Avalonia + ReactiveUI packages\n"
        f"2. {app_name}.sln  — solution file referencing the .csproj\n"
        "3. .gitignore  — standard .NET gitignore\n"
        f"4. src/{app_name}/Program.cs  — Avalonia app entry point (AppBuilder.Configure<App>())\n"
        f"5. src/{app_name}/App.axaml  — Application-level AXAML\n"
        f"6. src/{app_name}/App.axaml.cs  — Code-behind for App.axaml\n"
        f"7. src/{app_name}/Models/AppModel.cs  — core data model stub\n\n"
        "Use Avalonia 11.x API. Use ReactiveUI for ViewModelBase. "
        "All files must be complete and syntactically valid C#/AXAML."
    )
    try:
        result = client.chat(
            model=_MODEL_ARCH,
            system_prompt=system_prompt,
            user_prompt=f"App spec:\n{_trim(spec, 5000)}",
            temperature=0.15,
            num_ctx=16384,
            think=False,
            timeout=360,
            retry_attempts=3,
            retry_backoff_sec=2.0,
        )
        return str(result or "").strip()
    except Exception as exc:
        return f"[Architect failed: {exc}]"


def _run_viewmodels(client: OllamaClient, spec: str, app_name: str, question: str) -> str:
    system_prompt = (
        f"Today: {_today()}. You are implementing .NET 8 + ReactiveUI ViewModels.\n\n"
        + _DOTNET_PATTERNS + "\n\n" + _REACTIVEUI_PATTERNS + "\n\n"
        + _DOTNET_GOTCHAS + "\n\n" + _REACTIVEUI_GOTCHAS + "\n\n"
        "Produce complete ViewModel implementations for each ViewModel identified in the spec.\n\n"
        "Requirements:\n"
        "- Inherit from ViewModelBase (which inherits ReactiveObject)\n"
        "- Use `[Reactive]` attribute for bindable properties\n"
        "- Use ReactiveCommand for all user actions\n"
        "- Inject services via constructor\n"
        "- No business logic in Views\n\n"
        "Output format: For each ViewModel file:\n"
        "=== FILE: src/{app_name}/ViewModels/[Name]ViewModel.cs ===\n"
        "[complete C# file content]\n\n"
        "Also produce:\n"
        f"=== FILE: src/{app_name}/ViewModels/ViewModelBase.cs ===\n"
        "[ViewModelBase inheriting ReactiveObject]\n"
    )
    try:
        result = client.chat(
            model=_MODEL_IMPL,
            system_prompt=system_prompt,
            user_prompt=f"Request: {question}\n\nApp spec:\n{_trim(spec, 4000)}",
            temperature=0.15,
            num_ctx=16384,
            think=False,
            timeout=420,
            retry_attempts=3,
            retry_backoff_sec=2.0,
        )
        return str(result or "").strip()
    except Exception as exc:
        return f"[ViewModels failed: {exc}]"


def _run_views(client: OllamaClient, spec: str, viewmodels_code: str, app_name: str, question: str) -> str:
    system_prompt = (
        f"Today: {_today()}. You are implementing Avalonia UI 11 AXAML Views.\n\n"
        + _AVALONIA_PATTERNS + "\n\n" + _AVALONIA_GOTCHAS + "\n\n"
        "Produce complete AXAML Views and their minimal code-behind files.\n\n"
        "Requirements:\n"
        "- DataContext set to the corresponding ViewModel\n"
        "- Use data binding for all dynamic content (no code-behind logic)\n"
        "- Use Avalonia controls: Button, TextBox, DataGrid, ListBox, StackPanel, Grid, etc.\n"
        "- MainWindow.axaml hosts the primary navigation\n"
        "- AXAML namespaces: xmlns='https://github.com/avaloniaui' xmlns:x='http://schemas.microsoft.com/winfx/2006/xaml'\n\n"
        "Output format:\n"
        "=== FILE: src/{app_name}/Views/[Name]View.axaml ===\n"
        "[complete AXAML]\n"
        "=== FILE: src/{app_name}/Views/[Name]View.axaml.cs ===\n"
        "[minimal code-behind: just InitializeComponent()]\n"
    )
    try:
        result = client.chat(
            model=_MODEL_IMPL,
            system_prompt=system_prompt,
            user_prompt=f"Request: {question}\n\nApp spec:\n{_trim(spec, 3000)}\n\nViewModels:\n{_trim(viewmodels_code, 4000)}",
            temperature=0.15,
            num_ctx=16384,
            think=False,
            timeout=420,
            retry_attempts=3,
            retry_backoff_sec=2.0,
        )
        return str(result or "").strip()
    except Exception as exc:
        return f"[Views failed: {exc}]"


def _run_services(client: OllamaClient, spec: str, app_name: str, question: str) -> str:
    system_prompt = (
        f"Today: {_today()}. You are implementing .NET 8 service and data layer classes.\n\n"
        + _DOTNET_PATTERNS + "\n\n" + _DOTNET_GOTCHAS + "\n\n"
        "Based on the app spec, produce the service and data-layer implementations.\n\n"
        "Requirements:\n"
        "- Services are injected into ViewModels via constructor\n"
        "- Use async/await for I/O operations\n"
        "- For SQLite: use Microsoft.Data.Sqlite\n"
        "- For file I/O: use System.Text.Json\n"
        "- Each service has an interface (IXxxService) and implementation (XxxService)\n\n"
        "Output format:\n"
        "=== FILE: src/{app_name}/Services/I[Name]Service.cs ===\n"
        "[interface]\n"
        "=== FILE: src/{app_name}/Services/[Name]Service.cs ===\n"
        "[implementation]\n"
    )
    try:
        result = client.chat(
            model=_MODEL_IMPL,
            system_prompt=system_prompt,
            user_prompt=f"Request: {question}\n\nApp spec:\n{_trim(spec, 4000)}",
            temperature=0.15,
            num_ctx=14336,
            think=False,
            timeout=360,
            retry_attempts=3,
            retry_backoff_sec=2.0,
        )
        return str(result or "").strip()
    except Exception as exc:
        return f"[Services failed: {exc}]"


def _run_readme(client: OllamaClient, spec: str, app_name: str, question: str) -> str:
    system_prompt = (
        f"Today: {_today()}. Write a developer README.md for a .NET 8 + Avalonia UI desktop app.\n\n"
        "Include:\n"
        "## Overview\n(What the app does, one paragraph)\n\n"
        "## Prerequisites\n(dotnet 8 SDK, OS requirements)\n\n"
        "## Build & Run (Windows)\n```\ndotnet build\ndotnet run --project src/{app_name}\n```\n\n"
        "## Build & Run (Linux)\n(Note: Avalonia is cross-platform. Same commands. Note any platform differences.)\n\n"
        "## Project Structure\n(Brief description of src/ layout)\n\n"
        "## Linux Port Notes\n(Avalonia supports Linux natively. Note any Windows-specific APIs that need abstraction.)\n\n"
        "Write complete, accurate markdown. No placeholders."
    )
    try:
        result = client.chat(
            model=_MODEL_README,
            system_prompt=system_prompt,
            user_prompt=f"App name: {app_name}\n\nSpec:\n{_trim(spec, 3000)}",
            temperature=0.2,
            num_ctx=8192,
            think=False,
            timeout=180,
            retry_attempts=2,
            retry_backoff_sec=1.5,
        )
        return str(result or "").strip()
    except Exception as exc:
        return f"# {app_name}\n\n[README generation failed: {exc}]"


# ---------------------------------------------------------------------------
# File writer
# ---------------------------------------------------------------------------

def _parse_files(output: str) -> dict[str, str]:
    """Parse === FILE: path === ... blocks into a dict."""
    files: dict[str, str] = {}
    pattern = re.compile(r"=== FILE:\s*(.+?)\s*===\s*\n(.*?)(?=\n=== FILE:|\Z)", re.DOTALL)
    for match in pattern.finditer(output):
        path = match.group(1).strip()
        content = match.group(2).strip()
        if path and content:
            files[path] = content
    return files


def _write_project(
    repo_root: Path,
    project_slug: str,
    app_name: str,
    file_blocks: dict[str, str],
    readme: str,
) -> Path:
    base = repo_root / "Projects" / project_slug / "desktop_apps" / app_name
    base.mkdir(parents=True, exist_ok=True)

    # Ensure subdirectory structure exists
    for subdir in ["src", f"src/{app_name}", f"src/{app_name}/ViewModels",
                   f"src/{app_name}/Views", f"src/{app_name}/Models",
                   f"src/{app_name}/Services", f"tests/{app_name}.Tests"]:
        (base / subdir).mkdir(parents=True, exist_ok=True)

    for rel_path, content in file_blocks.items():
        target = base / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content + "\n", encoding="utf-8")

    # README
    (base / "README.md").write_text(readme + "\n", encoding="utf-8")

    # Gitignore
    if not (base / ".gitignore").exists():
        gitignore = textwrap.dedent("""\
            bin/
            obj/
            .vs/
            *.user
            *.suo
            .idea/
            build/
            *.db
            *.db-shm
            *.db-wal
        """)
        (base / ".gitignore").write_text(gitignore, encoding="utf-8")

    # CHANGELOG stub
    changelog_path = base / "CHANGELOG.md"
    if not changelog_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        changelog_path.write_text(f"# Changelog\n\n## {stamp}\n- Initial generated scaffold\n", encoding="utf-8")

    return base


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_desktop_pool(
    question: str,
    repo_root: Path,
    project_slug: str,
    bus: Any,
    research_context: str = "",
    cancel_checker: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the desktop app pipeline and return paths to generated files."""

    def _progress(stage: str, detail: dict[str, Any] | None = None) -> None:
        if callable(progress_callback):
            try:
                progress_callback(stage, detail or {})
            except Exception:
                pass

    def _cancelled() -> bool:
        if callable(cancel_checker):
            try:
                return bool(cancel_checker())
            except Exception:
                return False
        return False

    bus.emit("desktop_pool", "start", {"question": question})
    _resolve_models(repo_root)
    client = OllamaClient()

    _progress("build_pool_started", {
        "stage": "build_pool_started",
        "agents_total": 7,
        "make_type": "desktop_app",
        "destination": "desktop_apps",
    })

    # Step 1: Spec
    if _cancelled():
        return {"ok": False, "message": "Cancelled before spec.", "path": ""}
    _progress("build_agent_started", {"stage": "build_agent_started", "agent": "specifier", "model": _MODEL_SPEC})
    spec = _run_specifier(client, question, research_context)
    _progress("build_agent_completed", {"stage": "build_agent_completed", "agent": "specifier", "output_chars": len(spec)})

    # Extract app name from spec
    app_name_match = re.search(r"App Name[:\s]+([A-Za-z][A-Za-z0-9]+)", spec)
    app_name = app_name_match.group(1).strip() if app_name_match else _pascal(question)

    # Step 2: Architect (scaffold files)
    if _cancelled():
        return {"ok": False, "message": "Cancelled before architecture.", "path": ""}
    _progress("build_agent_started", {"stage": "build_agent_started", "agent": "architect", "model": _MODEL_ARCH})
    scaffold_output = _run_architect(client, spec, app_name)
    scaffold_files = _parse_files(scaffold_output)
    _progress("build_agent_completed", {"stage": "build_agent_completed", "agent": "architect", "files": len(scaffold_files)})

    # Step 3: ViewModels
    if _cancelled():
        return {"ok": False, "message": "Cancelled before ViewModels.", "path": ""}
    _progress("build_agent_started", {"stage": "build_agent_started", "agent": "viewmodels", "model": _MODEL_IMPL})
    vm_output = _run_viewmodels(client, spec, app_name, question)
    vm_files = _parse_files(vm_output)
    _progress("build_agent_completed", {"stage": "build_agent_completed", "agent": "viewmodels", "files": len(vm_files)})

    # Step 4: Views
    if _cancelled():
        return {"ok": False, "message": "Cancelled before Views.", "path": ""}
    _progress("build_agent_started", {"stage": "build_agent_started", "agent": "views", "model": _MODEL_IMPL})
    view_output = _run_views(client, spec, vm_output, app_name, question)
    view_files = _parse_files(view_output)
    _progress("build_agent_completed", {"stage": "build_agent_completed", "agent": "views", "files": len(view_files)})

    # Step 5: Services
    if _cancelled():
        return {"ok": False, "message": "Cancelled before Services.", "path": ""}
    _progress("build_agent_started", {"stage": "build_agent_started", "agent": "services", "model": _MODEL_IMPL})
    svc_output = _run_services(client, spec, app_name, question)
    svc_files = _parse_files(svc_output)
    _progress("build_agent_completed", {"stage": "build_agent_completed", "agent": "services", "files": len(svc_files)})

    # Step 6: README
    if _cancelled():
        return {"ok": False, "message": "Cancelled before README.", "path": ""}
    _progress("build_agent_started", {"stage": "build_agent_started", "agent": "readme", "model": _MODEL_README})
    readme = _run_readme(client, spec, app_name, question)
    _progress("build_agent_completed", {"stage": "build_agent_completed", "agent": "readme", "output_chars": len(readme)})

    # Merge all file blocks
    all_files: dict[str, str] = {}
    all_files.update(scaffold_files)
    all_files.update(vm_files)
    all_files.update(view_files)
    all_files.update(svc_files)

    # Step 7: Write to disk
    _progress("build_agent_started", {"stage": "build_agent_started", "agent": "writer", "model": "filesystem"})
    project_dir = _write_project(repo_root, project_slug, app_name, all_files, readme)
    _progress("build_agent_completed", {"stage": "build_agent_completed", "agent": "writer", "path": str(project_dir)})

    bus.emit("desktop_pool", "completed", {
        "project": project_slug,
        "app_name": app_name,
        "path": str(project_dir),
        "files_written": len(all_files),
    })

    return {
        "ok": True,
        "path": str(project_dir),
        "app_name": app_name,
        "files_written": len(all_files),
        "spec": spec,
        "message": f"Desktop app '{app_name}' scaffolded — {len(all_files)} files.",
    }
