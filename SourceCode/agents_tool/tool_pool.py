from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from shared_tools.config_ini import load_config
from shared_tools.feedback_learning import FeedbackLearningEngine
from shared_tools.file_store import ProjectStore
from shared_tools.model_routing import lane_model_config
from shared_tools.ollama_client import OllamaClient


_MAX_FIX_CYCLES = 3
_RUN_TIMEOUT_SEC = 30
_CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

_PYTHON_CLI_PATTERNS = """\
# Validated against: Python 3.12+

Python CLI patterns:
1. Use `from __future__ import annotations` at file top.
2. Prefer `pathlib.Path` over `os.path`.
3. Use `argparse` for CLI parsing unless richer UX is explicitly requested.
4. Use `subprocess.run([...], check=False, text=True, capture_output=True)` when invoking commands.
5. Use `logging` (not print) for non-interactive status and error reporting.
6. Add type hints on all function signatures.
7. Use distinct exit codes from `main()` and `sys.exit(code)`.
8. Include a smoke entry path under `if __name__ == "__main__":`.
"""

_PYTHON_GOTCHAS = """\
Python gotchas to avoid:
- Never use mutable default arguments; use `None` guard.
- Never use `subprocess.run(..., shell=True)` with variable user input.
- Do not use `os.path.join` for URLs (use `urllib.parse.urljoin`).
- Never call `requests.get(url)` without a timeout.
- Never use bare `except:`; use `except Exception:`.
- Prefer `datetime.now(timezone.utc)` over naive `datetime.now()`.
- Always open text files with `encoding="utf-8"` and a context manager.
- Never use `eval`/`exec` on untrusted input.
- Avoid unbounded `json.load` on untrusted large payloads.
- Avoid fragile relative imports in single-file scripts.
"""

_TOOL_AGENTS = [
    {
        "persona": "tool_architect",
        "model": "qwen2.5-coder:7b",
        "directive": (
            "Design the tool's structure, interface, and dependencies. "
            "Specify inputs, outputs, data flow, and error handling approach. "
            "Be concrete — name the functions, their signatures, and any external libs needed.\n\n"
            + _PYTHON_CLI_PATTERNS + "\n\n" + _PYTHON_GOTCHAS
        ),
    },
    {
        "persona": "tool_implementer",
        "model": "qwen2.5-coder:7b",
        "directive": (
            "Produce the complete, runnable implementation. "
            "Write clean, well-commented code. Include a usage example at the bottom. "
            "Handle edge cases and errors explicitly. "
            "Use only Python standard library unless the request clearly requires a third-party package.\n\n"
            + _PYTHON_CLI_PATTERNS + "\n\n" + _PYTHON_GOTCHAS
        ),
    },
]


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _extract_code(text: str) -> str:
    """Return the largest fenced code block found in text, or '' if none."""
    blocks = _CODE_FENCE_RE.findall(str(text or ""))
    if not blocks:
        return ""
    return max(blocks, key=len).strip()


def _run_code(code: str) -> tuple[bool, str, str]:
    """Write code to a temp file and run it. Returns (success, stdout, stderr)."""
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_SEC,
        )
        return result.returncode == 0, result.stdout[:2000], result.stderr[:2000]
    except subprocess.TimeoutExpired:
        return False, "", f"[Execution timed out after {_RUN_TIMEOUT_SEC}s]"
    except Exception as exc:
        return False, "", f"[Execution error: {exc}]"
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass


def _run_fix_agent(
    client: OllamaClient,
    code: str,
    error_text: str,
    question: str,
    cancel_checker: Callable[[], bool] | None,
    model_name: str,
) -> str:
    """Ask the model to fix code given an error. Returns fixed code or original if it fails."""
    if callable(cancel_checker):
        try:
            if cancel_checker():
                return code
        except Exception:
            pass

    system_prompt = (
        "You are a Python debugging agent. "
        "You will be given Python code and the error it produced. "
        "Return the complete corrected code inside a ```python code block. "
        "Do not truncate — return the full file. No explanations outside the code block."
    )
    user_prompt = (
        f"Original request: {question}\n\n"
        f"Code that failed:\n```python\n{code}\n```\n\n"
        f"Error output:\n{error_text}\n\n"
        "Return the complete corrected Python code."
    )
    try:
        result = client.chat(
            model=str(model_name or "qwen2.5-coder:7b"),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            num_ctx=16384,
            think=False,
            timeout=300,
            retry_attempts=3,
            retry_backoff_sec=1.5,
        )
        fixed = _extract_code(str(result or "").strip())
        return fixed if fixed else code
    except Exception:
        return code


def _generate_test_harness(
    client: OllamaClient,
    code: str,
    question: str,
    cancel_checker: Callable[[], bool] | None,
    model_name: str,
) -> str:
    """Generate a test block appended to the tool code that verifies logic, not just crash-free.

    The returned string is Python code intended to be appended after the tool's existing code.
    It calls the tool's main functions with sample data and uses assert statements to check that
    results are non-empty and of the expected type.
    """
    if callable(cancel_checker):
        try:
            if cancel_checker():
                return ""
        except Exception:
            pass

    system_prompt = (
        "You are a Python test writer. Write a SHORT test block (15–30 lines) that verifies "
        "a Python tool produces correct, meaningful output. "
        "Rules:\n"
        "- The test will be APPENDED to the existing code — all functions are already in scope.\n"
        "- Create minimal but realistic sample input data (use StringIO or tempfile for stdin/files).\n"
        "- Call the tool's main function(s) with this sample data.\n"
        "- Use assert statements: check output is not None, not empty, has expected type or keys.\n"
        "- Print 'TEST PASSED: [what was verified]' on each successful assert.\n"
        "- Do NOT use if __name__ == '__main__' — write the test code at module level.\n"
        "- Do NOT redefine any functions — they are already in scope.\n"
        "Return ONLY the test code inside a ```python block. No explanations."
    )
    user_prompt = (
        f"Tool request: {question}\n\n"
        f"Tool code (functions are already defined — do not redefine them):\n"
        f"```python\n{code[:4000]}\n```\n\n"
        "Write the test block."
    )
    try:
        result = client.chat(
            model=str(model_name or "qwen2.5-coder:7b"),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.15,
            num_ctx=12288,
            think=False,
            timeout=120,
            retry_attempts=3,
            retry_backoff_sec=1.5,
        )
        return _extract_code(str(result or "").strip())
    except Exception:
        return ""


def _run_tool_agent(
    client: OllamaClient,
    agent_cfg: dict[str, Any],
    question: str,
    context_knowledge: str,
    prior_findings: str = "",
    cancel_checker: Callable[[], bool] | None = None,
) -> dict[str, str]:
    persona = str(agent_cfg.get("persona", "tool_agent")).strip()
    directive = str(agent_cfg.get("directive", "")).strip()
    model = str(agent_cfg.get("model", "qwen2.5-coder:7b")).strip()

    system_prompt = (
        f"Today's date: {_today()}. "
        "You are a Make mode tool-generation agent. No web research is available — "
        "use only accumulated project knowledge and your training. "
        f"Your role is {persona}. {directive} "
        "Output clean, runnable code with explanatory comments."
    )
    context_parts: list[str] = []
    if context_knowledge.strip():
        context_parts.append(f"Context knowledge:\n{context_knowledge.strip()}")
    if prior_findings.strip():
        context_parts.append(f"Prior agent output (use as foundation):\n{prior_findings.strip()}")
    context_block = "\n\n".join(context_parts)
    user_prompt = (
        f"Tool request:\n{question}\n\n{context_block}".strip()
        + "\n\nReturn your output as plain text or markdown code blocks."
    )

    if callable(cancel_checker):
        try:
            if cancel_checker():
                return {"agent": persona, "finding": f"Cancelled before {persona} could run."}
        except Exception:
            pass

    try:
        result = client.chat(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.15,
            num_ctx=12288,
            think=False,
            timeout=300,
            retry_attempts=4,
            retry_backoff_sec=1.5,
        )
        return {"agent": persona, "finding": str(result or "").strip()}
    except Exception as exc:
        return {"agent": persona, "finding": f"Model call failed for {persona}: {exc}"}


def run_tool_pool(
    question: str,
    repo_root: Path,
    project_slug: str,
    bus: Any,
    research_context: str = "",
    prior_messages: list[dict[str, str]] | None = None,
    cancel_checker: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    def _progress(stage: str, detail: dict[str, Any] | None = None) -> None:
        if not callable(progress_callback):
            return
        try:
            progress_callback(stage, detail or {})
        except Exception:
            pass

    def _is_cancelled() -> bool:
        if callable(cancel_checker):
            try:
                return bool(cancel_checker())
            except Exception:
                return False
        return False

    bus.emit("tool_pool", "start", {"question": question, "project": project_slug})
    _progress("tool_pool_started", {"agents_total": len(_TOOL_AGENTS), "project": project_slug})

    cfg = lane_model_config(repo_root, "make_tool") or {}
    try:
        config_ini = load_config(repo_root)
        _TOOL_AGENTS[0]["model"] = config_ini.get_model("make_tool_architect")
        _TOOL_AGENTS[1]["model"] = config_ini.get_model("make_tool_implementer")
    except Exception:
        pass
    client = OllamaClient()
    orchestrator_cfg = lane_model_config(repo_root, "orchestrator_reasoning")
    learning = FeedbackLearningEngine(repo_root, client=client, model_cfg=orchestrator_cfg)
    learned_guidance = learning.guidance_for_lane("make_tool", limit=5)
    context_knowledge = "\n\n".join(
        part for part in [learned_guidance, f"Research context:\n{research_context.strip()}" if research_context.strip() else ""] if part
    ).strip()
    findings: list[dict[str, str]] = []
    prior_finding_text = ""

    for agent_cfg in _TOOL_AGENTS:
        if _is_cancelled():
            break
        _progress("tool_agent_started", {"agent": agent_cfg["persona"]})
        result = _run_tool_agent(
            client,
            agent_cfg,
            question,
            context_knowledge,
            prior_findings=prior_finding_text,
            cancel_checker=cancel_checker,
        )
        findings.append(result)
        prior_finding_text = str(result.get("finding", "")).strip()
        _progress("tool_agent_completed", {"agent": result["agent"], "finding_preview": prior_finding_text[:300]})

    # --- Code execution + fix loop ---
    impl_finding = findings[-1].get("finding", "") if findings else ""
    code = _extract_code(impl_finding)
    exec_status = "skipped"
    cycles_used = 0
    last_stdout = ""
    last_stderr = ""

    if code and not _is_cancelled():
        _progress("tool_execution_started", {"message": "Running generated code..."})
        success, last_stdout, last_stderr = _run_code(code)

        if success:
            exec_status = "passed"
            _progress("tool_execution_passed", {"cycle": 0, "stdout_preview": last_stdout[:300]})
        else:
            exec_status = "exhausted"
            for fix_cycle in range(1, _MAX_FIX_CYCLES + 1):
                if _is_cancelled():
                    exec_status = "cancelled"
                    break
                error_summary = (last_stderr or last_stdout or "unknown error")[:800]
                _progress("tool_execution_failed", {
                    "cycle": fix_cycle,
                    "total": _MAX_FIX_CYCLES,
                    "error_preview": error_summary[:200],
                })
                _progress("tool_fix_cycle_started", {"cycle": fix_cycle, "total": _MAX_FIX_CYCLES})
                code = _run_fix_agent(
                    client,
                    code,
                    error_summary,
                    question,
                    cancel_checker,
                    model_name=str(cfg.get("model", "qwen2.5-coder:7b")),
                )
                cycles_used = fix_cycle
                _progress("tool_fix_cycle_completed", {"cycle": fix_cycle, "total": _MAX_FIX_CYCLES})
                success, last_stdout, last_stderr = _run_code(code)
                if success:
                    exec_status = "passed"
                    _progress("tool_execution_passed", {
                        "cycle": fix_cycle,
                        "stdout_preview": last_stdout[:300],
                    })
                    break
            else:
                _progress("tool_execution_exhausted", {
                    "cycles": _MAX_FIX_CYCLES,
                    "last_error": (last_stderr or last_stdout or "")[:300],
                    "message": f"Could not fix after {_MAX_FIX_CYCLES} attempts — review manually.",
                })
    else:
        _progress("tool_execution_skipped", {"message": "No code block found in implementer output."})

    # --- Test harness: logic verification (only when crash-test passed) ---
    test_harness_code = ""
    test_harness_status = "skipped"

    if exec_status == "passed" and code and not _is_cancelled():
        _progress("tool_test_harness_started", {"message": "Generating logic test..."})
        test_harness_code = _generate_test_harness(
            client,
            code,
            question,
            cancel_checker,
            model_name=str(cfg.get("model", "qwen2.5-coder:7b")),
        )

        if test_harness_code and not _is_cancelled():
            combined = code + "\n\n# --- Test harness ---\n" + test_harness_code
            _progress("tool_test_harness_running", {})
            test_ok, test_stdout, test_stderr = _run_code(combined)

            if test_ok:
                test_harness_status = "passed"
                _progress("tool_test_harness_passed", {"stdout_preview": test_stdout[:300]})
            else:
                test_harness_status = "failed"
                error_msg = (test_stderr or test_stdout or "unknown test failure")
                _progress("tool_test_harness_failed", {"error_preview": error_msg[:300]})

                # One targeted logic-fix pass using the test failure as the error signal
                if not _is_cancelled():
                    logic_error = (
                        f"Logic test failed:\n{error_msg[:600]}\n\n"
                        f"Test code that failed:\n{test_harness_code[:600]}"
                    )
                    _progress("tool_test_harness_fix_started", {})
                    code = _run_fix_agent(
                        client,
                        code,
                        logic_error,
                        question,
                        cancel_checker,
                        model_name=str(cfg.get("model", "qwen2.5-coder:7b")),
                    )
                    retest_ok, retest_stdout, retest_stderr = _run_code(
                        code + "\n\n# --- Test harness ---\n" + test_harness_code
                    )
                    if retest_ok:
                        test_harness_status = "fixed"
                        exec_status = "passed"
                        _progress("tool_test_harness_fixed", {"stdout_preview": retest_stdout[:300]})
                    else:
                        test_harness_status = "exhausted"
                        exec_status = "logic_test_failed"
                        _progress("tool_test_harness_exhausted", {
                            "message": "Logic test still failing after fix — review manually.",
                            "last_error": (retest_stderr or retest_stdout or "")[:300],
                        })
        else:
            _progress("tool_test_harness_skipped", {"reason": "No test code generated."})

    # --- Persist artifacts ---
    store = ProjectStore(repo_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Save .py file if we have code
    py_path: Path | None = None
    if code:
        py_name = f"{timestamp}_tool.py"
        py_path = store.write_project_file(project_slug, "implementation", py_name, code)

    # Build execution result block for the .md
    exec_lines = [
        "## Execution Result",
        f"Status: {exec_status.upper()}  |  Fix cycles used: {cycles_used}  |  Logic test: {test_harness_status.upper()}",
    ]
    if last_stdout:
        exec_lines.append(f"\n**stdout:**\n```\n{last_stdout[:600]}\n```")
    if last_stderr and exec_status != "passed":
        exec_lines.append(f"\n**stderr:**\n```\n{last_stderr[:600]}\n```")
    exec_block = "\n".join(exec_lines)

    impl_name = f"{timestamp}_tool_implementation.md"
    combined = "\n\n---\n\n".join(
        f"## {r['agent']}\n\n{r['finding']}" for r in findings if r.get("finding")
    )
    impl_md = f"# Tool Implementation\n\nRequest: {question}\n\n{combined}\n\n---\n\n{exec_block}\n"
    impl_path = store.write_project_file(project_slug, "implementation", impl_name, impl_md)

    bus.emit("tool_pool", "completed", {"project": project_slug, "path": str(impl_path)})
    _progress("tool_pool_completed", {"path": str(impl_path), "exec_status": exec_status})

    return {
        "message": (
            f"Tool generation complete. Status: {exec_status}. "
            + (f"Saved to {py_path.name}." if py_path else "No runnable code extracted.")
        ),
        "path": str(impl_path),
        "py_path": str(py_path) if py_path else None,
        "exec_status": exec_status,
        "findings": findings,
    }
