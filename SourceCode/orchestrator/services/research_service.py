from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from agents_research.persona_router import (
    PERSONAS,
    build_project_context_block,
    generate_persona_queries,
)
from agents_research.source_diversity import rank_bucket_sources
from agents_research.citation_linker import build_retrieved_chunks, link as link_citations
from agents_research.deep_researcher import (
    _build_source_quality_footer,
    _count_recycled_open_questions,
    _extract_web_source_evidence,
    _gap_assess,
    _load_prior_open_questions,
    _reliability_summary,
    _run_fill_agents,
)
from agents_research.synthesizer import SynthesisUnavailableError, synthesize, run_skeptic_pass
from shared_tools.file_store import ProjectStore
from shared_tools.inference_router import InferenceRouter
from shared_tools.model_routing import lane_model_config
from shared_tools.project_engine import ProjectEngine
from shared_tools.web_research import build_web_progress_payload
from shared_tools.workspace_knowledge import read_workspace_knowledge, resolve_default_patterns_path
from .agent_contracts import AgentTask

LOGGER = logging.getLogger(__name__)

# Tracks project slugs that currently have a heavy Foraging run in progress.
# Keyed by project_slug (single-user system — slug is sufficient).
_active_foraging_lock = threading.Lock()
_active_foraging: set[str] = set()


def _is_stack_decided_question(text: str, topic_type: str) -> bool:
    if str(topic_type or "").strip().lower() == "technical":
        return False
    low = str(text or "").strip().lower()
    if not low:
        return False
    triggers = (
        "database choice", "which database", "db choice",
        "which framework", "framework choice", "stack recommendation",
        "vue vs react", "react vs vue", "flask vs fastapi",
        "sqlite vs postgres", "postgres vs sqlite",
        "sqlite or postgres", "postgres or sqlite",
        "flask or fastapi", "fastapi or flask",
        "vue or react", "react or vue",
        "dotnet vs", "electron vs", "tauri vs",
    )
    if any(token in low for token in triggers):
        return True
    pair_patterns = (
        r"\bsqlite\s+(?:vs|or)\s+postgres\b",
        r"\bpostgres\s+(?:vs|or)\s+sqlite\b",
        r"\bflask\s+(?:vs|or)\s+fastapi\b",
        r"\bfastapi\s+(?:vs|or)\s+flask\b",
        r"\bvue\s+(?:vs|or)\s+react\b",
        r"\breact\s+(?:vs|or)\s+vue\b",
        r"\bdotnet\s+(?:vs|or)\s+(?:electron|tauri)\b",
        r"\b(?:electron|tauri)\s+(?:vs|or)\s+dotnet\b",
    )
    return any(re.search(pattern, low) for pattern in pair_patterns)


def _resolve_research_intent(
    *,
    research_intent: str,
    forage_profile: str,
    lane: str,
    topic_type: str,
) -> str:
    explicit = str(research_intent or "").strip().lower()
    if explicit:
        return explicit
    if str(forage_profile or "").strip().lower() == "domain":
        return "domain_foraging"
    if str(lane or "").strip().lower() == "project" or str(topic_type or "").strip().lower() == "technical":
        return "technical_planning"
    return "general_research"


class ResearchService:
    """Encapsulates research-lane orchestration while preserving legacy behavior.

    The orchestrator still owns many helper methods and final formatting rules.
    This service extracts the lane-specific execution flow so `main.py` no longer
    carries the full research control path inline.
    """

    def __init__(self, repo_root: Path, research_pool_runner: Callable[..., dict[str, Any]] | None = None) -> None:
        self.repo_root = repo_root
        self._research_pool_runner = research_pool_runner

    def execute_research_lane(
        self,
        host: Any,
        *,
        text: str,
        history: list[dict[str, str]] | None,
        topic_type: str,
        turn_plan: Any,
        force_research: bool,
        forage_profile: str = "technical",
        research_intent: str = "",
        cancel_checker=None,
        pause_checker=None,
        yield_checker=None,
        progress_callback=None,
        perf=None,
        reminder_note: str = "",
        event_note: str = "",
        lane: str = "research",
        details_sink: dict[str, Any] | None = None,
    ) -> str:
        if self._is_cancelled(cancel_checker):
            return "Request cancelled before research execution started."
        if _is_stack_decided_question(text, topic_type):
            return "This is a system-level stack decision. To re-evaluate, switch to a Technical topic."
        full_foraging = force_research or bool(getattr(turn_plan, "should_run_foraging", False))
        if not full_foraging:
            return self._execute_light_research(
                host,
                text=text,
                lane=lane,
                topic_type=topic_type,
                perf=perf,
                reminder_note=reminder_note,
                event_note=event_note,
                details_sink=details_sink,
            )
        return self._execute_full_research(
            host,
            text=text,
            lane=lane,
            history=history,
            topic_type=topic_type,
            forage_profile=forage_profile,
            research_intent=research_intent,
            cancel_checker=cancel_checker,
            pause_checker=pause_checker,
            yield_checker=yield_checker,
            progress_callback=progress_callback,
            reminder_note=reminder_note,
            event_note=event_note,
            details_sink=details_sink,
        )

    def execute_project_lane(
        self,
        host: Any,
        *,
        text: str,
        history: list[dict[str, str]] | None,
        topic_type: str,
        cancel_checker=None,
        pause_checker=None,
        yield_checker=None,
        progress_callback=None,
        reminder_note: str = "",
        event_note: str = "",
        details_sink: dict[str, Any] | None = None,
        research_intent: str = "",
    ) -> str:
        if self._is_cancelled(cancel_checker):
            return "Request cancelled before project-research execution started."
        if _is_stack_decided_question(text, topic_type):
            return "This is a system-level stack decision. To re-evaluate, switch to a Technical topic."
        slug = str(getattr(host, "project_slug", "") or "").strip()
        with _active_foraging_lock:
            if slug and slug in _active_foraging:
                return (
                    f"A Foraging run is already in progress for project **{slug}**. "
                    "Wait for it to complete or cancel it before starting another."
                )
            if slug:
                _active_foraging.add(slug)
        try:
            self._emit_web_start(progress_callback, lane="project")
            web_note, web_context, web_details, project_context = self._prepare_persona_discovery_web_context(
                host,
                text=text,
                lane="project",
                topic_type=topic_type,
                progress_callback=progress_callback,
            )
            self._emit_web_progress(progress_callback, web_details)
            out = self._run_research_pool(
                text=text,
                host=host,
                history=history,
                topic_type=topic_type,
                web_context=web_context,
                project_context=project_context,
                research_intent=_resolve_research_intent(
                    research_intent=research_intent,
                    forage_profile="technical",
                    lane="project",
                    topic_type=topic_type,
                ),
                cancel_checker=cancel_checker,
                pause_checker=pause_checker,
                yield_checker=yield_checker,
                progress_callback=progress_callback,
            )
            if web_details:
                out["web_details"] = web_details
            host._postprocess_research_summary(question=text, worker_result=out, topic_type=topic_type)
            if not bool(out.get("canceled", False)):
                out = self._gap_fill_pass(
                    host,
                    out,
                    text=text,
                    web_context=web_context,
                    topic_type=topic_type,
                    cancel_checker=cancel_checker,
                    pause_checker=pause_checker,
                    progress_callback=progress_callback,
                )
            if bool(out.get("synthesis_unavailable", False)):
                return str(out.get("message") or "Research could not complete — the synthesis model was unavailable. Try again in a few minutes.")
            if bool(out.get("canceled", False)):
                return str(
                    out.get("cancel_summary")
                    or (
                        "Request cancelled during project research. "
                        f"Partial summary saved to {out.get('summary_path', '')}."
                    )
                )
            fallback = (
                "I treated this as project strategy and asked the Foraging pool for a baseline synthesis. "
                f"Summary: {out['summary_path']}"
            )
            return self._finalize_research_reply(
                host,
                text=text,
                lane="project",
                topic_type=topic_type,
                out=out,
                fallback=fallback,
                web_note=web_note,
                reminder_note=reminder_note,
                event_note=event_note,
                queue_proposals=True,
                details_sink=details_sink,
            )
        finally:
            if slug:
                with _active_foraging_lock:
                    _active_foraging.discard(slug)

    def _execute_light_research(
        self,
        host: Any,
        *,
        text: str,
        lane: str,
        topic_type: str,
        perf=None,
        reminder_note: str = "",
        event_note: str = "",
        details_sink: dict[str, Any] | None = None,
    ) -> str:
        out = host._light_research_flow(
            question=text,
            lane=lane,
            topic_type=topic_type,
            project_context="",
            trace=perf,
        )
        web_details = out.get("web_details", {}) if isinstance(out.get("web_details", {}), dict) else {}
        reply = f"{out['message']} Summary: {out['summary_path']}"
        sources = [dict(x) for x in (web_details.get("sources") or []) if isinstance(x, dict)]
        reply = host._apply_confidence_gate(
            reply,
            sources=sources,
            conflict_summary=web_details.get("conflict_summary", {}),
        )
        reply = host._reynard_relay(
            user_text=text,
            lane=lane,
            internal_reply=reply,
            worker_result=out,
            topic_type=topic_type,
        )
        artifacts = host._format_research_artifacts_block(out)
        if perf is not None:
            perf.write()
        reply = f"{reply}\n\n{artifacts}"
        reply = host._append_daymarker_note(reply, event_note)
        reply = host._append_daymarker_note(reply, reminder_note)
        if isinstance(details_sink, dict):
            details_sink["research_reply"] = {
                "type": "research_reply",
                "text": reply,
                "sentences": [],
                "retrieved_chunks": [],
            }
        return host._complete_turn(user_text=text, lane=lane, reply_text=reply, worker_result=out)

    def _execute_full_research(
        self,
        host: Any,
        *,
        text: str,
        lane: str,
        history: list[dict[str, str]] | None,
        topic_type: str,
        forage_profile: str = "technical",
        research_intent: str = "",
        cancel_checker=None,
        pause_checker=None,
        yield_checker=None,
        progress_callback=None,
        reminder_note: str = "",
        event_note: str = "",
        details_sink: dict[str, Any] | None = None,
    ) -> str:
        slug = str(getattr(host, "project_slug", "") or "").strip()
        with _active_foraging_lock:
            if slug and slug in _active_foraging:
                return (
                    f"A Foraging run is already in progress for project **{slug}**. "
                    "Wait for it to complete or cancel it before starting another."
                )
            if slug:
                _active_foraging.add(slug)
        try:
            return self._execute_full_research_inner(
                host,
                text=text,
                lane=lane,
                history=history,
                topic_type=topic_type,
                forage_profile=forage_profile,
                research_intent=research_intent,
                cancel_checker=cancel_checker,
                pause_checker=pause_checker,
                yield_checker=yield_checker,
                progress_callback=progress_callback,
                reminder_note=reminder_note,
                event_note=event_note,
                details_sink=details_sink,
            )
        finally:
            if slug:
                with _active_foraging_lock:
                    _active_foraging.discard(slug)

    def _execute_full_research_inner(
        self,
        host: Any,
        *,
        text: str,
        lane: str,
        history: list[dict[str, str]] | None,
        topic_type: str,
        forage_profile: str = "technical",
        research_intent: str = "",
        cancel_checker=None,
        pause_checker=None,
        yield_checker=None,
        progress_callback=None,
        reminder_note: str = "",
        event_note: str = "",
        details_sink: dict[str, Any] | None = None,
    ) -> str:
        self._emit_web_start(progress_callback, lane=lane)
        web_note, web_context, web_details, project_context = self._prepare_persona_discovery_web_context(
            host,
            text=text,
            lane=lane,
            topic_type=topic_type,
            forage_profile=forage_profile,
            research_intent=research_intent,
            progress_callback=progress_callback,
        )
        self._emit_web_progress(progress_callback, web_details)
        out = self._run_research_pool(
            text=text,
            host=host,
            history=history,
            topic_type=topic_type,
            web_context=web_context,
            project_context=project_context,
            forage_profile=forage_profile,
            research_intent=_resolve_research_intent(
                research_intent=research_intent,
                forage_profile=forage_profile,
                lane=lane,
                topic_type=topic_type,
            ),
            cancel_checker=cancel_checker,
            pause_checker=pause_checker,
            yield_checker=yield_checker,
            progress_callback=progress_callback,
        )
        if web_details:
            out["web_details"] = web_details
        host._postprocess_research_summary(question=text, worker_result=out, topic_type=topic_type)
        if not bool(out.get("canceled", False)):
            out = self._gap_fill_pass(
                host,
                out,
                text=text,
                web_context=web_context,
                topic_type=topic_type,
                cancel_checker=cancel_checker,
                pause_checker=pause_checker,
                progress_callback=progress_callback,
            )
        if bool(out.get("synthesis_unavailable", False)):
            return str(out.get("message") or "Research could not complete — the synthesis model was unavailable. Try again in a few minutes.")
        if bool(out.get("canceled", False)):
            return str(
                out.get("cancel_summary")
                or (
                    "Request cancelled during Foraging. "
                    f"Partial summary saved to {out.get('summary_path', '')}."
                )
            )
        fallback = f"{out['message']} Summary: {out['summary_path']}"
        return self._finalize_research_reply(
            host,
            text=text,
            lane=lane,
            topic_type=topic_type,
            out=out,
            fallback=fallback,
            web_note=web_note,
            reminder_note=reminder_note,
            event_note=event_note,
            queue_proposals=True,
            details_sink=details_sink,
        )

    def _prepare_persona_discovery_web_context(
        self,
        host: Any,
        *,
        text: str,
        lane: str,
        topic_type: str,
        forage_profile: str = "technical",
        research_intent: str = "",
        progress_callback=None,
    ) -> tuple[str, str, dict[str, Any], str]:
        lane_key = str(lane or "research").strip().lower() or "research"
        project_slug = str(getattr(host, "project_slug", "") or "").strip() or "general"
        project_context = build_project_context_block(self.repo_root, project_slug)
        workspace_context = ""
        try:
            project = ProjectEngine(self.repo_root).get_by_slug(project_slug)
            workspace_path = str(project.get("workspace_path") or "").strip() if isinstance(project, dict) else ""
            if workspace_path:
                workspace_context = read_workspace_knowledge(
                    workspace_path,
                    max_chars=4000,
                    default_design_path=self.repo_root / "DESIGN.md",
                    default_patterns_path=resolve_default_patterns_path(
                        self.repo_root,
                        dict(project.get("stack") or {}) if isinstance(project, dict) else None,
                    ),
                )
        except Exception:
            workspace_context = ""

        persona_project_context = project_context
        if workspace_context:
            persona_project_context = (
                f"{project_context}\n\nWorkspace knowledge:\n{workspace_context}"
                if project_context
                else f"Workspace knowledge:\n{workspace_context}"
            )
        planner_client = InferenceRouter(self.repo_root)

        try:
            persona_queries = generate_persona_queries(
                question=text,
                project_context=persona_project_context,
                client=planner_client,
                repo_root=self.repo_root,
                forage_profile=forage_profile,
                research_intent=_resolve_research_intent(
                    research_intent=research_intent,
                    forage_profile=forage_profile,
                    lane=lane,
                    topic_type=topic_type,
                ),
            )
        except Exception:
            persona_queries = []
        if not persona_queries:
            # Hard fallback to legacy single-query path.
            web_note, web_context, web_details = host._prepare_web_context(
                text=text,
                lane=lane_key,
                topic_type=topic_type,
                force=True,
                progress_callback=progress_callback,
            )
            return web_note, web_context, web_details, project_context

        if callable(progress_callback):
            try:
                progress_callback(
                    "persona_queries_planned",
                    {
                        "queries": [
                            {"persona": str(row.get("label", "")), "query": str(row.get("query", ""))}
                            for row in persona_queries
                        ],
                        "count": len(persona_queries),
                    },
                )
            except Exception:
                pass

        def _crawl_one(row: dict[str, str]) -> dict[str, Any]:
            pid = str(row.get("id", "")).strip()
            label = str(row.get("label", pid)).strip() or pid
            query = str(row.get("query", text)).strip() or str(text)
            request_id = f"persona_{pid}_{int(time.time() * 1000)}"
            reason = f"Persona discovery crawl ({label}) for project-specific research diversity."
            if hasattr(host, "web_engine"):
                result = host.web_engine.run_query(
                    project=project_slug,
                    lane=lane_key,
                    query=query,
                    reason=reason,
                    request_id=request_id,
                    note=f"persona_discovery:{pid}",
                    topic_type=topic_type,
                    progress_callback=progress_callback,
                )
                return {
                    "id": pid,
                    "label": label,
                    "query": query,
                    "result": result if isinstance(result, dict) else {},
                }
            # If host has no web_engine, degrade to orchestrator helper.
            _note, _ctx, details = host._prepare_web_context(
                text=query,
                lane=lane_key,
                topic_type=topic_type,
                force=True,
                progress_callback=progress_callback,
            )
            _ = (_note, _ctx)
            return {
                "id": pid,
                "label": label,
                "query": query,
                "result": details if isinstance(details, dict) else {},
            }

        crawls: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(persona_queries)))) as executor:
            futures = [executor.submit(_crawl_one, row) for row in persona_queries]
            for future in as_completed(futures):
                try:
                    crawls.append(future.result())
                except Exception as exc:
                    LOGGER.warning("persona crawl failed: %s", exc)

        crawls_by_id = {str(row.get("id", "")): row for row in crawls if isinstance(row, dict)}
        ordered_crawls = [crawls_by_id.get(str(p.get("id", ""))) for p in persona_queries]
        ordered_crawls = [row for row in ordered_crawls if isinstance(row, dict)]

        total_raw = sum(
            len([r for r in (c.get("result", {}) or {}).get("sources", []) if isinstance(r, dict)])
            for c in ordered_crawls
        )
        if callable(progress_callback):
            try:
                progress_callback("ranking_sources", {"source_count": total_raw, "buckets": len(ordered_crawls)})
            except Exception:
                pass

        selected_sources: list[dict[str, Any]] = []
        tier_counts = {"tier1": 0, "tier2": 0, "tier3": 0, "tier4": 0}
        for crawl in ordered_crawls:
            query = str(crawl.get("query", "")).strip()
            label = str(crawl.get("label", "")).strip()
            details = crawl.get("result", {}) if isinstance(crawl.get("result", {}), dict) else {}
            raw_sources = [
                dict(row) for row in (details.get("sources") or []) if isinstance(row, dict)
            ][:20]
            ranked = rank_bucket_sources(
                persona_query=query,
                sources=raw_sources,
                top_k=8,
                client=planner_client,
                embedding_model="qwen3-embedding:4b",
            )
            for row in ranked:
                source_row = dict(row)
                source_row["originating_persona"] = label
                selected_sources.append(source_row)
                tier = str(source_row.get("source_tier", "tier3")).strip().lower() or "tier3"
                if tier not in tier_counts:
                    tier = "tier3"
                tier_counts[tier] += 1

        if not selected_sources:
            web_note, web_context, web_details = host._prepare_web_context(
                text=text,
                lane=lane_key,
                topic_type=topic_type,
                force=True,
                progress_callback=progress_callback,
            )
            return web_note, web_context, web_details, project_context

        lines = ["Recent web source cache (use only if relevant):"]
        for row in selected_sources:
            title = str(row.get("title", "")).strip() or str(row.get("url", "")).strip()
            url = str(row.get("url", "")).strip()
            if not url:
                continue
            snippet = str(row.get("snippet", "")).strip()
            tier = str(row.get("source_tier", "tier3")).strip().lower() or "tier3"
            score = float(row.get("source_score", row.get("diversity_score", 0.0)) or 0.0)
            fresh = float(row.get("freshness_score", 0.0) or 0.0)
            persona = str(row.get("originating_persona", "")).strip()
            analogy_flag = bool(row.get("analogy", False))
            if analogy_flag and tier != "tier4":
                tier = "tier4"
            note = f"persona={persona}"
            if analogy_flag:
                note += " [A]"
            lines.append(f"- [{tier} {score:.2f} fresh={fresh:.2f} {note}] {title} | {url}")
            if snippet:
                lines.append(f"  snippet: {snippet}")
        web_context = "\n".join(lines)

        web_note = (
            "Persona-driven discovery complete: 4 query lenses, "
            f"{len(selected_sources)} curated sources."
        )
        web_details = {
            "mode": "persona_discovery",
            "requested": True,
            "source_count": len(selected_sources),
            "seed_count": sum(int((row.get("result", {}) or {}).get("seed_count", 0) or 0) for row in ordered_crawls),
            "crawl_pages": sum(int((row.get("result", {}) or {}).get("crawl_pages", 0) or 0) for row in ordered_crawls),
            "crawl_gated_links": sum(int((row.get("result", {}) or {}).get("crawl_gated_links", 0) or 0) for row in ordered_crawls),
            "query_variants_count": len(persona_queries),
            "source_scoring_summary": {
                "tier_counts": tier_counts,
            },
            "sources": selected_sources,
            "persona_queries": persona_queries,
        }
        return web_note, web_context, web_details, project_context

    def _run_research_pool(
        self,
        *,
        text: str,
        host: Any,
        history: list[dict[str, str]] | None,
        topic_type: str,
        web_context: str,
        project_context: str = "",
        forage_profile: str = "technical",
        research_intent: str = "",
        cancel_checker=None,
        pause_checker=None,
        yield_checker=None,
        progress_callback=None,
    ) -> dict[str, Any]:
        if hasattr(host, "_run_registered_agent") and hasattr(host, "_make_agent_task"):
            return host._run_registered_agent(
                "research",
                host._make_agent_task(
                    lane="research",
                    text=text,
                    history=history,
                    context={
                        "web_context": web_context,
                        "topic_type": topic_type,
                        "project_context": project_context,
                        "forage_profile": forage_profile,
                        "research_intent": research_intent,
                    },
                    cancel_checker=cancel_checker,
                    pause_checker=pause_checker,
                    yield_checker=yield_checker,
                    progress_callback=progress_callback,
                ),
            )
        if callable(self._research_pool_runner):
            return self._research_pool_runner(
                text,
                self.repo_root,
                host.project_slug,
                host.bus,
                web_context=web_context,
                prior_messages=history or [],
                cancel_checker=cancel_checker,
                pause_checker=pause_checker,
                yield_checker=yield_checker,
                progress_callback=progress_callback,
                topic_type=topic_type,
                project_context=project_context,
                forage_profile=forage_profile,
                research_intent=research_intent,
                workspace_knowledge=str(project_context or ""),
            )
        raise RuntimeError("No research pool executor is available.")

    def _finalize_research_reply(
        self,
        host: Any,
        *,
        text: str,
        lane: str,
        topic_type: str,
        out: dict[str, Any],
        fallback: str,
        web_note: str,
        reminder_note: str,
        event_note: str,
        queue_proposals: bool,
        details_sink: dict[str, Any] | None = None,
    ) -> str:
        if web_note:
            fallback = f"{fallback}\n{web_note}"
        internal_reply = host._orchestrator_finalize(text, lane, out, fallback)
        reply = host._reynard_relay(
            user_text=text,
            lane=lane,
            internal_reply=internal_reply,
            worker_result=out,
            topic_type=topic_type,
        )
        if web_note and web_note not in reply:
            reply = f"{reply}\n{web_note}"
        if queue_proposals:
            host._queue_action_proposals(reply)
        citation_threshold = 0.45
        try:
            model_routing = getattr(host, "model_routing", {})
            if isinstance(model_routing, dict):
                raw_thresh = model_routing.get("research.citation_cosine_threshold", 0.45)
                citation_threshold = float(raw_thresh)
        except Exception:
            citation_threshold = 0.45
        retrieved_chunks = [dict(x) for x in (out.get("retrieved_chunks") or []) if isinstance(x, dict)]
        research_reply = link_citations(
            reply,
            retrieved_chunks=retrieved_chunks,
            threshold=max(0.0, min(1.0, citation_threshold)),
            embedding_client=getattr(host, "ollama", None),
        )
        if isinstance(research_reply, dict):
            out["research_reply"] = research_reply
            if isinstance(details_sink, dict):
                details_sink["research_reply"] = research_reply
        artifacts = host._format_research_artifacts_block(out)
        reply = f"{reply}\n\n{artifacts}"
        topic_reviews = int(out.get("topic_reviews_created", 0) or 0)
        if topic_reviews > 0:
            reply = f"{reply}\n\n_{topic_reviews} fact(s) queued for Postbag review._"
        reply = host._append_daymarker_note(reply, event_note)
        reply = host._append_daymarker_note(reply, reminder_note)
        return host._complete_turn(user_text=text, lane=lane, reply_text=reply, worker_result=out)

    def _gap_fill_pass(
        self,
        host: Any,
        out: dict[str, Any],
        *,
        text: str,
        web_context: str,
        topic_type: str,
        cancel_checker=None,
        pause_checker=None,
        progress_callback=None,
    ) -> dict[str, Any]:
        """Run a targeted gap-fill pass after initial synthesis if trigger conditions are met.

        Reads trigger signals from the reliability dict and synthesis text.
        Returns the original `out` unchanged when the loop is skipped.
        When triggered, returns an updated `out` with a new filled summary path.
        """
        # Kill switch — check before any work.
        model_cfg = lane_model_config(self.repo_root, "research_pool") or {}
        if not bool(model_cfg.get("gap_fill_enabled", False)):
            return out

        if self._is_cancelled(cancel_checker):
            return out

        # Read the synthesis that was written to disk.
        summary_path = str(out.get("summary_path", "")).strip()
        if not summary_path:
            return out
        try:
            summary_md = Path(summary_path).read_text(encoding="utf-8")
        except Exception:
            return out

        # Evaluate trigger conditions — rule-based, zero LLM cost.
        reliability = out.get("reliability") or {}
        weak = int(reliability.get("weak", 0))
        failed = int(reliability.get("failed", 0))
        agents_total = int(reliability.get("agents_total", 0))
        good = int(reliability.get("good", 0))

        findings = list(out.get("findings") or [])
        _all_scores = [f.get("confidence") for f in findings]
        _scored = [int(s) for s in _all_scores if isinstance(s, (int, float)) and int(s) > 0]
        avg_conf = sum(_scored) / len(_scored) if _scored else 0.0

        synth_low = summary_md.lower()
        has_low_confidence = (
            "evidence confidence: low" in synth_low
            or "evidence confidence: mixed" in synth_low
        )

        # Skip when all agents are good and avg confidence is high.
        if good == agents_total and agents_total > 0 and avg_conf >= 4.0:
            return out

        # Fire if any trigger condition is met.
        trigger = (
            (weak + failed >= 2)
            or (avg_conf > 0 and avg_conf < 3.0)
            or has_low_confidence
        )
        if not trigger:
            return out

        # Gap assessment — identify specific gaps via a fast LLM call.
        client = InferenceRouter(self.repo_root)
        gap_queries = _gap_assess(client, model_cfg, text, summary_md)
        if not gap_queries:
            return out

        # Emit progress so the UI can show the gap-fill phase.
        trigger_reason = (
            "weak_agents" if weak + failed >= 2
            else "low_avg_confidence" if avg_conf > 0 and avg_conf < 3.0
            else "synthesis_low_confidence"
        )
        if callable(progress_callback):
            try:
                progress_callback("gap_fill_started", {
                    "gap_queries": gap_queries,
                    "trigger_reason": trigger_reason,
                })
            except Exception:
                pass

        LOGGER.info(
            "gap_fill trigger=%s weak=%d failed=%d avg_conf=%.2f gaps=%d",
            trigger_reason, weak, failed, avg_conf, len(gap_queries),
        )

        # Run exactly 2 fill agents targeting the identified gaps.
        fill_findings = _run_fill_agents(
            client=client,
            model_cfg=model_cfg,
            question=text,
            gap_queries=gap_queries,
            web_context=web_context,
            prior_messages=None,
            findings=findings,
            source_evidence=_extract_web_source_evidence(web_context),
            cancel_checker=cancel_checker,
            pause_checker=pause_checker,
        )
        if not fill_findings:
            return out

        merged_findings = findings + fill_findings

        # Re-synthesize with the prior synthesis as focused context.
        synth_lane = lane_model_config(self.repo_root, "synthesis") or {}
        synth_cfg = dict(synth_lane)
        synth_cfg.setdefault("synthesis_timeout_sec", int(synth_cfg.get("timeout_sec", 480)))
        synth_cfg.setdefault("synthesis_retry_attempts", 4)
        synth_cfg.setdefault("synthesis_retry_backoff_sec", 2.0)
        synth_cfg.setdefault("synthesis_validation_cycles", 1)
        fb = list(model_cfg.get("fallback_models") or [])
        main_model = str(model_cfg.get("model", "")).strip()
        if main_model:
            fb.append(main_model)
        synth_cfg.setdefault("synthesis_fallback_models", fb)
        if str(topic_type).strip().lower() == "underground":
            synth_cfg["model"] = "huihui_ai/qwen3-abliterated:8b-Q4_K_M"
            synth_cfg["synthesis_fallback_models"] = ["huihui_ai/qwen3-abliterated:8b-Q4_K_M"]

        project_slug = str(getattr(host, "project_slug", "") or "").strip()
        prior_open_questions = _load_prior_open_questions(self.repo_root, project_slug)
        gap_source_evidence = _extract_web_source_evidence(web_context)
        gap_source_tier_map = {
            str(row.get("url", "")).strip().rstrip("/,."): str(row.get("source_tier", "")).strip().lower()
            for row in gap_source_evidence
            if isinstance(row, dict) and str(row.get("url", "")).strip()
        }
        try:
            new_summary_md = synthesize(
                text,
                merged_findings,
                client=client,
                model_cfg=synth_cfg,
                prior_synthesis=summary_md,
                prior_open_questions=prior_open_questions,
                source_tier_map=gap_source_tier_map,
            )
        except SynthesisUnavailableError as exc:
            LOGGER.error("Synthesis unavailable during gap-fill pass: %s", exc)
            return out

        # Adversarial skeptic pass on the updated synthesis.
        if callable(progress_callback):
            try:
                progress_callback("skeptic_pass_started", {
                    "phase": "gap_fill",
                    "note": "Running critique pass on updated synthesis.",
                })
            except Exception:
                pass
        new_summary_md, critique_log = run_skeptic_pass(
            text,
            new_summary_md,
            client=client,
            model_cfg=synth_cfg,
            findings=merged_findings,
        )
        if callable(progress_callback):
            try:
                progress_callback("skeptic_pass_completed", {
                    "phase": "gap_fill",
                    "critique_chars": len(str(critique_log or "").strip()),
                    "note": "Critique pass finished.",
                })
            except Exception:
                pass

        # Append source quality + recycled-open-question signal.
        new_reliability = _reliability_summary(merged_findings)
        recycled_questions = _count_recycled_open_questions(new_summary_md, prior_open_questions)
        quality_suffix = " | gap-fill pass applied"
        if recycled_questions > 0:
            quality_suffix += f" | recycled prior questions: {recycled_questions}"
        new_summary_md = f"{new_summary_md}\n\n---\n\n{_build_source_quality_footer(merged_findings, suffix=quality_suffix)}"

        # Write the filled summary to a new file.
        store = ProjectStore(self.repo_root)
        filled_name = store.timestamped_name("research_summary_filled")
        filled_path = store.write_project_file(
            project_slug, "research_summaries", filled_name, new_summary_md
        )
        if not critique_log.strip():
            critique_log = "_Skeptic pass produced no output for this run._"
        filled_critique_path = str(
            store.write_project_file(
                project_slug,
                "research_summaries",
                f"{filled_name}.critique.md",
                critique_log,
            )
        )

        LOGGER.info("gap_fill completed filled_summary=%s fill_agents=%d", filled_path, len(fill_findings))

        if callable(progress_callback):
            try:
                progress_callback("gap_fill_completed", {
                    "filled_summary_path": str(filled_path),
                    "filled_critique_path": filled_critique_path,
                    "fill_agents_used": len(fill_findings),
                    "gap_queries": gap_queries,
                })
            except Exception:
                pass

        updated_out = dict(out)
        updated_out["summary_path"] = str(filled_path)
        if filled_critique_path:
            updated_out["critique_path"] = filled_critique_path
        updated_out["gap_fill_used"] = True
        updated_out["reliability"] = new_reliability
        updated_out["findings"] = merged_findings
        updated_out["retrieved_chunks"] = build_retrieved_chunks(merged_findings)
        updated_out["recycled_open_questions"] = recycled_questions
        return updated_out

    def _emit_web_start(self, progress_callback, *, lane: str) -> None:
        if not callable(progress_callback):
            return
        try:
            progress_callback("web_research_started", {
                "note": "Collecting fresh web sources.",
                "lane": str(lane or "").strip(),
            })
        except Exception:
            pass

    def _emit_web_progress(self, progress_callback, web_details: dict[str, Any] | None) -> None:
        details = web_details if isinstance(web_details, dict) else {}
        if not details.get("requested") or not callable(progress_callback):
            return
        try:
            payload = build_web_progress_payload(details)
            payload["note"] = (
                f"Web crawl complete — {int(payload.get('source_count', 0) or 0)} sources, "
                f"{int(payload.get('crawl_pages', 0) or 0)} pages."
            )
            progress_callback("web_stack_ready", payload)
        except Exception:
            pass

    @staticmethod
    def _is_cancelled(cancel_checker) -> bool:
        if callable(cancel_checker):
            try:
                return bool(cancel_checker())
            except Exception:
                return False
        return False
