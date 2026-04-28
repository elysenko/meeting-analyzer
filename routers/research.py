"""
Research routes.

GET/POST/DELETE /workspaces/{id}/research/*
"""

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

from models import ResearchRefineRequest, ResearchRequest
from services import chat_svc as _chat_svc
from services.research_svc import serialize_research_row as _serialize_research_row
from services.text_svc import _excerpt_text, _json_line, _render_markdown_html
from services.workspace_svc import _ensure_user_workspace

router = APIRouter()
logger = logging.getLogger("meeting-analyzer")


@router.get("/workspaces/{workspace_id}/research")
async def list_research_sessions(request: Request, workspace_id: int):
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT rs.id, rs.title, rs.topic, rs.mode, rs.research_type, rs.status, rs.summary, rs.linked_todo_id,
                      rs.source_research_ids, rs.source_document_refs, rs.refinement, rs.created_at, rs.updated_at,
                      run_meta.artifact_template
               FROM research_sessions rs
               LEFT JOIN LATERAL (
                   SELECT gtr.artifact_template
                   FROM generate_task_runs gtr
                   WHERE gtr.grounding_research_id = rs.id
                   ORDER BY gtr.updated_at DESC, gtr.id DESC
                   LIMIT 1
               ) run_meta ON TRUE
               WHERE rs.workspace_id = $1
               ORDER BY rs.created_at DESC""",
            workspace_id,
        )
    return [_serialize_research_row(r) for r in rows]


@router.get("/workspaces/{workspace_id}/research/{research_id}")
async def get_research_session(request: Request, workspace_id: int, research_id: int):
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT rs.id, rs.title, rs.topic, rs.mode, rs.research_type, rs.status, rs.summary, rs.content,
                      rs.sources, rs.llm_provider, rs.llm_model, rs.error, rs.refinement, rs.linked_todo_id,
                      rs.source_research_ids, rs.source_document_refs, rs.created_at, rs.updated_at,
                      run_meta.artifact_template
               FROM research_sessions rs
               LEFT JOIN LATERAL (
                   SELECT gtr.artifact_template
                   FROM generate_task_runs gtr
                   WHERE gtr.grounding_research_id = rs.id
                   ORDER BY gtr.updated_at DESC, gtr.id DESC
                   LIMIT 1
               ) run_meta ON TRUE
               WHERE rs.workspace_id = $1 AND rs.id = $2""",
            workspace_id,
            research_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Research session not found")
    return _serialize_research_row(row)


@router.delete("/workspaces/{workspace_id}/research/{research_id}")
async def delete_research_session(request: Request, workspace_id: int, research_id: int):
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM research_sessions WHERE workspace_id = $1 AND id = $2",
            workspace_id, research_id,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Research session not found")
    return {"ok": True}


@router.post("/workspaces/{workspace_id}/research/batch-delete")
async def batch_delete_research_sessions(request: Request, workspace_id: int, body: dict):
    await _ensure_user_workspace(request, workspace_id)
    ids = body.get("ids", [])
    if not ids:
        raise HTTPException(status_code=400, detail="No session IDs provided")
    async with request.app.state.db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM research_sessions WHERE workspace_id = $1 AND id = ANY($2::int[])",
            workspace_id, ids,
        )
    deleted = int(result.split()[-1]) if result.startswith("DELETE ") else 0
    return {"ok": True, "deleted": deleted}


@router.post("/workspaces/{workspace_id}/research/refine")
async def refine_research_question(request: Request, workspace_id: int, body: ResearchRefineRequest):
    from main_live import (
        _generate_research_refinement_questions,
        _suggest_related_research_sessions, _suggest_refinement_prefill,
    )
    await _ensure_user_workspace(request, workspace_id)
    topic = body.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic is required")
    mode = (body.mode or "deep").strip().lower()
    research_type = (body.research_type or "general").strip() or "general"
    document_manifest: list[dict[str, Any]] = []
    if body.document_ids:
        try:
            async with request.app.state.db_pool.acquire() as conn:
                doc_rows = await conn.fetch(
                    "SELECT id, filename, executive_summary FROM documents WHERE workspace_id = $1 AND id = ANY($2::int[])",
                    workspace_id, [int(d) for d in body.document_ids],
                )
            for row in doc_rows:
                document_manifest.append({"id": row["id"], "type": "document", "filename": row["filename"], "title": row["filename"], "summary": _excerpt_text(row["executive_summary"], 200)})
        except Exception:
            pass
    if body.meeting_ids:
        try:
            async with request.app.state.db_pool.acquire() as conn:
                mtg_rows = await conn.fetch(
                    "SELECT id, title, summary FROM meetings WHERE workspace_id = $1 AND id = ANY($2::int[])",
                    workspace_id, [int(m) for m in body.meeting_ids],
                )
            for row in mtg_rows:
                document_manifest.append({"id": row["id"], "type": "meeting", "title": row["title"], "summary": _excerpt_text(row["summary"], 200)})
        except Exception:
            pass
    result, meta = await _generate_research_refinement_questions(
        workspace_id,
        topic,
        mode,
        research_type,
        document_manifest=document_manifest or None,
    )
    result["llm_provider"] = meta.get("provider")
    result["llm_model"] = meta.get("model")
    related = await _suggest_related_research_sessions(workspace_id, topic)
    context_lines = []
    for item in related[:4]:
        if item.get("score", 0) <= 0:
            continue
        context_lines.append(
            f'Research [{item.get("id")}] {item.get("title") or item.get("topic")}\n'
            f'{_excerpt_text(item.get("summary"), 900)}'
        )
    if document_manifest:
        manifest_lines = []
        for item in document_manifest:
            label = item.get("type", "document").capitalize()
            name = item.get("title") or item.get("filename") or f"ID {item.get('id')}"
            summary = item.get("summary") or ""
            manifest_lines.append(f"[{label}: {name}]\n{summary}" if summary else f"[{label}: {name}]")
        context_lines.insert(0, "Selected source documents:\n" + "\n".join(manifest_lines))
    prefill_state = await _suggest_refinement_prefill(
        workspace_id,
        topic,
        mode,
        research_type,
        result,
        context_brief="\n\n".join(context_lines),
    )
    if prefill_state:
        result["prefill_state"] = prefill_state
    return result


@router.post("/workspaces/{workspace_id}/research")
async def create_research_session(request: Request, workspace_id: int, body: ResearchRequest):
    from main_live import (
        _create_research_session,
        _update_research_session, _build_research_refinement_contract,
        _normalize_research_refinement, _retrieve_document_evidence,
        _retrieve_meeting_evidence, _retrieve_research_evidence,
        _run_deep_research, _run_quick_research,
    )
    await _ensure_user_workspace(request, workspace_id)
    mode = (body.mode or "quick").strip().lower()
    if mode not in ("quick", "deep"):
        raise HTTPException(status_code=400, detail="mode must be 'quick' or 'deep'")
    topic = body.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic is required")

    research_id = await _create_research_session(
        workspace_id,
        topic,
        mode,
        (body.research_type or "general").strip() or "general",
    )

    async def stream():
        yield json.dumps({"research_id": research_id, "status": f"Starting {mode} research..."}) + "\n"
        try:
            refinement_contract = None
            normalized_refinement = _normalize_research_refinement(body.refinement)
            if normalized_refinement:
                yield json.dumps({"status": "Refining research brief..."}) + "\n"
                refinement_contract, refine_meta = await _build_research_refinement_contract(
                    workspace_id,
                    topic,
                    body.research_type,
                    normalized_refinement,
                )
                await _update_research_session(
                    research_id,
                    refinement=refinement_contract,
                    llm_provider=refine_meta.get("provider"),
                    llm_model=refine_meta.get("model"),
                )
            document_evidence: list[dict[str, Any]] = []
            prior_research: list[dict[str, Any]] = []
            has_sources = bool(body.document_ids or body.meeting_ids or body.research_ids)
            if has_sources:
                yield json.dumps({"status": "Retrieving evidence from selected sources..."}) + "\n"
            if body.document_ids:
                try:
                    doc_ev = await _retrieve_document_evidence(workspace_id, body.document_ids, topic)
                    document_evidence.extend(doc_ev)
                except Exception as exc:
                    logger.warning("Research document evidence retrieval failed: %s", exc)
            if body.meeting_ids:
                try:
                    mtg_ev = await _retrieve_meeting_evidence(workspace_id, body.meeting_ids, topic)
                    document_evidence.extend(mtg_ev)
                except Exception as exc:
                    logger.warning("Research meeting evidence retrieval failed: %s", exc)
            if body.research_ids:
                try:
                    res_ev = await _retrieve_research_evidence(workspace_id, body.research_ids, topic)
                    prior_research = res_ev
                except Exception as exc:
                    logger.warning("Research prior-research retrieval failed: %s", exc)

            yield json.dumps({"status": "Planning search strategy..."}) + "\n"
            progress_queue: asyncio.Queue[str] = asyncio.Queue()

            async def emit_progress(message: str) -> None:
                await progress_queue.put(message)

            async def perform_research():
                if mode == "deep":
                    return await _run_deep_research(
                        workspace_id,
                        topic,
                        body.research_type,
                        refinement_contract=refinement_contract,
                        document_evidence=document_evidence or None,
                        prior_research=prior_research or None,
                        progress_callback=emit_progress,
                    )
                return await _run_quick_research(
                    workspace_id,
                    topic,
                    body.research_type,
                    refinement_contract=refinement_contract,
                    document_evidence=document_evidence or None,
                    prior_research=prior_research or None,
                    progress_callback=emit_progress,
                )

            research_task = asyncio.create_task(perform_research())
            while not research_task.done():
                try:
                    message = await asyncio.wait_for(progress_queue.get(), timeout=0.35)
                    yield json.dumps({"research_id": research_id, "status": message}) + "\n"
                except asyncio.TimeoutError:
                    continue
            while not progress_queue.empty():
                yield json.dumps({"research_id": research_id, "status": await progress_queue.get()}) + "\n"
            result, meta = await research_task
            if meta.get("warning"):
                yield json.dumps({"warning": meta["warning"]}) + "\n"
            yield json.dumps({"status": "Writing final report..."}) + "\n"
            await _update_research_session(
                research_id,
                title=result["title"],
                summary=result["summary"],
                content=result["content"],
                sources=result["sources"],
                status="completed",
                llm_provider=meta.get("provider"),
                llm_model=meta.get("model"),
                refinement=refinement_contract,
                source_research_ids=result.get("source_research_ids") or [],
                source_document_refs=result.get("source_document_refs") or (document_evidence if document_evidence else None),
            )
            if body.chat_session_id:
                try:
                    await _chat_svc.append_chat_session_message(
                        request.app.state.db_pool, workspace_id, body.chat_session_id, "user", topic,
                    )
                    sources_lines = []
                    for s in (result.get("sources") or [])[:10]:
                        url = s.get("url", "")
                        title_s = s.get("title", "") or url
                        if url:
                            sources_lines.append(f"- [{title_s}]({url})")
                        elif title_s:
                            sources_lines.append(f"- {title_s}")
                    sources_md = ("\n\n**Sources:**\n" + "\n".join(sources_lines)) if sources_lines else ""
                    chat_content = f"**Research: {result['title']}**\n\n{result.get('content') or result.get('summary', '')}{sources_md}"
                    await _chat_svc.append_chat_session_message(
                        request.app.state.db_pool, workspace_id, body.chat_session_id, "assistant", chat_content,
                    )
                except Exception as _persist_exc:
                    logger.warning("Failed to persist research result to chat session %s: %s", body.chat_session_id, _persist_exc)
            yield _json_line({
                "result": {
                    "id": research_id,
                    **result,
                    "content_html": _render_markdown_html(result.get("content") or result.get("summary") or ""),
                    "llm_provider": meta.get("provider"),
                    "llm_model": meta.get("model"),
                }
            })
        except Exception as exc:
            await _update_research_session(
                research_id,
                status="failed",
                error=str(exc),
            )
            yield json.dumps({"error": str(exc), "research_id": research_id}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
