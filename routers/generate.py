"""
Generate task routes.

GET/POST/PATCH/DELETE /workspaces/{id}/generate/tasks/*
POST /workspaces/{id}/generate (legacy streaming generate)
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from starlette.responses import StreamingResponse

from models import (
    AddCustomSectionRequest, BrandRefreshRequest, DeliverableRequest,
    GenerateRequest, GenerateTaskAutofillRequest, GenerateTaskCreateRequest,
    GenerateTaskQuestionResearchRequest, GenerateTaskResearchRequest,
    GenerateTaskUpdateRequest, QuestionChatRequest,
)
from services.llm_prefs import _resolve_task_llm
from services.llm_prefs_svc import _get_workspace_llm_preferences
from services.research_svc import _get_template_config
from services.utils import _json_dict, _json_line
from services.workspace_svc import _ensure_user_workspace

# Heavy orchestration helpers live in main_live during the refactor.
# Consolidated here so import errors surface at startup rather than per-request.
from main_live import (  # noqa: E402
    _build_docx_bytes,
    _build_generate_answer_meta_patch,
    _build_generate_question_context,
    _build_generate_task_research_topic,
    _build_generation_context,
    _build_pdf_bytes,
    _build_pptx_bytes,
    _build_task_document_evidence,
    _call_llm_runner,
    _call_llm_runner_json,
    _collect_manual_answer_conflicts,
    _create_or_get_generate_task_session,
    _dedupe_document_refs,
    _delete_generate_task_run,
    _delete_generate_task_session,
    _derive_generate_run_intake,
    _document_evidence_prompt_block,
    _ensure_task_linked_research_session,
    _extract_branding_from_website,
    _fork_generate_task_run,
    _generate_research_refinement_questions,
    _generate_task_answer_autofill,
    _get_generate_task_session,
    _list_generate_task_sessions,
    _map_reduce_synthesis,
    _normalize_generate_template_draft,
    _reset_generate_task_step,
    _run_generate_task_deliverable_stream,
    _run_generate_task_question_chat,
    _run_generate_task_question_research,
    _run_generate_task_research_stream,
    _run_quick_research,
    _sanitize_question_plan_item,
    _slugify,
    _split_context_sections,
    _store_generated_document,
    _suggest_refinement_prefill,
    _update_generate_task_session_row,
    _update_research_session,
    test_intake_flow as _test_intake_flow_impl,
)

router = APIRouter()
logger = logging.getLogger("meeting-analyzer")


# ---------------------------------------------------------------------------
# Generate task CRUD
# ---------------------------------------------------------------------------

@router.get("/workspaces/{workspace_id}/generate/tasks")
async def list_generate_tasks(request: Request, workspace_id: int):
    await _ensure_user_workspace(request, workspace_id)
    return await _list_generate_task_sessions(workspace_id)


@router.post("/workspaces/{workspace_id}/generate/tasks")
async def create_generate_task(request: Request, workspace_id: int, body: GenerateTaskCreateRequest):
    await _ensure_user_workspace(request, workspace_id)
    return await _create_or_get_generate_task_session(
        workspace_id,
        body.todo_id,
        body.artifact_template,
        body.output_type,
        reset_setup=body.reset_setup,
    )


@router.get("/workspaces/{workspace_id}/generate/tasks/{task_id}")
async def get_generate_task(request: Request, workspace_id: int, task_id: int, run_id: int | None = None):
    await _ensure_user_workspace(request, workspace_id)
    return await _get_generate_task_session(workspace_id, task_id, run_id=run_id)


@router.patch("/workspaces/{workspace_id}/generate/tasks/{task_id}")
async def patch_generate_task(request: Request, workspace_id: int, task_id: int, body: GenerateTaskUpdateRequest):
    await _ensure_user_workspace(request, workspace_id)
    return await _update_generate_task_session_row(workspace_id, task_id, body)


@router.delete("/workspaces/{workspace_id}/generate/tasks/{task_id}")
async def delete_generate_task(request: Request, workspace_id: int, task_id: int):
    await _ensure_user_workspace(request, workspace_id)
    return await _delete_generate_task_session(workspace_id, task_id)


@router.post("/workspaces/{workspace_id}/generate/tasks/{task_id}/reset-step")
async def reset_generate_task_step_endpoint(request: Request, workspace_id: int, task_id: int, body: dict = Body(...)):
    step = str(body.get("step") or "").strip()
    if step not in ("setup", "research", "intake", "review"):
        raise HTTPException(status_code=400, detail="Invalid step")
    await _ensure_user_workspace(request, workspace_id)
    return await _reset_generate_task_step(workspace_id, task_id, step)


@router.post("/workspaces/{workspace_id}/generate/tasks/{task_id}/runs/{run_id}/fork")
async def fork_generate_task_run(request: Request, workspace_id: int, task_id: int, run_id: int):
    await _ensure_user_workspace(request, workspace_id)
    return await _fork_generate_task_run(workspace_id, task_id, run_id)


@router.delete("/workspaces/{workspace_id}/generate/tasks/{task_id}/runs/{run_id}")
async def delete_generate_task_run(request: Request, workspace_id: int, task_id: int, run_id: int):
    await _ensure_user_workspace(request, workspace_id)
    return await _delete_generate_task_run(workspace_id, task_id, run_id)


# ---------------------------------------------------------------------------
# Brand refresh
# ---------------------------------------------------------------------------

@router.post("/workspaces/{workspace_id}/generate/tasks/{task_id}/brand/refresh")
async def refresh_generate_task_branding(request: Request, workspace_id: int, task_id: int, body: BrandRefreshRequest):
    await _ensure_user_workspace(request, workspace_id)
    session = await _get_generate_task_session(workspace_id, task_id)
    branding = await _extract_branding_from_website(body.website_url)
    merged = dict(session.get("branding") or {})
    merged.update(branding)
    await _update_generate_task_session_row(
        workspace_id,
        task_id,
        GenerateTaskUpdateRequest(website_url=branding["website_url"], branding=merged),
    )
    return await _get_generate_task_session(workspace_id, task_id)


# ---------------------------------------------------------------------------
# Research refinement
# ---------------------------------------------------------------------------

@router.post("/workspaces/{workspace_id}/generate/tasks/{task_id}/research/refine")
async def refine_generate_task_research(request: Request, workspace_id: int, task_id: int, body: GenerateTaskResearchRequest):
    await _ensure_user_workspace(request, workspace_id)
    session = await _get_generate_task_session(workspace_id, task_id)
    mode = (body.mode or "quick").strip().lower()
    if mode not in ("quick", "deep"):
        raise HTTPException(status_code=400, detail="mode must be quick or deep")
    topic = _build_generate_task_research_topic(session, body.topic)
    result, meta = await _generate_research_refinement_questions(
        workspace_id,
        topic,
        mode,
        (body.research_type or "general").strip() or "general",
    )
    result["suggested_reframe"] = topic
    result["llm_provider"] = meta.get("provider")
    result["llm_model"] = meta.get("model")
    result["task_topic"] = session["todo"]["task"]
    _, document_evidence = await _build_task_document_evidence(
        workspace_id,
        session,
        extra_topics=[topic],
        limit=4,
    )
    context_brief = await _build_generate_question_context(workspace_id, session)
    document_block = _document_evidence_prompt_block(document_evidence)
    if document_block:
        context_brief = f"{context_brief}\n\n{document_block}"
    prefill_state = await _suggest_refinement_prefill(
        workspace_id,
        topic,
        mode,
        (body.research_type or "general").strip() or "general",
        result,
        context_brief=context_brief,
    )
    if prefill_state:
        result["prefill_state"] = prefill_state
    if document_evidence:
        result["source_document_refs"] = _dedupe_document_refs(document_evidence, limit=4)
    return result


# ---------------------------------------------------------------------------
# Questions / template / sections / autofill
# ---------------------------------------------------------------------------

@router.post("/workspaces/{workspace_id}/generate/tasks/{task_id}/questions")
async def generate_task_questions(request: Request, workspace_id: int, task_id: int):
    await _ensure_user_workspace(request, workspace_id)
    session = await _get_generate_task_session(workspace_id, task_id)
    return await _derive_generate_run_intake(workspace_id, task_id, session)


@router.post("/workspaces/{workspace_id}/generate/tasks/{task_id}/template/refine")
async def refine_generate_task_template(request: Request, workspace_id: int, task_id: int, body: dict):
    await _ensure_user_workspace(request, workspace_id)
    user_message = (body.get("message") or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message is required")
    session = await _get_generate_task_session(workspace_id, task_id)
    template_draft = session.get("template_draft") or {}
    chat_history = list(session.get("template_chat_history") or [])
    config = _get_template_config(session.get("artifact_template") or "requirements")
    template_label = config["label"]
    preferences = await _get_workspace_llm_preferences(request.app.state.db_pool, workspace_id)
    provider, model = _resolve_task_llm(preferences, "generate")
    current_draft_json = json.dumps(template_draft, indent=2)
    system_prompt = f"""\
You are a document outline assistant. You maintain and refine a structured document outline based on user requests.
The outline is stored as a template_draft with sections and fields for a {template_label}.

When the user requests a change, return a JSON object with:
1. "template_draft": the updated template_draft JSON (same schema as input, with the requested change applied)
2. "assistant_message": a short 1-2 sentence description of what you changed

Keep sections focused and coherent. Maintain the existing schema exactly.
Only return valid JSON - no markdown, no code blocks."""
    messages = []
    for item in chat_history[-8:]:
        role = str(item.get("role") or "user")
        if role not in ("user", "assistant"):
            role = "user"
        messages.append({"role": role, "content": str(item.get("content") or "")})
    messages.append({
        "role": "user",
        "content": f"Current outline:\n{current_draft_json}\n\nRequest: {user_message}"
    })
    try:
        result_payload, _ = await _call_llm_runner_json(
            messages,
            system=system_prompt,
            provider=provider,
            model=model,
            use_case="chat",
            max_tokens=2000,
            timeout=60.0,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Outline refinement failed: {exc}") from exc
    result_payload = _json_dict(result_payload)
    updated_draft_raw = result_payload.get("template_draft") or template_draft
    assistant_message = str(result_payload.get("assistant_message") or "Outline updated.").strip()
    updated_draft = _normalize_generate_template_draft(updated_draft_raw)
    if not updated_draft.get("sections"):
        updated_draft = _normalize_generate_template_draft(template_draft)
        assistant_message = "I wasn't able to update the outline structure. Could you rephrase your request?"
    now_iso = datetime.now(timezone.utc).isoformat()
    chat_history.append({"role": "user", "content": user_message, "ts": now_iso})
    chat_history.append({"role": "assistant", "content": assistant_message, "ts": now_iso})
    await _update_generate_task_session_row(
        workspace_id,
        task_id,
        GenerateTaskUpdateRequest(template_draft=updated_draft),
    )
    async with request.app.state.db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE generate_task_sessions SET template_chat_history = $1::jsonb, updated_at = NOW() WHERE id = $2",
            json.dumps(chat_history),
            task_id,
        )
    refreshed_session = await _get_generate_task_session(workspace_id, task_id)
    return {
        "assistant_message": assistant_message,
        "template_draft": updated_draft,
        "session": refreshed_session,
    }


@router.post("/workspaces/{workspace_id}/generate/tasks/{task_id}/sections")
async def add_custom_section(request: Request, workspace_id: int, task_id: int, body: AddCustomSectionRequest):
    """Add a custom section to the question plan, then run FTS + autofill for it."""
    await _ensure_user_workspace(request, workspace_id)
    heading = (body.heading or "").strip()
    if not heading:
        raise HTTPException(status_code=400, detail="heading is required")
    content = (body.content or "").strip()
    section_key = "custom_" + _slugify(heading, 40).replace("-", "_")
    if not section_key or section_key == "custom_":
        section_key = f"custom_section_{int(datetime.now(timezone.utc).timestamp())}"
    session = await _get_generate_task_session(workspace_id, task_id)
    existing_keys = {
        str(item.get("key") or "")
        for item in (session.get("question_plan") or [])
        if isinstance(item, dict)
    }
    if section_key in existing_keys:
        raise HTTPException(status_code=409, detail="A section with that name already exists")
    new_item = _sanitize_question_plan_item({
        "key": section_key,
        "label": heading,
        "group": heading,
        "input_type": "textarea",
        "required": False,
        "help": content,
    }, len(existing_keys) + 1)
    if not new_item:
        raise HTTPException(status_code=400, detail="Invalid section heading")
    updated_plan = list(session.get("question_plan") or []) + [new_item]
    answers_update: dict[str, Any] = {}
    if content:
        answers_update[section_key] = content
    await _update_generate_task_session_row(
        workspace_id, task_id,
        GenerateTaskUpdateRequest(question_plan=updated_plan, answers=answers_update or None),
    )
    session = await _get_generate_task_session(workspace_id, task_id)
    autofill_answers, answer_evidence = await _generate_task_answer_autofill(
        workspace_id, session,
        overwrite=True,
        question_keys=[section_key],
        question_guidance={section_key: content} if content else None,
    )
    if autofill_answers:
        answer_meta = _build_generate_answer_meta_patch(autofill_answers, "autofill")
        await _update_generate_task_session_row(
            workspace_id, task_id,
            GenerateTaskUpdateRequest(
                answers=autofill_answers,
                answer_evidence=answer_evidence,
                answer_meta=answer_meta,
            ),
        )
    updated_session = await _get_generate_task_session(workspace_id, task_id)
    return {
        "section_key": section_key,
        "autofill_answer": autofill_answers.get(section_key, ""),
        "session": updated_session,
    }


@router.post("/workspaces/{workspace_id}/generate/tasks/{task_id}/answers/autofill")
async def autofill_generate_task_answers(request: Request, workspace_id: int, task_id: int, body: GenerateTaskAutofillRequest):
    await _ensure_user_workspace(request, workspace_id)
    session = await _get_generate_task_session(workspace_id, task_id)
    guidance = (body.guidance or "").strip() or None
    question_keys = [str(key).strip() for key in (body.question_keys or []) if str(key).strip()]
    effective_question_keys = question_keys or [
        str(item.get("key") or "").strip()
        for item in session.get("question_plan") or []
        if isinstance(item, dict) and str(item.get("key") or "").strip()
    ]
    conflicts: list[dict[str, Any]] = []
    if body.overwrite and effective_question_keys and not body.overwrite_manual:
        conflicts = _collect_manual_answer_conflicts(session, effective_question_keys)
        blocked_keys = {item["question_key"] for item in conflicts}
        effective_question_keys = [key for key in effective_question_keys if key not in blocked_keys]
    question_guidance = {key: guidance for key in question_keys} if guidance and question_keys else None
    answers, answer_evidence = await _generate_task_answer_autofill(
        workspace_id,
        session,
        overwrite=bool(body.overwrite),
        question_keys=effective_question_keys or question_keys or body.question_keys,
        question_guidance=question_guidance,
    )
    question_research_update = None
    if question_guidance:
        existing_question_research = session.get("question_research") or {}
        question_research_update = {}
        now_iso = datetime.now(timezone.utc).isoformat()
        for key, value in question_guidance.items():
            merged_meta = dict(existing_question_research.get(key) or {})
            merged_meta["guidance"] = value
            merged_meta["updated_at"] = now_iso
            question_research_update[key] = merged_meta
    answer_meta_update = _build_generate_answer_meta_patch(
        answers,
        "refresh" if body.overwrite else "autofill",
        research_id=session.get("linked_research_id"),
    )
    if answers or answer_evidence or question_research_update:
        updated = await _update_generate_task_session_row(
            workspace_id,
            task_id,
            GenerateTaskUpdateRequest(
                answers=answers,
                answer_evidence=answer_evidence,
                answer_meta=answer_meta_update if answer_meta_update else None,
                question_research=question_research_update,
                current_step="intake",
            ),
        )
        return {"ok": True, "answers": answers, "conflicts": conflicts, "session": updated}
    return {"ok": True, "answers": {}, "conflicts": conflicts, "session": session}


@router.post("/workspaces/{workspace_id}/generate/tasks/{task_id}/question-research")
async def generate_task_question_research(request: Request, workspace_id: int, task_id: int, body: GenerateTaskQuestionResearchRequest):
    await _ensure_user_workspace(request, workspace_id)
    session = await _get_generate_task_session(workspace_id, task_id)
    question_key = (body.question_key or "").strip()
    if not question_key:
        raise HTTPException(status_code=400, detail="question_key is required")

    async def stream():
        try:
            async for chunk in _run_generate_task_question_research(
                workspace_id, task_id, session, question_key, body.guidance
            ):
                yield chunk
        except Exception as exc:
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@router.post("/workspaces/{workspace_id}/generate/tasks/{task_id}/question-chat")
async def generate_task_question_chat(request: Request, workspace_id: int, task_id: int, body: QuestionChatRequest):
    await _ensure_user_workspace(request, workspace_id)
    session = await _get_generate_task_session(workspace_id, task_id)
    question_key = (body.question_key or "").strip()
    message = (body.message or "").strip()
    if not question_key:
        raise HTTPException(status_code=400, detail="question_key is required")
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    async def stream():
        try:
            async for chunk in _run_generate_task_question_chat(
                workspace_id, task_id, session, question_key, message
            ):
                yield chunk
        except Exception as exc:
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@router.post("/workspaces/{workspace_id}/generate/tasks/{task_id}/research")
async def run_generate_task_research(request: Request, workspace_id: int, task_id: int, body: GenerateTaskResearchRequest):
    await _ensure_user_workspace(request, workspace_id)
    session = await _get_generate_task_session(workspace_id, task_id)

    async def stream():
        try:
            async for chunk in _run_generate_task_research_stream(
                workspace_id, task_id, session, body
            ):
                yield chunk
        except Exception as exc:
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@router.post("/workspaces/{workspace_id}/generate/tasks/{task_id}/deliverable")
async def generate_task_deliverable(request: Request, workspace_id: int, task_id: int, body: DeliverableRequest):
    await _ensure_user_workspace(request, workspace_id)
    session = await _get_generate_task_session(workspace_id, task_id)

    async def stream():
        try:
            async for chunk in _run_generate_task_deliverable_stream(
                workspace_id, task_id, session, body.output_type
            ):
                yield chunk
        except Exception as exc:
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@router.post("/workspaces/{workspace_id}/generate/tasks/{task_id}/test-intake")
async def test_intake_flow(request: Request, workspace_id: int, task_id: int):
    """Test endpoint: run research → approve outline → derive intake, return summary."""
    await _ensure_user_workspace(request, workspace_id)
    session = await _get_generate_task_session(workspace_id, task_id)
    # defer to the full complex logic in main_live
    return await _test_intake_flow_impl(request, workspace_id, task_id)


# ---------------------------------------------------------------------------
# Legacy generate (streaming document generation)
# ---------------------------------------------------------------------------

@router.post("/workspaces/{workspace_id}/generate")
async def generate_document(request: Request, workspace_id: int, body: GenerateRequest):
    await _ensure_user_workspace(request, workspace_id)
    output_type = (body.output_type or "document").strip().lower()
    if output_type not in ("document", "pdf", "pptx", "docx"):
        raise HTTPException(status_code=400, detail="output_type must be document, pdf, pptx, or docx")
    prompt_text = body.prompt.strip()
    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt is required")

    async def stream():
        yield json.dumps({"status": "Building context..."}) + "\n"
        context, warned = await _build_generation_context(
            workspace_id,
            body.meeting_ids,
            body.include_transcripts,
            body.include_document_ids,
            body.include_research_ids,
            body.include_todo_ids,
            body.include_todo_people,
        )
        if warned:
            yield json.dumps({"warning": "Context exceeded limit. Some longer content was excluded."}) + "\n"

        preferences = await _get_workspace_llm_preferences(request.app.state.db_pool, workspace_id)
        provider, model = _resolve_task_llm(preferences, "generate")

        base_title = (body.title or prompt_text[:60]).strip()
        safe_title = base_title or f"Generated {output_type.upper()}"

        try:
            if output_type == "document":
                context_for_generation = context
                if context:
                    _sections = _split_context_sections(context)
                    if len(_sections) >= 10:
                        yield json.dumps({"status": f"Extracting insights from {len(_sections)} sources..."}) + "\n"
                        context_for_generation = await _map_reduce_synthesis(
                            _sections, prompt_text, provider, model
                        )
                yield json.dumps({"status": "Generating document..."}) + "\n"
                generation_prompt = (
                    "Create a polished document in markdown. Return only the markdown body.\n\n"
                    "SYNTHESIS RULES - follow these strictly:\n"
                    "1. Do NOT summarize each source individually in sequence.\n"
                    "2. Identify 3 to 5 cross-cutting themes or insights that emerge across multiple sources.\n"
                    "3. For each theme, weave together evidence from multiple sources, noting where they agree, "
                    "diverge, or build on each other.\n"
                    "4. Cite sources inline (e.g. 'per [Document name]') when attributing specific claims.\n"
                    "5. Prioritize connection, pattern, and insight over exhaustive coverage.\n"
                    "6. Do not use em-dashes; use commas or reword instead.\n"
                    "7. Keep a warm, polite, and concise tone throughout.\n\n"
                    f"Document request:\n{prompt_text}\n\n"
                )
                if context_for_generation:
                    generation_prompt += f"Context:\n{context_for_generation}\n\n"
                result = await _call_llm_runner(
                    [{"role": "user", "content": generation_prompt}],
                    provider=provider,
                    model=model,
                    use_case="chat",
                    max_tokens=4200,
                )
                content = result["content"].strip()
                filename = f'{_slugify(safe_title, 70) or "generated-document"}.md'
                document = await _store_generated_document(
                    workspace_id,
                    filename,
                    "text/markdown",
                    content.encode("utf-8"),
                    content,
                )
                yield _json_line({
                    "result": {
                        "type": output_type,
                        "document": document,
                        "preview": content,
                        "download_url": f'/workspaces/{workspace_id}/documents/{document["id"]}/download',
                        "llm_provider": result.get("provider"),
                        "llm_model": result.get("model"),
                    }
                })
                return

            if output_type in ("pdf", "docx", "pptx"):
                yield json.dumps({"status": f"Generating {output_type.upper()} content..."}) + "\n"
                doc_prompt = (
                    f"Create a professional {output_type.upper()} document. Return ONLY valid JSON with keys "
                    '"title" and "sections" (or "slides" for pptx), where sections is an array of objects '
                    'with "heading" and "body".\n\n'
                    f"Document request:\n{prompt_text}\n\n"
                )
                if context:
                    doc_prompt += f"Context:\n{context}\n\n"
                payload, meta = await _call_llm_runner_json(
                    [{"role": "user", "content": doc_prompt}],
                    provider=provider,
                    model=model,
                    use_case="chat",
                    max_tokens=4200,
                )
                payload = _json_dict(payload)
                title = payload.get("title") or safe_title
                if output_type == "pdf":
                    sections = payload.get("sections") or []
                    file_bytes = await asyncio.to_thread(_build_pdf_bytes, title, sections)
                    mime = "application/pdf"
                    ext = "pdf"
                elif output_type == "docx":
                    sections = payload.get("sections") or []
                    file_bytes = await asyncio.to_thread(_build_docx_bytes, title, sections)
                    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    ext = "docx"
                else:  # pptx
                    slides = payload.get("slides") or payload.get("sections") or []
                    file_bytes = await asyncio.to_thread(_build_pptx_bytes, title, slides)
                    mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    ext = "pptx"
                extracted_text = title
                filename = f'{_slugify(title, 70) or f"generated-{output_type}"}.{ext}'
                document = await _store_generated_document(
                    workspace_id,
                    filename,
                    mime,
                    file_bytes,
                    extracted_text,
                )
                yield _json_line({
                    "result": {
                        "type": output_type,
                        "document": document,
                        "preview": extracted_text,
                        "download_url": f'/workspaces/{workspace_id}/documents/{document["id"]}/download',
                        "llm_provider": meta.get("provider"),
                        "llm_model": meta.get("model"),
                    }
                })
        except Exception as exc:
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
