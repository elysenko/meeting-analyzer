"""
Chat, chat-session, and LLM proxy routes.

GET/POST/PATCH/DELETE /workspaces/{id}/chat/*
POST /v1/chat/turn, /v1/chat/turn/stream
GET /v1/runtime/assignments, /v1/skills
POST /render-markdown, /chat
"""

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from starlette.responses import StreamingResponse

from models import (
    ChatRenderRequest, ChatRequest, ChatSessionCreateRequest,
    ChatSessionUpdateRequest, ChatTurnProxyRequest,
)
from services.workspace_svc import _ensure_user_workspace

router = APIRouter()
logger = logging.getLogger("meeting-analyzer")


# ---------------------------------------------------------------------------
# LLM runtime proxy
# ---------------------------------------------------------------------------

@router.get("/v1/runtime/assignments")
async def proxy_runtime_assignments(request: Request):
    from main_live import _llm_runner_proxy_get
    return await _llm_runner_proxy_get("/v1/runtime/assignments")


@router.get("/v1/skills")
async def proxy_skill_catalog(request: Request):
    from main_live import _llm_runner_proxy_get
    return await _llm_runner_proxy_get("/v1/skills")


# ---------------------------------------------------------------------------
# v1/chat/turn (non-streaming and streaming)
# ---------------------------------------------------------------------------

@router.post("/v1/chat/turn")
async def proxy_chat_turn(request: Request, body: ChatTurnProxyRequest):
    from main_live import (
        _prepare_chat_turn_request,
        _append_chat_session_message, _call_llm_runner,
    )
    if body.workspace_id is not None:
        await _ensure_user_workspace(request, body.workspace_id)
    if body.workspace_id is not None and body.attachment_ids:
        att_ids = [int(a["id"]) for a in body.attachment_ids if a.get("id") is not None]
        if att_ids:
            async with request.app.state.db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE chat_session_attachments SET activated = TRUE WHERE id = ANY($1::int[]) AND workspace_id = $2",
                    att_ids, body.workspace_id,
                )

    prepared = await _prepare_chat_turn_request(
        workspace_id=body.workspace_id,
        chat_session_id=body.chat_session_id,
        message=body.message,
        messages=body.messages,
        system=body.system,
        meeting_ids=body.meeting_ids,
        include_transcripts=body.include_transcripts,
        include_document_ids=body.include_document_ids,
        include_research_ids=body.include_research_ids,
    )
    messages = prepared["messages"]
    system_prompt = prepared["system_prompt"]
    warned = prepared["warned"]
    latest_user_message = prepared["latest_user_message"]
    if body.workspace_id is not None and body.chat_session_id is not None and latest_user_message:
        await _append_chat_session_message(body.workspace_id, body.chat_session_id, "user", latest_user_message,
                                           attachment_ids=body.attachment_ids or [])

    _is_qa_mode = bool((body.system or "").strip() and body.chat_session_id is None)
    _use_case = "voice" if _is_qa_mode else "chat"

    if _is_qa_mode and len(messages) >= 2:
        rag_marker = "[Retrieved context"
        second_last = messages[-2]
        last = messages[-1]
        if (
            second_last.get("role") == "user"
            and str(second_last.get("content", "")).startswith(rag_marker)
            and last.get("role") == "user"
        ):
            rag_text = second_last["content"]
            question_content = last["content"]
            if isinstance(question_content, list):
                merged_content: list[Any] = [
                    {"type": "text", "text": f"Context from meeting transcripts and documents:\n\n{rag_text}\n\n---"},
                ] + question_content
            else:
                merged_content = (
                    f"Context from meeting transcripts and documents:\n\n{rag_text}\n\n"
                    f"---\nQuestion: {question_content}"
                )
            messages = messages[:-2] + [{"role": "user", "content": merged_content}]

    try:
        result = await _call_llm_runner(
            messages,
            system=system_prompt,
            provider=(body.provider or "").strip() or None,
            model=(body.model or "").strip() or None,
            tools=body.tools or None,
            use_case=_use_case,
            max_tokens=body.max_tokens or 4096,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    stored_message = None
    if body.workspace_id is not None and body.chat_session_id is not None and result.get("content", "").strip():
        stored_message = await _append_chat_session_message(
            body.workspace_id,
            body.chat_session_id,
            "assistant",
            result.get("content", ""),
        )

    response = {
        "content": result.get("content", ""),
        "provider": result.get("provider"),
        "model": result.get("model"),
        "usage": result.get("usage", {}),
        "response_mode": result.get("response_mode"),
        "skill_name": result.get("skill_name"),
    }
    if stored_message is not None:
        response["message"] = stored_message
    if warned:
        response["warning"] = "Context exceeded limit. Some transcripts were excluded."
    return response


@router.post("/v1/chat/turn/stream")
async def proxy_chat_turn_stream(body: ChatTurnProxyRequest, request: Request):
    from main_live import (
        _prepare_chat_turn_request,
        _append_chat_session_message, _stream_llm_runner,
    )
    if body.workspace_id is not None:
        await _ensure_user_workspace(request, body.workspace_id)
    if body.workspace_id is not None and body.attachment_ids:
        att_ids = [int(a["id"]) for a in body.attachment_ids if a.get("id") is not None]
        if att_ids:
            async with request.app.state.db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE chat_session_attachments SET activated = TRUE WHERE id = ANY($1::int[]) AND workspace_id = $2",
                    att_ids, body.workspace_id,
                )
    prepared = await _prepare_chat_turn_request(
        workspace_id=body.workspace_id,
        chat_session_id=body.chat_session_id,
        message=body.message,
        messages=body.messages,
        system=body.system,
        meeting_ids=body.meeting_ids,
        include_transcripts=body.include_transcripts,
        include_document_ids=body.include_document_ids,
        include_research_ids=body.include_research_ids,
    )
    messages = prepared["messages"]
    system_prompt = prepared["system_prompt"]
    warned = prepared["warned"]
    latest_user_message = prepared["latest_user_message"]
    if body.workspace_id is not None and body.chat_session_id is not None and latest_user_message:
        await _append_chat_session_message(body.workspace_id, body.chat_session_id, "user", latest_user_message,
                                           attachment_ids=body.attachment_ids or [])

    async def stream():
        assistant_text = ""
        done_event: dict[str, Any] | None = None
        if warned:
            yield f"data: {json.dumps({'type': 'status', 'content': 'Context exceeded limit. Some transcripts were excluded.'}, ensure_ascii=False)}\n\n"
        try:
            async for event in _stream_llm_runner(
                messages,
                system=system_prompt,
                provider=(body.provider or "").strip() or None,
                model=(body.model or "").strip() or None,
                tools=body.tools or None,
                use_case="chat",
                max_tokens=body.max_tokens or 4096,
            ):
                event_type = event.get("type")
                if event_type == "text_delta":
                    assistant_text += event.get("content", "")
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    continue
                if event_type == "error":
                    raise RuntimeError(event.get("content") or "Unknown llm-runner error")
                if event_type == "done":
                    done_event = dict(event)
                    assistant_text = event.get("content") or assistant_text
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            if body.workspace_id is not None and body.chat_session_id is not None and assistant_text.strip():
                try:
                    await _append_chat_session_message(body.workspace_id, body.chat_session_id, "assistant", assistant_text)
                except Exception:
                    logger.warning("Failed to persist interrupted assistant message for workspace %s session %s", body.workspace_id, body.chat_session_id)
            raise
        except Exception as exc:
            error_event = {"type": "error", "content": str(exc)}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            return

        stored_message = None
        if body.workspace_id is not None and body.chat_session_id is not None and assistant_text.strip():
            stored_message = await _append_chat_session_message(body.workspace_id, body.chat_session_id, "assistant", assistant_text)
        done_payload = done_event or {"type": "done", "content": assistant_text}
        if stored_message is not None:
            done_payload["message"] = stored_message
        yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Chat session CRUD
# ---------------------------------------------------------------------------

@router.get("/workspaces/{workspace_id}/chat/history")
async def get_workspace_chat_history(request: Request, workspace_id: int):
    from main_live import (
        _get_latest_workspace_chat_session,
        _create_workspace_chat_session, _list_chat_session_messages,
    )
    await _ensure_user_workspace(request, workspace_id)
    session = await _get_latest_workspace_chat_session(workspace_id)
    if not session:
        session = await _create_workspace_chat_session(workspace_id)
    return {
        "session": session,
        "messages": await _list_chat_session_messages(workspace_id, session["id"]),
    }


@router.delete("/workspaces/{workspace_id}/chat/history")
async def clear_workspace_chat_history(request: Request, workspace_id: int):
    from main_live import _delete_all_workspace_chat_sessions
    await _ensure_user_workspace(request, workspace_id)
    await _delete_all_workspace_chat_sessions(workspace_id)
    return {"ok": True}


@router.get("/workspaces/{workspace_id}/chat/sessions")
async def list_workspace_chat_sessions(request: Request, workspace_id: int):
    from main_live import _list_workspace_chat_sessions
    await _ensure_user_workspace(request, workspace_id)
    return await _list_workspace_chat_sessions(workspace_id)


@router.post("/workspaces/{workspace_id}/chat/sessions")
async def create_workspace_chat_session(request: Request, workspace_id: int, body: ChatSessionCreateRequest | None = None):
    from main_live import _create_workspace_chat_session
    await _ensure_user_workspace(request, workspace_id)
    return await _create_workspace_chat_session(workspace_id, (body.title if body else None))


@router.get("/workspaces/{workspace_id}/chat/sessions/{session_id}")
async def get_workspace_chat_session(request: Request, workspace_id: int, session_id: int):
    from main_live import (
        _get_workspace_chat_session,
        _list_chat_session_messages,
    )
    await _ensure_user_workspace(request, workspace_id)
    session = await _get_workspace_chat_session(workspace_id, session_id)
    return {
        "session": session,
        "messages": await _list_chat_session_messages(workspace_id, session_id),
    }


@router.patch("/workspaces/{workspace_id}/chat/sessions/{session_id}")
async def update_workspace_chat_session(request: Request, workspace_id: int, session_id: int, body: ChatSessionUpdateRequest):
    from main_live import _update_workspace_chat_session
    await _ensure_user_workspace(request, workspace_id)
    return await _update_workspace_chat_session(workspace_id, session_id, body)


@router.delete("/workspaces/{workspace_id}/chat/sessions/{session_id}")
async def delete_workspace_chat_session(request: Request, workspace_id: int, session_id: int):
    from main_live import _delete_workspace_chat_session
    await _ensure_user_workspace(request, workspace_id)
    await _delete_workspace_chat_session(workspace_id, session_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Chat attachments
# ---------------------------------------------------------------------------

@router.get("/workspaces/{workspace_id}/chat/sessions/{session_id}/attachments")
async def list_chat_attachments(request: Request, workspace_id: int, session_id: int):
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, workspace_id, chat_session_id, filename, mime_type, file_size,
                      status, error_message, activated, created_at
               FROM chat_session_attachments
               WHERE workspace_id = $1 AND chat_session_id = $2
               ORDER BY created_at""",
            workspace_id, session_id,
        )
    return [dict(r) for r in rows]


@router.post("/workspaces/{workspace_id}/chat/sessions/{session_id}/attachments")
async def upload_chat_attachment(
    request: Request,
    workspace_id: int,
    session_id: int,
    file: UploadFile = File(...),
):
    from main_live import (
        _get_workspace_chat_session, _extract_text_sync,
    )
    await _ensure_user_workspace(request, workspace_id)
    await _get_workspace_chat_session(workspace_id, session_id)
    data = await file.read()
    filename = file.filename or "attachment"
    mime_type = file.content_type or ""
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO chat_session_attachments
               (workspace_id, chat_session_id, filename, mime_type, file_size, status)
               VALUES ($1, $2, $3, $4, $5, 'processing')
               RETURNING id, workspace_id, chat_session_id, filename, mime_type, file_size, status, created_at""",
            workspace_id, session_id, filename, mime_type, len(data),
        )
    attachment_id = row["id"]

    pool = request.app.state.db_pool

    async def _extract():
        try:
            loop = asyncio.get_event_loop()
            extracted = await loop.run_in_executor(
                None, _extract_text_sync, data, mime_type, filename
            )
            extracted = extracted or ""
            max_chars = 120_000
            status = "truncated" if len(extracted) > max_chars else "ready"
            if status == "truncated":
                extracted = extracted[:max_chars]
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE chat_session_attachments SET extracted_text = $1, status = $2 WHERE id = $3",
                    extracted, status, attachment_id,
                )
        except Exception as exc:
            logger.error("chat attachment extraction failed id=%d: %s", attachment_id, exc)
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE chat_session_attachments SET status = 'failed', error_message = $1 WHERE id = $2",
                    str(exc)[:500], attachment_id,
                )

    asyncio.create_task(_extract())
    return dict(row)


@router.delete("/workspaces/{workspace_id}/chat/sessions/{session_id}/attachments/{attachment_id}")
async def delete_chat_attachment(
    request: Request, workspace_id: int, session_id: int, attachment_id: int
):
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM chat_session_attachments WHERE id = $1 AND workspace_id = $2 AND chat_session_id = $3",
            attachment_id, workspace_id, session_id,
        )
    return {"ok": True}


@router.post("/workspaces/{workspace_id}/chat/sessions/{session_id}/document")
async def chat_generate_document(
    request: Request,
    workspace_id: int,
    session_id: int,
    body: ChatTurnProxyRequest,
    format: str = "pdf",
):
    """Generate a document (pdf/docx/pptx) from a chat prompt and stream progress events."""
    from main_live import (
        _prepare_chat_turn_request,
        _append_chat_session_message, _generate_structured_document,
        _json_default,
    )
    fmt = format.strip().lower()
    if fmt not in ("pdf", "docx", "pptx"):
        fmt = "pdf"
    await _ensure_user_workspace(request, workspace_id)

    async def _stream():
        try:
            if body.attachment_ids:
                att_ids = [int(a["id"]) for a in body.attachment_ids if a.get("id") is not None]
                if att_ids:
                    async with request.app.state.db_pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE chat_session_attachments SET activated = TRUE"
                            " WHERE id = ANY($1::int[]) AND workspace_id = $2",
                            att_ids, workspace_id,
                        )

            yield f"data: {json.dumps({'type': 'status', 'content': 'Building context...'}, ensure_ascii=False)}\n\n"

            prepared = await _prepare_chat_turn_request(
                workspace_id=workspace_id,
                chat_session_id=session_id,
                message=body.message,
                messages=body.messages,
                system=None,
                meeting_ids=body.meeting_ids,
                include_transcripts=body.include_transcripts,
                include_document_ids=body.include_document_ids,
                include_research_ids=body.include_research_ids,
            )
            user_text = prepared["latest_user_message"] or body.message or ""
            if user_text:
                await _append_chat_session_message(
                    workspace_id, session_id, "user", user_text,
                    attachment_ids=body.attachment_ids or [],
                )

            context_parts = [
                msg["content"] for msg in prepared["messages"]
                if msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
                and msg["content"].startswith("[Retrieved context")
            ]
            context_block = "\n\n".join(context_parts).strip()
            context_section = f"Workspace context:\n{context_block}\n\n" if context_block else ""
            safe_title = (user_text or "Generated Document")[:200]

            prior_user_msgs = [
                msg["content"] for msg in prepared["messages"]
                if msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
                and not msg["content"].startswith("[Retrieved context")
                and msg["content"].strip() != user_text.strip()
            ]
            prior_user_msgs = prior_user_msgs[-10:]
            if prior_user_msgs:
                history_lines = "\n".join(
                    f"{i + 1}. {m}" for i, m in enumerate(prior_user_msgs)
                )
                history_section = (
                    "Prior requests in this conversation (oldest first):\n"
                    + history_lines
                    + "\n\n"
                )
            else:
                history_section = ""

            yield f"data: {json.dumps({'type': 'status', 'content': 'Generating document content...'}, ensure_ascii=False)}\n\n"

            if fmt == "pptx":
                generation_prompt = (
                    "You are generating a presentation based on the user's request"
                    " and any available workspace context.\n\n"
                    + history_section
                    + f"User request: {user_text}\n\n"
                    + context_section
                    + "Return ONLY valid JSON with:\n"
                    '- "title": string — presentation title\n'
                    '- "slides": array of objects, each with:\n'
                    '  - "title": string — slide title\n'
                    '  - "bullets": array of strings — slide bullet points\n\n'
                    "Produce a complete presentation with a title slide and 6-10 content slides."
                )
            else:
                generation_prompt = (
                    "You are generating a well-structured document based on the user's request"
                    " and any available workspace context.\n\n"
                    + history_section
                    + f"User request: {user_text}\n\n"
                    + context_section
                    + "Return ONLY valid JSON with:\n"
                    '- "title": string — concise document title\n'
                    '- "sections": array of objects, each with:\n'
                    '  - "heading": string — section heading\n'
                    '  - "body": string — section body text (markdown: bullets, bold, '
                    'tables with a reasonable number of columns so cells wrap cleanly)\n\n'
                    "Produce a complete, thorough document with multiple sections."
                )

            yield f"data: {json.dumps({'type': 'status', 'content': 'Building ' + fmt.upper() + '...'}, ensure_ascii=False)}\n\n"

            result = await _generate_structured_document(
                workspace_id,
                output_type=fmt,
                safe_title=safe_title,
                generation_prompt=generation_prompt,
                branding={},
            )
            document = result["document"]
            download_url = result["download_url"]
            filename = document["filename"]

            assistant_content = f"Here's your document: **{filename}**\n\n[Download]({download_url})"
            stored_msg = await _append_chat_session_message(
                workspace_id, session_id, "assistant", assistant_content,
            )

            yield f"data: {json.dumps({'type': 'done', 'document': document, 'download_url': download_url, 'filename': filename, 'format': fmt, 'message': stored_msg}, default=_json_default, ensure_ascii=False)}\n\n"

        except Exception as exc:
            logger.error("chat document generation failed for workspace %s: %s", workspace_id, exc)
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Render markdown and legacy chat endpoint
# ---------------------------------------------------------------------------

@router.post("/render-markdown")
async def render_markdown(body: ChatRenderRequest):
    from main_live import _render_markdown_html
    return {"html": _render_markdown_html(body.text)}


@router.post("/chat")
async def chat(request: Request, body: ChatRequest):
    from main_live import (
        _get_workspace_chat_session,
        _prepare_chat_turn_request, _append_chat_session_message,
        _list_chat_session_messages, _get_workspace_llm_preferences,
        _resolve_task_llm, _stream_llm_runner, _json_line,
    )
    if body.workspace_id is None:
        raise HTTPException(status_code=400, detail="workspace_id is required.")
    if body.chat_session_id is None:
        raise HTTPException(status_code=400, detail="chat_session_id is required.")
    await _ensure_user_workspace(request, body.workspace_id)
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question is required.")
    await _get_workspace_chat_session(body.workspace_id, body.chat_session_id)

    prepared = await _prepare_chat_turn_request(
        workspace_id=body.workspace_id,
        chat_session_id=body.chat_session_id,
        message=body.question,
        meeting_ids=body.meeting_ids,
        include_transcripts=body.include_transcripts,
        include_document_ids=body.include_document_ids,
        include_research_ids=body.include_research_ids,
    )
    system_prompt = prepared["system_prompt"]
    warned = prepared["warned"]
    await _append_chat_session_message(body.workspace_id, body.chat_session_id, "user", body.question.strip())
    history_rows = await _list_chat_session_messages(body.workspace_id, body.chat_session_id)
    history_messages = [
        {"role": item["role"], "content": item["content"]}
        for item in history_rows
        if item.get("role") in {"user", "assistant"}
    ]
    chat_preferences = await _get_workspace_llm_preferences(body.workspace_id)
    chat_provider, chat_model = _resolve_task_llm(chat_preferences, "chat")

    async def stream():
        if warned:
            yield json.dumps({"warning": "Context exceeded limit. Some transcripts were excluded."}) + "\n"

        assistant_text = ""
        try:
            async for event in _stream_llm_runner(
                history_messages,
                system=system_prompt,
                provider=chat_provider,
                model=chat_model,
                use_case="chat",
                max_tokens=4096,
            ):
                if event.get("type") == "text_delta":
                    token = event.get("content", "")
                    assistant_text += token
                    yield json.dumps({"token": token}) + "\n"
                elif event.get("type") == "status":
                    continue
                elif event.get("type") == "error":
                    raise RuntimeError(event.get("content") or "Unknown llm-runner error")
        except Exception as e:
            if assistant_text.strip():
                await _append_chat_session_message(body.workspace_id, body.chat_session_id, "assistant", assistant_text)
            yield json.dumps({"error": str(e)}) + "\n"
            return

        stored_message = await _append_chat_session_message(body.workspace_id, body.chat_session_id, "assistant", assistant_text)
        yield _json_line({"done": True, "message": stored_message})

    return StreamingResponse(stream(), media_type="application/x-ndjson")
