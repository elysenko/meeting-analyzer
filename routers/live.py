"""
Live / real-time routes.

POST /workspaces/{id}/live-qa, GET /workspaces/{id}/live-qa/{session_id}/entries
POST /analyze-text
WebSocket /ws/tts-stream, WebSocket /ws/transcribe
POST /livekit/token, GET /livekit/rooms
"""

import asyncio
import json
import logging
import time as _time
import uuid

import httpx
import websockets
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from jose import jwt
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from config import (
    LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_INTERNAL_URL,
    LIVEKIT_WS_URL, WHISPER_LIVE_URL,
)
from llm import _stream_llm_runner
from models import AnalyzeTextRequest, LiveQARequest
from services.documents_svc import retrieve_document_evidence as _retrieve_document_evidence
from services.documents_svc import retrieve_meeting_evidence as _retrieve_meeting_evidence
from services.meetings_svc import analyze_with_llm as _meetings_analyze_with_llm
from services.meetings_svc import save_meeting as _meetings_save_meeting
from services.text_svc import _excerpt_text
from services.utils import _json_line
from services.workspace_svc import _ensure_user_workspace

router = APIRouter()
logger = logging.getLogger("meeting-analyzer")

LIVE_QA_SYSTEM_PROMPT = (
    "Answer the question using the provided context. "
    "Cite which document or meeting the information comes from. "
    "If the context does not contain the answer, say so."
)

LIVE_QA_LOW_CONFIDENCE_PROMPT = (
    "The retrieved context does not strongly match the question. "
    "Indicate that you don't have enough information from the selected documents to fully answer, "
    "but share what limited context was found if any seems relevant. "
    "Be honest about the uncertainty."
)


# ---------------------------------------------------------------------------
# Live Q&A
# ---------------------------------------------------------------------------

@router.post("/workspaces/{workspace_id}/live-qa")
async def post_live_qa(request: Request, workspace_id: int, body: LiveQARequest):
    await _ensure_user_workspace(request, workspace_id)
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    session_id = body.session_id
    if not session_id:
        async with request.app.state.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO live_qa_sessions
                   (workspace_id, context_meeting_ids, context_document_ids, context_research_ids)
                   VALUES ($1, $2::jsonb, $3::jsonb, $4::jsonb)
                   RETURNING id""",
                workspace_id,
                json.dumps(body.meeting_ids),
                json.dumps(body.document_ids),
                json.dumps(body.research_ids),
            )
            session_id = row["id"]

    pool = request.app.state.db_pool

    async def stream():
        from typing import Any
        doc_evidence: list[dict[str, Any]] = []
        meeting_evidence: list[dict[str, Any]] = []
        research_context: list[dict[str, Any]] = []

        if body.document_ids:
            try:
                doc_evidence = await _retrieve_document_evidence(pool, workspace_id, body.document_ids, question)
            except Exception as exc:
                logger.warning("Document evidence retrieval failed: %s", exc)

        if body.meeting_ids:
            try:
                meeting_evidence = await _retrieve_meeting_evidence(pool, workspace_id, body.meeting_ids, question)
            except Exception as exc:
                logger.warning("Meeting evidence retrieval failed: %s", exc)

        if body.research_ids:
            try:
                async with pool.acquire() as conn:
                    r_rows = await conn.fetch(
                        """SELECT id, title, topic, summary, content
                           FROM research_sessions
                           WHERE workspace_id = $1 AND id = ANY($2::int[])""",
                        workspace_id,
                        [int(r) for r in body.research_ids if r],
                    )
                for r in r_rows:
                    research_context.append({
                        "research_id": int(r["id"]),
                        "title": r["title"] or r["topic"],
                        "summary": _excerpt_text(r["summary"] or r["content"], 500),
                    })
            except Exception as exc:
                logger.warning("Research retrieval failed: %s", exc)

        all_scores = [ev.get("score", 0.0) for ev in doc_evidence] + [ev.get("score", 0.0) for ev in meeting_evidence]
        max_score = max(all_scores) if all_scores else 0.0

        if max_score >= 0.7:
            confidence_tier = "high"
        elif max_score >= 0.5:
            confidence_tier = "medium"
        else:
            confidence_tier = "low"

        low_confidence = max_score < body.confidence_threshold and len(all_scores) > 0

        context_parts: list[str] = []

        doc_chars = 0
        for ev in doc_evidence:
            chunk_text = ev.get("content") or ev.get("snippet", "")
            if doc_chars > 0 and doc_chars + len(chunk_text) > 4000:
                break
            context_parts.append(f'[Document: {ev.get("filename", "unknown")}]\n{chunk_text}')
            doc_chars += len(chunk_text)

        mtg_chars = 0
        for ev in meeting_evidence:
            chunk_text = ev.get("content") or ev.get("snippet", "")
            if mtg_chars > 0 and mtg_chars + len(chunk_text) > 4000:
                break
            context_parts.append(f'[Meeting: {ev.get("meeting_title", "unknown")}]\n{chunk_text}')
            mtg_chars += len(chunk_text)

        for r in research_context[:3]:
            context_parts.append(f'[Research: {r["title"]}]\n{r["summary"]}')

        if body.transcript_context:
            context_parts.append(f'[Recent Transcript]\n{body.transcript_context[:500]}')

        context_block = "\n\n".join(context_parts)
        user_message = f"Context:\n{context_block}\n\nQuestion: {question}" if context_block else question

        system_prompt = LIVE_QA_LOW_CONFIDENCE_PROMPT if low_confidence else LIVE_QA_SYSTEM_PROMPT

        sources: list[dict[str, Any]] = []
        for ev in doc_evidence:
            sources.append({"type": "document", "id": ev.get("document_id"), "name": ev.get("filename")})
        for ev in meeting_evidence:
            sources.append({"type": "meeting", "id": ev.get("meeting_id"), "name": ev.get("meeting_title")})
        for r in research_context:
            sources.append({"type": "research", "id": r.get("research_id"), "name": r.get("title")})
        seen_sources: set[tuple] = set()
        unique_sources: list[dict[str, Any]] = []
        for s in sources:
            key = (s["type"], s["id"])
            if key not in seen_sources:
                seen_sources.add(key)
                unique_sources.append(s)

        yield json.dumps({"type": "session", "session_id": session_id}) + "\n"

        assistant_text = ""
        try:
            async for event in _stream_llm_runner(
                [{"role": "user", "content": user_message}],
                system=system_prompt,
                use_case="voice",
                max_tokens=2048,
            ):
                event_type = event.get("type")
                if event_type == "text_delta":
                    token = event.get("content", "")
                    assistant_text += token
                    yield json.dumps({"type": "text_delta", "content": token}) + "\n"
                elif event_type == "error":
                    raise RuntimeError(event.get("content") or "Unknown LLM error")
        except Exception as exc:
            yield json.dumps({"type": "error", "content": str(exc)}) + "\n"
            return

        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO live_qa_entries
                       (session_id, question, answer, sources, detected, transcript_context)
                       VALUES ($1, $2, $3, $4::jsonb, $5, $6)""",
                    session_id,
                    question,
                    assistant_text,
                    json.dumps(unique_sources),
                    body.auto_lookup,
                    body.transcript_context,
                )
        except Exception as exc:
            logger.warning("Failed to persist Q&A entry: %s", exc)

        yield json.dumps({
            "type": "done",
            "sources": unique_sources,
            "confidence": confidence_tier,
            "max_score": round(max_score, 3),
            "low_confidence": low_confidence,
            "auto_lookup": body.auto_lookup
        }) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@router.get("/workspaces/{workspace_id}/live-qa/{session_id}/entries")
async def get_live_qa_entries(request: Request, workspace_id: int, session_id: int):
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        session = await conn.fetchrow(
            "SELECT id FROM live_qa_sessions WHERE id = $1 AND workspace_id = $2",
            session_id,
            workspace_id,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        rows = await conn.fetch(
            """SELECT id, question, answer, sources, detected, created_at
               FROM live_qa_entries
               WHERE session_id = $1
               ORDER BY created_at ASC""",
            session_id,
        )
    return [
        {
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "sources": json.loads(row["sources"]) if isinstance(row["sources"], str) else row["sources"],
            "detected": row["detected"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Analyze text
# ---------------------------------------------------------------------------

@router.post("/analyze-text")
async def analyze_text(request: Request, body: AnalyzeTextRequest):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")

    pool = request.app.state.db_pool

    async def stream():
        yield json.dumps({"status": "Analyzing with selected model..."}) + "\n"
        try:
            analysis, _ = await _meetings_analyze_with_llm(pool, text, body.workspace_id)
        except Exception as e:
            logger.error("LLM analysis failed for text input: %s", e)
            yield json.dumps({"error": f"Analysis failed: {e}"}) + "\n"
            return

        try:
            meeting_id = await _meetings_save_meeting(
                pool, "live-transcription", text, analysis, body.workspace_id,
                user_id=getattr(request.state, "user_id", None),
            )
        except Exception as e:
            logger.error("Save failed for text input: %s", e)
            yield json.dumps({"error": f"Failed to save meeting: {e}"}) + "\n"
            return

        result = {
            "id": meeting_id,
            "transcript": text,
            "summary": analysis.get("summary", ""),
            "action_items": analysis.get("action_items", []),
            "todos": analysis.get("todos", []),
            "email_body": analysis.get("email_body", ""),
        }
        yield _json_line({"result": result})

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# TTS streaming WebSocket
# ---------------------------------------------------------------------------

@router.websocket("/ws/tts-stream")
async def ws_tts_stream(ws: WebSocket):
    from main_live import (
        _ensure_ws_user_workspace, _get_workspace_chat_session,
        _prepare_chat_turn_request, _append_chat_session_message,
        _list_chat_session_messages, _get_workspace_llm_preferences,
        _resolve_task_llm, _stream_llm_runner, SentenceBuffer, synthesize_and_send,
    )
    await ws.accept()
    try:
        if not ws.app.state.piper_voice:
            await ws.send_json({"type": "error", "message": "Streaming TTS not available — Piper model not loaded"})
            await ws.close()
            return

        request_data = await ws.receive_json()
        question = request_data.get("question", "").strip()
        messages = request_data.get("messages") if isinstance(request_data.get("messages"), list) else None
        meeting_ids = request_data.get("meeting_ids", [])
        include_transcripts = request_data.get("include_transcripts", [])
        include_document_ids = request_data.get("include_document_ids", [])
        include_research_ids = request_data.get("include_research_ids", [])
        workspace_id = request_data.get("workspace_id")
        chat_session_id = request_data.get("chat_session_id")

        if workspace_id is None:
            await ws.send_json({"type": "error", "message": "workspace_id is required"})
            await ws.close()
            return
        if chat_session_id is None:
            await ws.send_json({"type": "error", "message": "chat_session_id is required"})
            await ws.close()
            return
        try:
            chat_session_id = int(chat_session_id)
        except Exception:
            await ws.send_json({"type": "error", "message": "chat_session_id must be an integer"})
            await ws.close()
            return
        await _ensure_ws_user_workspace(ws, workspace_id)
        await _get_workspace_chat_session(workspace_id, chat_session_id)
        prepared = await _prepare_chat_turn_request(
            workspace_id=workspace_id,
            chat_session_id=chat_session_id,
            message=question,
            messages=messages,
            meeting_ids=meeting_ids,
            include_transcripts=include_transcripts,
            include_document_ids=include_document_ids,
            include_research_ids=include_research_ids,
        )
        system_prompt = prepared["system_prompt"]
        warned = prepared["warned"]
        latest_user_message = prepared["latest_user_message"]
        if latest_user_message:
            await _append_chat_session_message(workspace_id, chat_session_id, "user", latest_user_message)
        if messages:
            history_messages = prepared["messages"]
        else:
            history_rows = await _list_chat_session_messages(workspace_id, chat_session_id)
            history_messages = [
                {"role": item["role"], "content": item["content"]}
                for item in history_rows
                if item.get("role") in {"user", "assistant"}
            ]

        if warned:
            await ws.send_json({"type": "warning", "message": "Context exceeded limit. Some transcripts were excluded."})

        await ws.send_json({
            "type": "audio_config",
            "sample_rate": ws.app.state.piper_voice.config.sample_rate,
            "channels": 1,
            "bit_depth": 16,
            "encoding": "pcm_s16le",
        })

        sentence_buffer = SentenceBuffer()
        text_parts = []

        tts_preferences = await _get_workspace_llm_preferences(workspace_id)
        tts_provider, tts_model = _resolve_task_llm(tts_preferences, "chat")
        async for event in _stream_llm_runner(
            history_messages,
            system=system_prompt,
            provider=tts_provider,
            model=tts_model,
            use_case="chat",
            max_tokens=4096,
        ):
            if event.get("type") != "text_delta":
                if event.get("type") == "error":
                    raise RuntimeError(event.get("content") or "Unknown llm-runner error")
                continue
            token = event.get("content", "")
            text_parts.append(token)
            await ws.send_json({"type": "token", "text": token})

            for sentence in sentence_buffer.add_token(token):
                await ws.send_json({"type": "sentence", "text": sentence})
                await synthesize_and_send(sentence, ws)

        remaining = sentence_buffer.flush()
        if remaining:
            await ws.send_json({"type": "sentence", "text": remaining})
            await synthesize_and_send(remaining, ws)

        stored_message = await _append_chat_session_message(workspace_id, chat_session_id, "assistant", "".join(text_parts))
        await ws.send_json({"type": "done", "full_text": "".join(text_parts), "message": stored_message})

    except WebSocketDisconnect:
        logger.info("TTS stream client disconnected")
    except Exception as e:
        logger.error("TTS stream error: %s", e)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LiveKit
# ---------------------------------------------------------------------------

class LiveKitTokenRequest(BaseModel):
    room: str = "meeting"
    identity: str | None = None


def _livekit_token(claims: dict) -> str:
    return jwt.encode(claims, LIVEKIT_API_SECRET, algorithm="HS256")


@router.post("/livekit/token")
async def livekit_token(body: LiveKitTokenRequest):
    """Generate a LiveKit JWT for a subscriber joining a room."""
    identity = body.identity or f"listener-{uuid.uuid4().hex[:8]}"
    now = int(_time.time())
    claims = {
        "video": {
            "roomJoin": True,
            "room": body.room,
            "canPublish": False,
            "canSubscribe": True,
        },
        "sub": identity,
        "iss": LIVEKIT_API_KEY,
        "exp": now + 3600,
        "nbf": now,
        "jti": uuid.uuid4().hex,
    }
    token = _livekit_token(claims)
    return {"token": token, "url": LIVEKIT_WS_URL, "identity": identity}


@router.get("/livekit/rooms")
async def livekit_rooms():
    """List active LiveKit rooms using the internal admin API."""
    now = int(_time.time())
    claims = {
        "video": {"roomList": True},
        "sub": "server",
        "iss": LIVEKIT_API_KEY,
        "exp": now + 60,
        "nbf": now,
        "jti": uuid.uuid4().hex,
    }
    admin_token = _livekit_token(claims)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{LIVEKIT_INTERNAL_URL}/twirp/livekit.RoomService/ListRooms",
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Content-Type": "application/json",
                },
                json={},
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return {"rooms": [r.get("name") for r in data.get("rooms", [])]}
    except Exception as e:
        logger.warning("Failed to list LiveKit rooms: %s", e)
    return {"rooms": []}


# ---------------------------------------------------------------------------
# WebSocket transcription proxy to WhisperLive
# ---------------------------------------------------------------------------

@router.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await ws.accept()
    whisper_ws = None

    try:
        init_data = await ws.receive_json()
        if init_data.get("action") != "start":
            await ws.send_json({"error": "Expected {action: 'start'}"})
            await ws.close()
            return

        language = init_data.get("language", "en")
        uid = str(uuid.uuid4())

        last_connect_error: Exception | None = None
        for attempt in range(3):
            try:
                whisper_ws = await websockets.connect(
                    WHISPER_LIVE_URL,
                    open_timeout=10,
                    close_timeout=5,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=None,
                )
                break
            except Exception as exc:
                last_connect_error = exc
                if attempt >= 2:
                    raise RuntimeError(f"Failed to connect to WhisperLive: {exc}") from exc
                await asyncio.sleep(0.25 * (attempt + 1))
        if whisper_ws is None:
            raise RuntimeError(f"Failed to connect to WhisperLive: {last_connect_error}")

        config = {
            "uid": uid,
            "language": language,
            "model": "base",
            "task": "transcribe",
            "use_vad": True,
            "same_output_threshold": 3,
        }
        await whisper_ws.send(json.dumps(config))

        async def relay_from_whisper():
            try:
                async for message in whisper_ws:
                    if isinstance(message, str):
                        data = json.loads(message)
                        await ws.send_json(data)
            except websockets.ConnectionClosed:
                pass
            except WebSocketDisconnect:
                pass

        relay_task = asyncio.create_task(relay_from_whisper())

        try:
            while True:
                message = await ws.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                if "text" in message:
                    data = json.loads(message["text"])
                    if data.get("action") == "stop":
                        await whisper_ws.send(b"END_OF_AUDIO")
                        await asyncio.sleep(0.5)
                        await ws.send_json({"done": True})
                        break

                if "bytes" in message and message["bytes"]:
                    await whisper_ws.send(message["bytes"])

        except WebSocketDisconnect:
            pass
        finally:
            relay_task.cancel()
            try:
                await relay_task
            except asyncio.CancelledError:
                pass

    except websockets.ConnectionClosedOK:
        pass
    except Exception as e:
        logger.error("WebSocket transcribe error: %s", e)
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        if whisper_ws:
            await whisper_ws.close()
