"""
Integration routes: Drive, Canvas, TTS, team-chat.js, library, analyze/jobs, admin.

GET/POST /drive/*, /canvas/*
POST /tts
GET /team-chat.js
GET/POST /library/*, /workspaces/{id}/link, /workspaces/{id}/unlink/*, /workspaces/{id}/linked-content
GET/POST /workspaces/{id}/drive-*, /workspaces/{id}/canvas-*
POST /analyze-async, GET /jobs/{id}, POST /analyze
POST /admin/*
"""

import asyncio
import json
import logging
import os
import tempfile

from datetime import datetime, timezone

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from starlette.responses import StreamingResponse

from config import MINIO_BUCKET, PIPER_TTS_URL
from models import TTSRequest
from services import meetings_svc as _meetings_svc
from services.storage import get_minio_client
from services.utils import _json_line

# Consolidated main_live imports — surfaced at startup rather than per-request.
# Drive/Canvas/Library/admin handlers live in main_live until E9 migrates them.
from main_live import (  # noqa: E402
    admin_backfill_office_ocr as _ml_admin_backfill_office_ocr,
    canvas_connect as _ml_canvas_connect,
    canvas_list_courses as _ml_canvas_courses,
    canvas_status as _ml_canvas_status,
    drive_callback as _ml_drive_callback,
    drive_files as _ml_drive_files,
    drive_import as _ml_drive_import,
    drive_status as _ml_drive_status,
    library_list as _ml_library_list,
    library_share as _ml_library_share,
    library_share_get as _ml_library_share_get,
    library_upload_document as _ml_library_upload_document,
    team_chat_js as _ml_team_chat_js,
    workspace_canvas_link as _ml_workspace_canvas_link,
    workspace_canvas_status as _ml_workspace_canvas_status,
    workspace_canvas_sync as _ml_workspace_canvas_sync,
    workspace_canvas_unlink as _ml_workspace_canvas_unlink,
    workspace_drive_link as _ml_workspace_drive_link,
    workspace_drive_status as _ml_workspace_drive_status,
    workspace_drive_sync as _ml_workspace_drive_sync,
    workspace_drive_unlink as _ml_workspace_drive_unlink,
    workspace_link_content as _ml_workspace_link_content,
    workspace_linked_content as _ml_workspace_linked_content,
    workspace_unlink_content as _ml_workspace_unlink_content,
)

router = APIRouter()
logger = logging.getLogger("meeting-analyzer")


# ---------------------------------------------------------------------------
# TTS proxy
# ---------------------------------------------------------------------------

@router.post("/tts")
async def tts_proxy(request: Request, body: TTSRequest):
    import httpx
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")

    if request.app.state.piper_voice:
        import io
        import wave

        def _synthesize_wav():
            import numpy as np
            pcm_chunks = []
            for audio_chunk in request.app.state.piper_voice.synthesize(text):
                pcm_chunks.append((audio_chunk.audio_float_array * 32767).astype(np.int16).tobytes())
            pcm = b"".join(pcm_chunks)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(request.app.state.piper_voice.config.sample_rate)
                wf.writeframes(pcm)
            return buf.getvalue()

        wav_bytes = await asyncio.to_thread(_synthesize_wav)
        return StreamingResponse(
            iter([wav_bytes]),
            media_type="audio/wav",
            headers={"Content-Disposition": "inline"},
        )

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            PIPER_TTS_URL,
            json={"text": text},
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="TTS service error.")
        return StreamingResponse(
            iter([resp.content]),
            media_type="audio/wav",
            headers={"Content-Disposition": "inline"},
        )


# ---------------------------------------------------------------------------
# Google Drive integration
# ---------------------------------------------------------------------------

@router.get("/drive/status")
async def drive_status(request: Request):
    return await _ml_drive_status(request)


@router.get("/drive/callback")
async def drive_callback(request: Request):
    return await _ml_drive_callback(request)


@router.get("/drive/files")
async def drive_files(request: Request):
    return await _ml_drive_files(request)


@router.post("/workspaces/{workspace_id}/drive/import")
async def drive_import(request: Request, workspace_id: int):
    return await _ml_drive_import(request, workspace_id)


@router.post("/workspaces/{workspace_id}/drive-link")
async def workspace_drive_link(request: Request, workspace_id: int):
    return await _ml_workspace_drive_link(request, workspace_id)


@router.delete("/workspaces/{workspace_id}/drive-link")
async def workspace_drive_unlink(request: Request, workspace_id: int):
    return await _ml_workspace_drive_unlink(request, workspace_id)


@router.post("/workspaces/{workspace_id}/drive-sync")
async def workspace_drive_sync(request: Request, workspace_id: int):
    return await _ml_workspace_drive_sync(request, workspace_id)


@router.get("/workspaces/{workspace_id}/drive-status")
async def workspace_drive_status(request: Request, workspace_id: int):
    return await _ml_workspace_drive_status(request, workspace_id)


# ---------------------------------------------------------------------------
# Canvas integration
# ---------------------------------------------------------------------------

@router.get("/canvas/status")
async def canvas_status(request: Request):
    return await _ml_canvas_status(request)


@router.post("/canvas/connect")
async def canvas_connect(request: Request):
    return await _ml_canvas_connect(request)


@router.get("/canvas/courses")
async def canvas_courses(request: Request):
    return await _ml_canvas_courses(request)


@router.post("/workspaces/{workspace_id}/canvas-link")
async def workspace_canvas_link(request: Request, workspace_id: int):
    return await _ml_workspace_canvas_link(request, workspace_id)


@router.delete("/workspaces/{workspace_id}/canvas-link")
async def workspace_canvas_unlink(request: Request, workspace_id: int):
    return await _ml_workspace_canvas_unlink(request, workspace_id)


@router.get("/workspaces/{workspace_id}/canvas-status")
async def workspace_canvas_status(request: Request, workspace_id: int):
    return await _ml_workspace_canvas_status(request, workspace_id)


@router.post("/workspaces/{workspace_id}/canvas-sync")
async def workspace_canvas_sync(request: Request, workspace_id: int):
    return await _ml_workspace_canvas_sync(request, workspace_id)


# ---------------------------------------------------------------------------
# Library / linked content
# ---------------------------------------------------------------------------

@router.get("/library")
async def library_list(request: Request):
    return await _ml_library_list(request)


@router.post("/library/share")
async def library_share(request: Request):
    return await _ml_library_share(request)


@router.get("/library/shares/{content_type}/{content_id}")
async def library_share_get(request: Request, content_type: str, content_id: str):
    return await _ml_library_share_get(request, content_type, content_id)


@router.post("/library/documents")
async def library_upload_document(request: Request):
    return await _ml_library_upload_document(request)


@router.post("/workspaces/{workspace_id}/link")
async def workspace_link_content(request: Request, workspace_id: int):
    return await _ml_workspace_link_content(request, workspace_id)


@router.delete("/workspaces/{workspace_id}/unlink/{content_type}/{content_id}")
async def workspace_unlink_content(request: Request, workspace_id: int, content_type: str, content_id: str):
    return await _ml_workspace_unlink_content(request, workspace_id, content_type, content_id)


@router.get("/workspaces/{workspace_id}/linked-content")
async def workspace_linked_content(request: Request, workspace_id: int):
    return await _ml_workspace_linked_content(request, workspace_id)


# ---------------------------------------------------------------------------
# Team chat JS
# ---------------------------------------------------------------------------

@router.get("/team-chat.js")
async def team_chat_js(request: Request):
    return await _ml_team_chat_js(request)


# ---------------------------------------------------------------------------
# Analyze async (background upload + job status)
# ---------------------------------------------------------------------------

@router.post("/analyze-async")
async def analyze_async(
    request: Request,
    file: UploadFile = File(...),
    workspace_id: int | None = Query(default=None),
):
    """Accept a file upload, store it in MinIO, and queue it for background processing."""
    if not file.filename or not file.filename.lower().endswith((".mp4", ".m4a", ".mp3")):
        raise HTTPException(status_code=400, detail="Only MP4, M4A, and MP3 files are supported.")
    uid = getattr(request.state, "user_id", None)
    contents = await file.read()
    original_filename = file.filename
    logger.info("Async upload queued: %s (%.1f MB), workspace=%s", original_filename, len(contents)/1024/1024, workspace_id)

    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO upload_jobs (workspace_id, user_id, filename, minio_key, status)
            VALUES ($1, $2, $3, '', 'uploading')
            RETURNING id
            """,
            workspace_id, uid, original_filename,
        )
        job_id = row["id"]
        ext = os.path.splitext(original_filename)[1] or ".mp4"
        minio_key = f"upload-jobs/{job_id}/original{ext}"
        await conn.execute(
            "UPDATE upload_jobs SET minio_key=$1 WHERE id=$2",
            minio_key, job_id,
        )

    async with get_minio_client(request.app.state.minio_client) as s3:
        await s3.put_object(Bucket=MINIO_BUCKET, Key=minio_key, Body=contents)

    async with request.app.state.db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE upload_jobs SET status='queued', updated_at=NOW() WHERE id=$1",
            job_id,
        )

    logger.info("Job %d: file stored at %s, queued for processing", job_id, minio_key)
    return {"job_id": job_id, "filename": original_filename, "status": "queued"}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: int, request: Request):
    """Poll the status of a background upload job."""
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, workspace_id, filename, status, error, meeting_id, created_at, updated_at FROM upload_jobs WHERE id=$1",
            job_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "id": row["id"],
        "workspace_id": row["workspace_id"],
        "filename": row["filename"],
        "status": row["status"],
        "error": row["error"],
        "meeting_id": row["meeting_id"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


# ---------------------------------------------------------------------------
# Analyze (synchronous streaming file upload + transcription + LLM)
# ---------------------------------------------------------------------------

@router.post("/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    workspace_id: int | None = Query(default=None),
):
    if not file.filename or not file.filename.lower().endswith((".mp4", ".m4a", ".mp3")):
        raise HTTPException(status_code=400, detail="Only MP4, M4A, and MP3 files are supported.")

    contents = await file.read()
    original_filename = file.filename
    file_size_mb = len(contents) / (1024 * 1024)
    logger.info("Upload received: %s (%.1f MB), workspace=%s", original_filename, file_size_mb, workspace_id)
    uid = getattr(request.state, "user_id", None)

    pool = request.app.state.db_pool

    async def stream():
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            audio_path = os.path.join(tmpdir, "audio.wav")

            yield json.dumps({"status": "Saving uploaded file..."}) + "\n"

            with open(input_path, "wb") as f:
                f.write(contents)

            yield json.dumps({"status": "Extracting audio with ffmpeg..."}) + "\n"
            try:
                await _meetings_svc.extract_audio(input_path, audio_path)
            except RuntimeError as e:
                logger.error("ffmpeg failed for %s: %s", original_filename, e)
                yield json.dumps({"error": str(e)}) + "\n"
                return

            yield json.dumps({"status": "Transcribing audio..."}) + "\n"
            try:
                transcript = await _meetings_svc.transcribe(audio_path)
            except Exception as e:
                logger.error("Transcription failed for %s: %s", original_filename, e)
                yield json.dumps({"error": f"Transcription failed: {e}"}) + "\n"
                return

            if not transcript.strip():
                yield json.dumps({"error": "Transcription returned empty result."}) + "\n"
                return

            yield json.dumps({"status": "Analyzing with AI..."}) + "\n"
            try:
                analysis, _ = await _meetings_svc.analyze_with_llm(pool, transcript, workspace_id)
            except Exception as e:
                logger.error("LLM analysis failed for %s: %s", original_filename, e)
                yield json.dumps({"error": f"Analysis failed: {e}"}) + "\n"
                return

            yield json.dumps({"status": "Saving results..."}) + "\n"
            try:
                meeting_id = await _meetings_svc.save_meeting(pool, original_filename, transcript, analysis, workspace_id, uid)
            except Exception as e:
                logger.error("Save failed for %s: %s", original_filename, e)
                yield json.dumps({"error": f"Failed to save meeting: {e}"}) + "\n"
                return

            result = {
                "id": meeting_id,
                "filename": original_filename,
                "transcript": transcript,
                "summary": analysis.get("summary", ""),
                "action_items": analysis.get("action_items", []),
                "todos": analysis.get("todos", []),
                "email_body": analysis.get("email_body", ""),
            }
            yield _json_line({"result": result})

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

@router.post("/admin/backfill-office-ocr")
async def admin_backfill_office_ocr(request: Request):
    return await _ml_admin_backfill_office_ocr(request)
