"""
Meeting service helpers.

Audio extraction, transcription, meeting analysis, save/upload pipeline,
and upload-job worker.

All DB-touching functions accept a `pool` parameter (asyncpg pool).
MinIO operations accept `minio_session` (request.app.state.minio_client).
"""

import asyncio
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import HTTPException

from config import (
    MINIO_BUCKET,
    RESEARCH_MATCH_STOPWORDS,
    WHISPER_CHUNK_SECONDS,
    WHISPER_CHUNK_SIZE_MB,
    WHISPER_RETRY_COUNT,
    WHISPER_URL,
)

logger = logging.getLogger("meeting-analyzer")


# ---------------------------------------------------------------------------
# Scoring / matching helpers
# ---------------------------------------------------------------------------


def tokenize_match_terms(*texts: str) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        for token in re.findall(r"[a-z0-9]{4,}", (text or "").lower()):
            if token not in RESEARCH_MATCH_STOPWORDS:
                tokens.add(token)
    return tokens


def score_term_overlap(left: str, right: str) -> int:
    return len(tokenize_match_terms(left) & tokenize_match_terms(right))


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


async def extract_audio(input_path: str, output_path: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-i", input_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path, "-y",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode()[-500:]}")


async def _probe_audio_duration_seconds(audio_path: str) -> float:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {stderr.decode()[-500:]}")
    return float((stdout.decode() or "0").strip() or "0")


async def _split_audio_for_transcription(audio_path: str, chunk_dir: str, chunk_seconds: int) -> list[str]:
    duration = await _probe_audio_duration_seconds(audio_path)
    if duration <= chunk_seconds:
        return [audio_path]
    paths: list[str] = []
    start = 0.0
    index = 0
    while start < duration:
        part_seconds = min(float(chunk_seconds), duration - start)
        output_path = os.path.join(chunk_dir, f"chunk-{index:03d}.wav")
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-ss", str(start),
            "-t", str(part_seconds),
            "-i", audio_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_path,
            "-y",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg chunking failed: {stderr.decode()[-500:]}")
        paths.append(output_path)
        start += float(chunk_seconds)
        index += 1
    return paths


async def _transcribe_single_file(audio_path: str) -> str:
    import time as _time  # noqa: PLC0415

    chunk_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    logger.info("Sending audio to Whisper: %s (%.1f MB)", audio_path, chunk_size_mb)
    t0 = _time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=3600.0) as client:
            with open(audio_path, "rb") as f:
                resp = await client.post(
                    f"{WHISPER_URL}/asr",
                    params={"language": "en", "output": "json", "encode": "true"},
                    files={"audio_file": ("audio.wav", f, "audio/wav")},
                )
            elapsed = _time.monotonic() - t0
            logger.info("Whisper response: status=%d, %.1fs, %.1f MB", resp.status_code, elapsed, chunk_size_mb)
            resp.raise_for_status()
            data = resp.json()
            return data.get("text", "")
    except Exception as e:
        elapsed = _time.monotonic() - t0
        logger.error("Whisper request failed after %.1fs for %s (%.1f MB): %s: %s",
            elapsed, audio_path, chunk_size_mb, type(e).__name__, e)
        raise


async def transcribe(audio_path: str) -> str:
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    try:
        duration = await _probe_audio_duration_seconds(audio_path)
    except Exception as exc:
        logger.warning("Unable to probe audio duration for %s: %s", audio_path, exc)
        duration = 0.0
    chunk_needed = size_mb > WHISPER_CHUNK_SIZE_MB or duration > WHISPER_CHUNK_SECONDS
    if chunk_needed:
        with tempfile.TemporaryDirectory(prefix="whisper-chunks-", dir=os.path.dirname(audio_path)) as chunk_dir:
            chunk_paths = await _split_audio_for_transcription(audio_path, chunk_dir, WHISPER_CHUNK_SECONDS)
            logger.info(
                "Chunking audio for Whisper: %s split into %d chunks (duration=%.1fs, size=%.1f MB)",
                audio_path,
                len(chunk_paths),
                duration,
                size_mb,
            )
            transcript_parts = []
            for index, chunk_path in enumerate(chunk_paths, start=1):
                for attempt in range(WHISPER_RETRY_COUNT + 1):
                    try:
                        logger.info("Transcribing chunk %d/%d: %s", index, len(chunk_paths), chunk_path)
                        transcript_parts.append((await _transcribe_single_file(chunk_path)).strip())
                        break
                    except (httpx.ConnectError, httpx.TimeoutException) as exc:
                        if attempt >= WHISPER_RETRY_COUNT:
                            raise
                        wait = 3 + attempt * 3
                        logger.warning(
                            "Whisper connection failed on chunk %d/%d (attempt %d/%d, retry in %ds): %s: %s",
                            index, len(chunk_paths), attempt + 1, WHISPER_RETRY_COUNT + 1,
                            wait, type(exc).__name__, exc,
                        )
                        await asyncio.sleep(wait)
                    except (httpx.RemoteProtocolError, httpx.HTTPStatusError) as exc:
                        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                            raise
                        if attempt >= WHISPER_RETRY_COUNT:
                            raise
                        wait = 1 + attempt * 2
                        logger.warning(
                            "Whisper error on chunk %d/%d (attempt %d/%d, retry in %ds): %s: %s",
                            index, len(chunk_paths), attempt + 1, WHISPER_RETRY_COUNT + 1,
                            wait, type(exc).__name__, exc,
                        )
                        await asyncio.sleep(wait)
            return "\n\n".join(part for part in transcript_parts if part).strip()

    for attempt in range(WHISPER_RETRY_COUNT + 1):
        try:
            return (await _transcribe_single_file(audio_path)).strip()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            if attempt >= WHISPER_RETRY_COUNT:
                raise
            wait = 3 + attempt * 3
            logger.warning(
                "Whisper connection failed for %s (attempt %d/%d, retry in %ds): %s: %s",
                audio_path, attempt + 1, WHISPER_RETRY_COUNT + 1, wait, type(exc).__name__, exc,
            )
            await asyncio.sleep(wait)
        except (httpx.RemoteProtocolError, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                raise
            if attempt >= WHISPER_RETRY_COUNT:
                raise
            wait = 1 + attempt * 2
            logger.warning(
                "Whisper error for %s (attempt %d/%d, retry in %ds): %s: %s",
                audio_path, attempt + 1, WHISPER_RETRY_COUNT + 1, wait, type(exc).__name__, exc,
            )
            await asyncio.sleep(wait)
    return ""


# ---------------------------------------------------------------------------
# Meeting analysis
# ---------------------------------------------------------------------------


ANALYSIS_PROMPT = """\
You are a meeting analyst. Given the following transcript, produce:

1. A short title for this meeting (5-8 words max)
2. A concise summary (2-4 sentences)
3. A list of action items (strings)
4. A list of to-dos (objects with keys: task, assignee, due_date, due_text)
5. A draft follow-up email body

Return ONLY valid JSON with keys: title, summary, action_items, todos, email_body.

Today's date: {analysis_date}

Transcript:
{transcript}
"""


async def analyze_with_llm(pool, transcript: str, workspace_id: int | None = None) -> tuple[dict, dict[str, Any]]:
    from services.llm_prefs import get_workspace_llm_preferences, _resolve_task_llm  # noqa: PLC0415
    from services.todos_svc import normalize_analysis_payload  # noqa: PLC0415
    from llm import call_llm_runner_json  # noqa: PLC0415

    analysis_dt = datetime.now(timezone.utc)
    prompt = ANALYSIS_PROMPT.format(
        transcript=transcript,
        analysis_date=analysis_dt.date().isoformat(),
    )
    preferences = await get_workspace_llm_preferences(pool, workspace_id)
    provider, model = _resolve_task_llm(preferences, "analysis")
    try:
        payload, llm_meta = await call_llm_runner_json(
            [{"role": "user", "content": prompt}],
            provider=provider,
            model=model,
            use_case="chat",
            max_tokens=4096,
        )
        return normalize_analysis_payload(payload, analysis_dt), llm_meta
    except Exception as exc:
        logger.warning("analyze_with_llm: JSON parse/repair failed: %s", exc)
        return normalize_analysis_payload({
            "title": "Untitled Meeting",
            "summary": "Analysis failed.",
            "action_items": [],
            "todos": [],
            "email_body": "",
        }, analysis_dt), {}


def title_from_filename(filename: str) -> str:
    """Derive a meeting title from the filename by stripping extension and cleaning up."""
    if not filename:
        return "Untitled Meeting"
    name = os.path.splitext(filename)[0]
    name = name.replace("_", " ").replace("-", " ").strip()
    return name if name else "Untitled Meeting"


async def save_meeting(
    pool,
    filename: str,
    transcript: str,
    analysis: dict,
    workspace_id: int | None = None,
    user_id: str | None = None,
    recorded_at: datetime | None = None,
) -> int:
    from services.documents_svc import replace_meeting_chunks  # noqa: PLC0415

    async with pool.acquire() as conn:
        if recorded_at is not None:
            row = await conn.fetchrow(
                """INSERT INTO meetings (title, filename, transcript, summary, action_items, todos, email_body, workspace_id, user_id, date)
                   VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, $9, $10)
                   RETURNING id""",
                analysis.get("title") or title_from_filename(filename),
                filename,
                transcript,
                analysis.get("summary", ""),
                json.dumps(analysis.get("action_items", [])),
                json.dumps(analysis.get("todos", [])),
                analysis.get("email_body", ""),
                workspace_id,
                user_id,
                recorded_at,
            )
        else:
            row = await conn.fetchrow(
                """INSERT INTO meetings (title, filename, transcript, summary, action_items, todos, email_body, workspace_id, user_id)
                   VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, $9)
                   RETURNING id""",
                analysis.get("title") or title_from_filename(filename),
                filename,
                transcript,
                analysis.get("summary", ""),
                json.dumps(analysis.get("action_items", [])),
                json.dumps(analysis.get("todos", [])),
                analysis.get("email_body", ""),
                workspace_id,
                user_id,
            )
        meeting_id = row["id"]

    if workspace_id and transcript and transcript.strip():
        async def _chunk_meeting_safe():
            try:
                await replace_meeting_chunks(pool, meeting_id, workspace_id, transcript)
            except Exception as exc:
                logger.warning("Auto-chunk failed for meeting %s: %s", meeting_id, exc)
        asyncio.create_task(_chunk_meeting_safe())

    return meeting_id


# ---------------------------------------------------------------------------
# Upload job worker
# ---------------------------------------------------------------------------


async def upload_job_worker(pool, minio_session) -> None:
    """Process upload_jobs rows in the background, one at a time."""
    logger.info("Upload job worker started")
    while True:
        try:
            async with pool.acquire() as conn:
                job = await conn.fetchrow(
                    """
                    UPDATE upload_jobs
                    SET status = 'extracting', updated_at = NOW()
                    WHERE id = (
                        SELECT id FROM upload_jobs
                        WHERE status = 'queued'
                        ORDER BY created_at
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING id, workspace_id, user_id, filename, minio_key
                    """
                )
            if not job:
                await asyncio.sleep(5)
                continue

            job_id = job["id"]
            workspace_id = job["workspace_id"]
            user_id = job["user_id"]
            filename = job["filename"]
            minio_key = job["minio_key"]

            logger.info("Job %d: processing %s", job_id, filename)

            async def _set_status(status: str, error: str | None = None, meeting_id: int | None = None):
                async with pool.acquire() as _conn:
                    await _conn.execute(
                        "UPDATE upload_jobs SET status=$1, error=$2, meeting_id=$3, updated_at=NOW() WHERE id=$4",
                        status, error, meeting_id, job_id,
                    )

            with tempfile.TemporaryDirectory() as tmpdir:
                ext = os.path.splitext(filename)[1] or ".mp4"
                input_path = os.path.join(tmpdir, f"input{ext}")
                audio_path = os.path.join(tmpdir, "audio.wav")

                # Download from MinIO
                try:
                    from services.storage import get_minio_client  # noqa: PLC0415

                    async with get_minio_client(minio_session) as s3:
                        resp = await s3.get_object(Bucket=MINIO_BUCKET, Key=minio_key)
                        body = await resp["Body"].read()
                    with open(input_path, "wb") as f:
                        f.write(body)
                except Exception as e:
                    logger.error("Job %d: MinIO download failed: %s", job_id, e)
                    await _set_status("failed", f"Download failed: {e}")
                    continue

                # Extract audio
                try:
                    await extract_audio(input_path, audio_path)
                except Exception as e:
                    logger.error("Job %d: audio extraction failed: %s", job_id, e)
                    await _set_status("failed", f"Audio extraction failed: {e}")
                    continue

                # Detect recording date
                recorded_at: datetime | None = None
                try:
                    probe_proc = await asyncio.create_subprocess_exec(
                        "ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", "-show_streams", input_path,
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    )
                    probe_stdout, _ = await probe_proc.communicate()
                    probe = json.loads(probe_stdout)
                    _tags = (probe.get("format") or {}).get("tags") or {}
                    _streams = probe.get("streams") or []
                    _stream_tags = (_streams[0].get("tags") or {}) if _streams else {}
                    for _tag_key in ("com.apple.quicktime.creationdate", "creation_time", "date"):
                        _val = _tags.get(_tag_key) or _stream_tags.get(_tag_key) or ""
                        if _val:
                            try:
                                recorded_at = datetime.fromisoformat(_val.replace("Z", "+00:00"))
                                break
                            except ValueError:
                                pass
                except Exception:
                    pass

                # Filename date fallback
                if not recorded_at:
                    import re as _re  # noqa: PLC0415

                    _months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                               "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
                    _stem = os.path.splitext(filename)[0]
                    for _pat in [
                        r'(\d{4})[._-](\d{2})[._-](\d{2})',
                        r'(\d{4})(\d{2})(\d{2})',
                    ]:
                        _m = _re.search(_pat, _stem)
                        if _m:
                            try:
                                recorded_at = datetime(int(_m.group(1)), int(_m.group(2)), int(_m.group(3)), tzinfo=timezone.utc)
                                break
                            except ValueError:
                                pass
                    if not recorded_at:
                        for _pat2 in [
                            r'([A-Za-z]{3,9})\s+(\d{1,2})',
                            r'(\d{1,2})\s+([A-Za-z]{3,9})',
                        ]:
                            _m2 = _re.search(_pat2, _stem)
                            if _m2:
                                try:
                                    _g = _m2.groups()
                                    mon_str = (_g[0] if _g[0].isalpha() else _g[1]).lower()[:3]
                                    day = int(_g[1] if _g[0].isalpha() else _g[0])
                                    mon = _months.get(mon_str)
                                    if mon and 1 <= day <= 31:
                                        recorded_at = datetime(datetime.now(timezone.utc).year, mon, day, tzinfo=timezone.utc)
                                        break
                                except (ValueError, AttributeError):
                                    pass

                # Transcribe
                await _set_status("transcribing")
                try:
                    transcript = await transcribe(audio_path)
                    logger.info("Job %d: transcription done (%d chars)", job_id, len(transcript))
                except Exception as e:
                    logger.error("Job %d: transcription failed: %s", job_id, e)
                    await _set_status("failed", f"Transcription failed: {e}")
                    continue

                if not transcript.strip():
                    await _set_status("failed", "No speech detected in the audio.")
                    continue

                # Analyze
                await _set_status("analyzing")
                try:
                    analysis, _ = await analyze_with_llm(pool, transcript, workspace_id)
                    logger.info("Job %d: analysis done", job_id)
                except Exception as e:
                    logger.error("Job %d: analysis failed: %s", job_id, e)
                    await _set_status("failed", f"Analysis failed: {e}")
                    continue

                # Save
                try:
                    meeting_id = await save_meeting(
                        pool, filename, transcript, analysis, workspace_id,
                        user_id=user_id, recorded_at=recorded_at,
                    )
                    logger.info("Job %d: saved as meeting %d", job_id, meeting_id)
                    await _set_status("done", meeting_id=meeting_id)
                except Exception as e:
                    logger.error("Job %d: save failed: %s", job_id, e)
                    await _set_status("failed", f"Save failed: {e}")

        except Exception as exc:
            logger.error("Upload job worker outer error: %s", exc)
            await asyncio.sleep(5)
