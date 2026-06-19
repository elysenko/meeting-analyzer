"""
Calendar sync routes.

iCal URL subscriptions (Apple / Google / Outlook / other) and
Google Calendar API direct sync (incremental via nextSyncToken).

GET    /calendar-sync/subscriptions
POST   /calendar-sync/subscriptions
PATCH  /calendar-sync/subscriptions/{id}
DELETE /calendar-sync/subscriptions/{id}
POST   /calendar-sync/subscriptions/{id}/sync

GET  /calendar-sync/google/status
POST /calendar-sync/google/sync
"""

import datetime
import logging
import os

import httpx
from fastapi import APIRouter, HTTPException, Request

from models import ICalSubscriptionCreate, ICalSubscriptionUpdate

router = APIRouter(prefix="/calendar-sync", tags=["calendar-sync"])
logger = logging.getLogger("meeting-analyzer")

_GOOGLE_CLIENT_ID = os.getenv("GOOGLE_PICKER_CLIENT_ID", "")
_GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_DRIVE_CLIENT_SECRET", "")

_GOOGLE_COLOR_MAP: dict[str, str] = {
    "1": "blue", "2": "green", "3": "purple", "4": "red",
    "5": "amber", "6": "amber", "7": "blue", "8": "green",
    "9": "blue", "10": "green", "11": "red",
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


async def _require_user(request: Request) -> str:
    uid = getattr(request.state, "user_id", None)
    if not uid:
        raise HTTPException(status_code=401, detail="Authentication required")
    return uid


async def _get_or_create_sync_workspace(conn, user_id: str) -> int:
    """Return the 'Calendar Sync' workspace id for the user, creating it if absent."""
    row = await conn.fetchrow(
        "SELECT id FROM workspaces WHERE user_id = $1 AND name = 'Calendar Sync' LIMIT 1",
        user_id,
    )
    if row:
        return row["id"]
    row = await conn.fetchrow(
        "INSERT INTO workspaces (user_id, name) VALUES ($1, $2) RETURNING id",
        user_id,
        "Calendar Sync",
    )
    return row["id"]


async def _get_google_token(request: Request, user_id: str) -> str | None:
    """Retrieve and refresh the user's Google access token if needed."""
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT google_access_token, google_refresh_token FROM user_tokens WHERE user_id = $1",
            user_id,
        )
    if not row or not row["google_access_token"]:
        return None
    token: str = row["google_access_token"]
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            info = await client.get(
                "https://www.googleapis.com/oauth2/v1/tokeninfo",
                params={"access_token": token},
            )
            if info.status_code == 200:
                return token
            # Token invalid — try refresh
            refresh = row["google_refresh_token"]
            if not refresh:
                return None
            r = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": _GOOGLE_CLIENT_ID,
                    "client_secret": _GOOGLE_CLIENT_SECRET,
                    "refresh_token": refresh,
                    "grant_type": "refresh_token",
                },
            )
            if not r.is_success:
                return None
            new_token: str | None = r.json().get("access_token")
            if new_token:
                async with request.app.state.db_pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE user_tokens SET google_access_token = $1, updated_at = NOW() WHERE user_id = $2",
                        new_token,
                        user_id,
                    )
            return new_token
    except Exception as exc:
        logger.warning("Google token refresh failed for %s: %s", user_id, exc)
        return None


def _normalize_sub(row) -> dict:
    data = dict(row)
    data["created_at"] = data["created_at"].isoformat() if data.get("created_at") else None
    data["last_synced_at"] = data["last_synced_at"].isoformat() if data.get("last_synced_at") else None
    return data


async def _upsert_events(conn, workspace_id: int, source_type: str, events: list[dict]) -> int:
    """Upsert normalised event dicts into calendar_events. Returns count."""
    count = 0
    for ev in events:
        await conn.execute(
            """
            INSERT INTO calendar_events
                (workspace_id, title, all_day, is_due_only, start_time, end_time,
                 notes, color, source_type, source_id)
            VALUES ($1, $2, $3, FALSE, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (workspace_id, source_type, source_id)
            WHERE source_type IS NOT NULL AND source_id IS NOT NULL
            DO UPDATE SET
                title      = EXCLUDED.title,
                all_day    = EXCLUDED.all_day,
                start_time = EXCLUDED.start_time,
                end_time   = EXCLUDED.end_time,
                notes      = EXCLUDED.notes,
                color      = EXCLUDED.color,
                updated_at = NOW()
            """,
            workspace_id,
            ev["title"],
            ev["all_day"],
            ev["start_time"],
            ev["end_time"],
            ev["notes"],
            ev["color"],
            source_type,
            ev["source_id"],
        )
        count += 1
    return count


# ---------------------------------------------------------------------------
# iCal parsing
# ---------------------------------------------------------------------------


def _to_utc_naive(dt) -> datetime.datetime:
    """Convert a date or datetime to a UTC-naive datetime (for asyncpg TIMESTAMPTZ)."""
    if isinstance(dt, datetime.datetime):
        if dt.tzinfo is not None:
            return dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return dt
    # plain date → midnight
    return datetime.datetime.combine(dt, datetime.time.min)


async def _parse_ical(content: bytes) -> list[dict]:
    """Parse ICS bytes into normalised event dicts."""
    from icalendar import Calendar as ICalendar  # noqa: PLC0415 — lazy import

    try:
        cal = ICalendar.from_ical(content)
    except Exception as exc:
        raise ValueError(f"Invalid ICS data: {exc}") from exc

    events: list[dict] = []
    for component in cal.walk():
        if component.name != "VEVENT":
            continue
        uid = str(component.get("UID") or "").strip()
        if not uid:
            continue
        dtstart_prop = component.get("DTSTART")
        if not dtstart_prop:
            continue

        dtstart = _to_utc_naive(dtstart_prop.dt)
        all_day = isinstance(dtstart_prop.dt, datetime.date) and not isinstance(
            dtstart_prop.dt, datetime.datetime
        )

        dtend_prop = component.get("DTEND") or component.get("DUE")
        dtend = _to_utc_naive(dtend_prop.dt) if dtend_prop else dtstart

        events.append({
            "source_id": uid,
            "title": (str(component.get("SUMMARY") or "").strip() or "(No title)")[:500],
            "all_day": all_day,
            "start_time": dtstart,
            "end_time": dtend,
            "notes": (str(component.get("DESCRIPTION") or "").strip()[:4000]) or None,
            "color": None,
        })
    return events


# ---------------------------------------------------------------------------
# iCal sync worker (called from background loop and on-demand endpoint)
# ---------------------------------------------------------------------------


async def sync_ical_subscription(app, sub: dict) -> tuple[int, str | None]:
    """Fetch and upsert one iCal subscription. Returns (count_upserted, error_or_None)."""
    url: str = sub["url"]
    source_type = f"ical_{sub['provider']}"
    workspace_id: int = sub["workspace_id"]

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "MeetingAnalyzer-CalSync/1.0"})
            resp.raise_for_status()
        events = await _parse_ical(resp.content)
    except Exception as exc:
        error = str(exc)[:500]
        async with app.state.db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE calendar_ical_subscriptions SET error_last = $1 WHERE id = $2",
                error,
                sub["id"],
            )
        return 0, error

    async with app.state.db_pool.acquire() as conn:
        count = await _upsert_events(conn, workspace_id, source_type, events)
        await conn.execute(
            "UPDATE calendar_ical_subscriptions SET last_synced_at = NOW(), error_last = NULL WHERE id = $1",
            sub["id"],
        )
    return count, None


async def background_ical_sync(app) -> None:
    """Background asyncio task: sync all enabled iCal subscriptions every 30 minutes."""
    import asyncio

    await asyncio.sleep(30)  # brief startup delay so the pool is ready
    while True:
        try:
            async with app.state.db_pool.acquire() as conn:
                subs = await conn.fetch(
                    "SELECT * FROM calendar_ical_subscriptions WHERE sync_enabled = TRUE"
                )
            for sub in subs:
                try:
                    count, err = await sync_ical_subscription(app, dict(sub))
                    if err:
                        logger.warning("iCal sync error sub=%s: %s", sub["id"], err)
                    else:
                        logger.info("iCal sync sub=%s: %d events upserted", sub["id"], count)
                except Exception as exc:
                    logger.warning("iCal sync sub=%s unexpected: %s", sub["id"], exc)
        except Exception as exc:
            logger.warning("iCal background sync loop error: %s", exc)
        await asyncio.sleep(1800)  # 30 minutes


# ---------------------------------------------------------------------------
# Google Calendar sync worker
# ---------------------------------------------------------------------------


async def sync_google_calendar(app, user_id: str, token: str, workspace_id: int) -> dict:
    """Pull events from Google Calendar API v3 with incremental sync."""
    async with app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT google_cal_sync_token FROM user_tokens WHERE user_id = $1", user_id
        )
    sync_token: str | None = row["google_cal_sync_token"] if row else None

    params: dict[str, str] = {"maxResults": "2500", "singleEvents": "true"}
    if sync_token:
        params["syncToken"] = sync_token
    else:
        now = datetime.datetime.utcnow()
        params["orderBy"] = "startTime"
        params["timeMin"] = (now - datetime.timedelta(days=365)).strftime("%Y-%m-%dT00:00:00Z")
        params["timeMax"] = (now + datetime.timedelta(days=730)).strftime("%Y-%m-%dT00:00:00Z")

    all_items: list[dict] = []
    next_sync_token: str | None = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        page_token: str | None = None
        while True:
            if page_token:
                params["pageToken"] = page_token
            resp = await client.get(
                "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code == 410:
                # Sync token expired — clear and redo a full sync
                logger.info("Google Calendar sync token expired for %s, doing full sync", user_id)
                async with app.state.db_pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE user_tokens SET google_cal_sync_token = NULL WHERE user_id = $1",
                        user_id,
                    )
                return await sync_google_calendar(app, user_id, token, workspace_id)
            resp.raise_for_status()
            data = resp.json()
            all_items.extend(data.get("items", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                next_sync_token = data.get("nextSyncToken")
                break

    upserted = deleted = 0
    async with app.state.db_pool.acquire() as conn:
        for item in all_items:
            if item.get("status") == "cancelled":
                await conn.execute(
                    "DELETE FROM calendar_events "
                    "WHERE workspace_id=$1 AND source_type='google_cal' AND source_id=$2",
                    workspace_id,
                    item["id"],
                )
                deleted += 1
                continue

            start = item.get("start", {})
            end = item.get("end", {})
            all_day = "date" in start and "dateTime" not in start

            if all_day:
                dtstart = datetime.datetime.fromisoformat(start["date"])
                dtend = datetime.datetime.fromisoformat(end.get("date", start["date"]))
            else:
                dtstart = _to_utc_naive(
                    datetime.datetime.fromisoformat(start["dateTime"].replace("Z", "+00:00"))
                )
                dtend = _to_utc_naive(
                    datetime.datetime.fromisoformat(
                        end.get("dateTime", start["dateTime"]).replace("Z", "+00:00")
                    )
                )

            upserted += await _upsert_events(
                conn,
                workspace_id,
                "google_cal",
                [{
                    "source_id": item["id"],
                    "title": (item.get("summary") or "(No title)")[:500],
                    "all_day": all_day,
                    "start_time": dtstart,
                    "end_time": dtend,
                    "notes": (item.get("description") or "")[:4000] or None,
                    "color": _GOOGLE_COLOR_MAP.get(str(item.get("colorId") or "")),
                }],
            )

        if next_sync_token:
            await conn.execute(
                """INSERT INTO user_tokens (user_id, google_cal_sync_token, updated_at)
                   VALUES ($1, $2, NOW())
                   ON CONFLICT (user_id) DO UPDATE
                   SET google_cal_sync_token = $2, updated_at = NOW()""",
                user_id,
                next_sync_token,
            )

    return {"synced": upserted, "deleted": deleted}


# ---------------------------------------------------------------------------
# Routes — iCal subscriptions
# ---------------------------------------------------------------------------


@router.get("/subscriptions")
async def list_subscriptions(request: Request):
    """List all iCal subscriptions for the current user."""
    uid = await _require_user(request)
    async with request.app.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM calendar_ical_subscriptions WHERE user_id = $1 ORDER BY created_at",
            uid,
        )
    return [_normalize_sub(r) for r in rows]


@router.post("/subscriptions")
async def create_subscription(request: Request, body: ICalSubscriptionCreate):
    """Register a new iCal URL subscription."""
    uid = await _require_user(request)
    label = (body.label or "").strip()[:200]
    if not label:
        raise HTTPException(status_code=400, detail="label is required")
    url = (body.url or "").strip()
    if not url.startswith(("http://", "https://", "webcal://")):
        raise HTTPException(status_code=400, detail="url must be http, https, or webcal")
    url = url.replace("webcal://", "https://", 1)
    provider = (body.provider or "other").strip()[:20]

    async with request.app.state.db_pool.acquire() as conn:
        workspace_id = await _get_or_create_sync_workspace(conn, uid)
        row = await conn.fetchrow(
            """INSERT INTO calendar_ical_subscriptions
                   (user_id, label, provider, url, workspace_id)
               VALUES ($1, $2, $3, $4, $5)
               RETURNING *""",
            uid,
            label,
            provider,
            url,
            workspace_id,
        )
    return {"ok": True, "subscription": _normalize_sub(row)}


@router.patch("/subscriptions/{sub_id}")
async def update_subscription(request: Request, sub_id: int, body: ICalSubscriptionUpdate):
    """Update subscription label or enabled state."""
    uid = await _require_user(request)
    async with request.app.state.db_pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT * FROM calendar_ical_subscriptions WHERE id = $1 AND user_id = $2",
            sub_id,
            uid,
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Subscription not found")
        label = body.label.strip()[:200] if body.label is not None else existing["label"]
        enabled = body.sync_enabled if body.sync_enabled is not None else existing["sync_enabled"]
        updated = await conn.fetchrow(
            """UPDATE calendar_ical_subscriptions
               SET label = $1, sync_enabled = $2
               WHERE id = $3 AND user_id = $4
               RETURNING *""",
            label,
            enabled,
            sub_id,
            uid,
        )
    return {"ok": True, "subscription": _normalize_sub(updated)}


@router.delete("/subscriptions/{sub_id}")
async def delete_subscription(request: Request, sub_id: int):
    """Remove a subscription. Synced events are kept."""
    uid = await _require_user(request)
    async with request.app.state.db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM calendar_ical_subscriptions WHERE id = $1 AND user_id = $2",
            sub_id,
            uid,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Subscription not found")
    return {"ok": True}


@router.post("/subscriptions/{sub_id}/sync")
async def trigger_subscription_sync(request: Request, sub_id: int):
    """Immediately sync one iCal subscription."""
    uid = await _require_user(request)
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM calendar_ical_subscriptions WHERE id = $1 AND user_id = $2",
            sub_id,
            uid,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Subscription not found")
    count, error = await sync_ical_subscription(request.app, dict(row))
    if error:
        raise HTTPException(status_code=502, detail=f"Sync failed: {error}")
    return {"ok": True, "synced": count}


# ---------------------------------------------------------------------------
# Routes — Google Calendar direct API
# ---------------------------------------------------------------------------


@router.get("/google/status")
async def google_calendar_status(request: Request):
    """Check whether the user has a Google account with Calendar access."""
    uid = await _require_user(request)
    token = await _get_google_token(request, uid)
    return {"connected": token is not None}


@router.post("/google/sync")
async def trigger_google_sync(request: Request):
    """Sync Google Calendar into the user's Calendar Sync workspace."""
    uid = await _require_user(request)
    token = await _get_google_token(request, uid)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Google account not linked — connect via Settings → Global Models → Google Drive",
        )
    async with request.app.state.db_pool.acquire() as conn:
        workspace_id = await _get_or_create_sync_workspace(conn, uid)
    try:
        result = await sync_google_calendar(request.app, uid, token, workspace_id)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Google Calendar API error: {exc.response.status_code}",
        ) from exc
    except Exception as exc:
        logger.error("Google Calendar sync error for %s: %s", uid, exc)
        raise HTTPException(status_code=500, detail="Google Calendar sync failed") from exc
    return {"ok": True, **result}
