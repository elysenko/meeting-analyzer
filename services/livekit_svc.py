"""
LiveKit JWT token helpers.
"""

import time as _time
import uuid

from jose import jwt

from config import LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_WS_URL


def livekit_token(claims: dict) -> str:
    return jwt.encode(claims, LIVEKIT_API_SECRET, algorithm="HS256")


def make_subscriber_token(room: str, identity: str | None = None) -> dict:
    """Return token + url + identity for a LiveKit subscriber."""
    resolved_identity = identity or f"listener-{uuid.uuid4().hex[:8]}"
    now = int(_time.time())
    claims = {
        "video": {
            "roomJoin": True,
            "room": room,
            "canPublish": False,
            "canSubscribe": True,
        },
        "sub": resolved_identity,
        "iss": LIVEKIT_API_KEY,
        "exp": now + 3600,
        "nbf": now,
        "jti": uuid.uuid4().hex,
    }
    token = livekit_token(claims)
    return {"token": token, "url": LIVEKIT_WS_URL, "identity": resolved_identity}


def make_room_list_token() -> str:
    """Return an admin JWT for listing rooms."""
    now = int(_time.time())
    claims = {
        "video": {"roomList": True},
        "sub": "server",
        "iss": LIVEKIT_API_KEY,
        "exp": now + 60,
        "nbf": now,
        "jti": uuid.uuid4().hex,
    }
    return livekit_token(claims)
