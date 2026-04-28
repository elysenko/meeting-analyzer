"""
OAuth client — single instance of authlib OAuth + Keycloak registration.

Imported by routers/auth.py and by main_live.py (lifespan pre-warm).
No imports from the rest of the app besides config.
"""

from authlib.integrations.starlette_client import OAuth

from config import (
    KEYCLOAK_URL, KEYCLOAK_REALM, KEYCLOAK_CLIENT_ID, KEYCLOAK_CLIENT_SECRET,
)

oauth = OAuth()
oauth.register(
    name="keycloak",
    client_id=KEYCLOAK_CLIENT_ID,
    client_secret=KEYCLOAK_CLIENT_SECRET,
    server_metadata_url=f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/.well-known/openid-configuration",
    # Override server-side endpoints to use internal cluster URL.
    # The OIDC discovery doc returns HTTPS frontend URLs (for the browser),
    # but the pod cannot reach those — it uses the internal service instead.
    access_token_url=f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token",
    jwks_uri=f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/certs",
    client_kwargs={
        "scope": "openid email profile",
        "code_challenge_method": "S256",
    },
)
