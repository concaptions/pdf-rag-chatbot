"""Authentication with Supabase Auth (email + password).

The browser signs in with Supabase and sends its access token as `Authorization: Bearer`.
Every row in documents, chunks, conversations and messages is keyed by `workspace_id`,
which is the signed-in user's id, so each account only ever sees its own data, from any device.

Tokens signed with the project's asymmetric key (ES256/RS256) are verified locally against
the cached JWKS, so there is no network call per request. Legacy HS256 tokens are checked
with Supabase's /auth/v1/user endpoint and the result is cached briefly.
"""

import hashlib
import time

import httpx
import jwt
from fastapi import Header, HTTPException

from app.config import get_settings

ASYMMETRIC = ("ES256", "RS256", "EdDSA")
JWKS_TTL_S = 3600
REMOTE_CACHE_S = 300

_http: httpx.AsyncClient | None = None
_jwks: dict[str, jwt.PyJWK] = {}
_jwks_fetched_at = 0.0
_remote_cache: dict[str, tuple[float, str]] = {}  # sha256(token) -> (valid_until, user_id)


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))
    return _http


def _base() -> str:
    return get_settings().supabase_url.rstrip("/")


def _admin_headers() -> dict:
    key = get_settings().supabase_service_role_key
    return {"apikey": key, "Authorization": f"Bearer {key}"}


def unauthorized(message: str = "Please sign in to continue.") -> HTTPException:
    return HTTPException(401, message, headers={"WWW-Authenticate": "Bearer"})


async def _signing_key(kid: str | None) -> jwt.PyJWK | None:
    global _jwks_fetched_at
    stale = time.time() - _jwks_fetched_at > JWKS_TTL_S
    # Refresh when stale, or when the key id is unknown (key rotation), at most once a minute.
    if stale or (kid not in _jwks and time.time() - _jwks_fetched_at > 60):
        try:
            r = await _client().get(f"{_base()}/auth/v1/.well-known/jwks.json", headers=_admin_headers())
            r.raise_for_status()
            keys = jwt.PyJWKSet.from_dict(r.json()).keys
            _jwks.clear()
            _jwks.update({k.key_id: k for k in keys})
            _jwks_fetched_at = time.time()
        except (httpx.HTTPError, jwt.PyJWTError, ValueError):
            if not _jwks:
                raise HTTPException(503, "Sign-in could not be verified right now. Please try again shortly.")
    return _jwks.get(kid) if kid else (next(iter(_jwks.values()), None))


async def _verify_remote(token: str) -> str:
    key = hashlib.sha256(token.encode()).hexdigest()
    hit = _remote_cache.get(key)
    if hit and hit[0] > time.time():
        return hit[1]
    try:
        r = await _client().get(f"{_base()}/auth/v1/user",
                                headers={"apikey": get_settings().supabase_service_role_key, "Authorization": f"Bearer {token}"})
    except httpx.HTTPError:
        raise HTTPException(503, "Sign-in could not be verified right now. Please try again shortly.")
    if r.status_code in (401, 403):
        raise unauthorized("Your session has expired. Please sign in again.")
    if r.status_code != 200:
        raise HTTPException(503, "Sign-in could not be verified right now. Please try again shortly.")
    if len(_remote_cache) > 5000:
        _remote_cache.clear()
    _remote_cache[key] = (time.time() + REMOTE_CACHE_S, r.json()["id"])
    return _remote_cache[key][1]


async def verify(token: str) -> str:
    """Return the user id for a valid access token; raise 401 otherwise."""
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError:
        raise unauthorized()
    alg = header.get("alg")
    if alg not in ASYMMETRIC:
        return await _verify_remote(token)
    key = await _signing_key(header.get("kid"))
    if key is None:
        raise unauthorized()
    try:
        claims = jwt.decode(token, key.key, algorithms=[alg], audience="authenticated",
                            issuer=f"{_base()}/auth/v1", options={"require": ["exp", "sub"]})
    except jwt.ExpiredSignatureError:
        raise unauthorized("Your session has expired. Please sign in again.")
    except jwt.PyJWTError:
        raise unauthorized()
    return claims["sub"]


async def current_user(authorization: str | None = Header(None)) -> str:
    """FastAPI dependency: the signed-in user's id, used as the workspace id for all data."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise unauthorized()
    return await verify(authorization[7:].strip())


async def is_registered_user(user_id: str) -> bool:
    """True if `user_id` belongs to an existing account (used to refuse claiming someone's data)."""
    try:
        r = await _client().get(f"{_base()}/auth/v1/admin/users/{user_id}", headers=_admin_headers())
    except httpx.HTTPError:
        raise HTTPException(503, "Your account could not be checked right now. Please try again shortly.")
    if r.status_code == 404:
        return False
    if r.status_code == 200:
        return True
    raise HTTPException(503, "Your account could not be checked right now. Please try again shortly.")
