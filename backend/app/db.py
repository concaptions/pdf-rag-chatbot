"""Supabase access over PostgREST with the server-side service_role key.

One shared HTTP/2-capable client with keep-alive keeps per-query latency low.
"""

import httpx

from app.config import get_settings


class DatabaseError(RuntimeError):
    pass


_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        s = get_settings()
        _client = httpx.AsyncClient(
            base_url=f"{s.supabase_url.rstrip('/')}/rest/v1",
            headers={
                "apikey": s.supabase_service_role_key,
                "Authorization": f"Bearer {s.supabase_service_role_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(20.0, connect=5.0),
        )
    return _client


async def close() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def _request(method: str, path: str, **kwargs):
    try:
        r = await client().request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        raise DatabaseError(f"Database unreachable: {exc.__class__.__name__}") from exc
    if r.status_code >= 400:
        raise DatabaseError(f"Database error {r.status_code}: {r.text[:300]}")
    return r.json() if r.content else None


async def select(table: str, params: dict) -> list[dict]:
    return await _request("GET", f"/{table}", params=params)


async def insert(table: str, rows, returning: bool = True):
    prefer = "return=representation" if returning else "return=minimal"
    return await _request("POST", f"/{table}", json=rows, headers={"Prefer": prefer})


async def update(table: str, params: dict, values: dict) -> None:
    await _request("PATCH", f"/{table}", params=params, json=values, headers={"Prefer": "return=minimal"})


async def delete(table: str, params: dict) -> None:
    await _request("DELETE", f"/{table}", params=params)


async def rpc(name: str, payload: dict) -> list[dict]:
    return await _request("POST", f"/rpc/{name}", json=payload)
