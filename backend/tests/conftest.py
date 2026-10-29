import itertools
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import auth, db, embeddings, llm, main

WS = "11111111-1111-1111-1111-111111111111"
OTHER_WS = "99999999-9999-9999-9999-999999999999"


class FakeDB:
    """In-memory stand-in for the PostgREST helpers in app.db (eq. filters, order, limit)."""

    def __init__(self):
        self.tables: dict[str, list[dict]] = {"documents": [], "chunks": [], "messages": [], "conversations": []}
        self.ids = itertools.count(1)
        self.rpc_rows: list[dict] | object = []  # a list, or a function of the query text
        self.rpc_calls: list[dict] = []

    @staticmethod
    def _match(row: dict, params: dict) -> bool:
        for key, cond in params.items():
            if key in ("select", "order", "limit"):
                continue
            op, _, value = cond.partition(".")
            if op == "eq" and str(row.get(key)) != value:
                return False
            if op == "gt" and not (row.get(key) is not None and float(row[key]) > float(value)):
                return False
            if op not in ("eq", "gt"):
                raise NotImplementedError(cond)
        return True

    async def select(self, table: str, params: dict) -> list[dict]:
        rows = [dict(r) for r in self.tables[table] if self._match(r, params)]
        for part in reversed(params.get("order", "").split(",")):
            if part:
                col, _, direction = part.partition(".")
                rows.sort(key=lambda r: (r.get(col) is None, r.get(col)), reverse=direction == "desc")
        if "limit" in params:
            rows = rows[: int(params["limit"])]
        if "select" in params:
            cols = params["select"].split(",")
            rows = [{c: r.get(c) for c in cols} for r in rows]
        return rows

    async def insert(self, table: str, rows, returning: bool = True):
        rows = rows if isinstance(rows, list) else [rows]
        if len({frozenset(r) for r in rows}) > 1:  # PostgREST: "All object keys must match" (PGRST102)
            raise db.DatabaseError("Database error 400: All object keys must match")
        now = datetime.now(timezone.utc).isoformat()
        out = []
        for r in rows:
            r = {"created_at": now, "updated_at": now, **r}
            if table in ("documents", "conversations"):
                r.setdefault("id", str(uuid.uuid4()))
                r.setdefault("status", "processing") if table == "documents" else None
                r.setdefault("title", "New chat") if table == "conversations" else None
            else:
                r.setdefault("id", next(self.ids))
            self.tables[table].append(r)
            out.append(dict(r))
        return out if returning else None

    async def update(self, table: str, params: dict, values: dict) -> None:
        for r in self.tables[table]:
            if self._match(r, params):
                r.update(values)

    async def delete(self, table: str, params: dict) -> None:
        self.tables[table] = [r for r in self.tables[table] if not self._match(r, params)]
        if table == "conversations":  # on delete cascade
            alive = {c["id"] for c in self.tables["conversations"]}
            self.tables["messages"] = [m for m in self.tables["messages"] if m.get("conversation_id") in alive]

    async def rpc(self, name: str, payload: dict) -> list[dict]:
        self.rpc_calls.append(payload)
        rows = self.rpc_rows(payload["query_text"]) if callable(self.rpc_rows) else self.rpc_rows
        return [dict(r) for r in rows]


@pytest.fixture
def fake_db(monkeypatch):
    fake = FakeDB()
    for name in ("select", "insert", "update", "delete", "rpc"):
        monkeypatch.setattr(db, name, getattr(fake, name))
    return fake


@pytest.fixture
def api(monkeypatch, fake_db):
    monkeypatch.setattr(embeddings, "warmup", lambda: None)

    async def fake_embed(_q):
        return [0.0] * 384

    monkeypatch.setattr(embeddings, "embed_query", fake_embed)

    async def fake_verify(token):  # tests sign in with "Bearer <user id>"; anything else is rejected
        try:
            return str(uuid.UUID(token))
        except ValueError:
            raise auth.unauthorized()

    monkeypatch.setattr(auth, "verify", fake_verify)
    saved = dict(llm.PROVIDERS)
    llm._last_status.clear()
    with TestClient(main.app) as client:
        yield client
    llm.PROVIDERS.clear()
    llm.PROVIDERS.update(saved)
    llm._last_status.clear()


def provider(*chunks, fail: Exception | None = None, calls: list | None = None):
    """A fake streaming provider: yields chunks, then optionally raises."""
    async def gen(model, messages, summary=""):
        if calls is not None:
            calls.append({"model": model, "messages": messages, "summary": summary})
        for c in chunks:
            yield c
        if fail:
            raise fail
    return gen


def ready_doc(fake: FakeDB, ws: str = WS, filename: str = "guide.pdf", **extra) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    doc = {"id": str(uuid.uuid4()), "workspace_id": ws, "filename": filename, "status": "ready", "page_count": 12,
           "chunk_count": 40, "progress": 100, "preview": "Hypertension management guideline.", "created_at": now,
           "updated_at": now, "error": None, **extra}
    fake.tables["documents"].append(doc)
    return doc
