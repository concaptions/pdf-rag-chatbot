import json

import pytest
from fastapi.testclient import TestClient

from app import db, embeddings, llm, main

WS = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(embeddings, "warmup", lambda: None)
    with TestClient(main.app) as c:
        yield c


def test_health(api):
    assert api.get("/health").json() == {"status": "ok"}


def test_requires_workspace_header(api):
    assert api.get("/documents").status_code == 400


def test_rejects_unsupported_and_invalid_files(api):
    r = api.post("/documents", headers={"X-Workspace-Id": WS},
                 files=[("files", ("notes.txt", b"hi", "text/plain")), ("files", ("fake.pdf", b"not a pdf", "application/pdf"))])
    assert [x["status"] for x in r.json()] == ["rejected", "rejected"]


def test_database_outage_returns_503(api, monkeypatch):
    async def boom(*a, **k):
        raise db.DatabaseError("down")
    monkeypatch.setattr(db, "select", boom)
    assert api.get("/documents", headers={"X-Workspace-Id": WS}).status_code == 503


def test_chat_without_evidence_does_not_call_llm(api, monkeypatch):
    async def fake_select(*a, **k):
        return []
    async def fake_rpc(*a, **k):
        return []
    async def fake_embed(q):
        return [0.0] * 384
    async def fake_insert(*a, **k):
        return None
    def no_llm(*a, **k):
        raise AssertionError("LLM must not be called without evidence")
    monkeypatch.setattr(db, "select", fake_select)
    monkeypatch.setattr(db, "rpc", fake_rpc)
    monkeypatch.setattr(db, "insert", fake_insert)
    monkeypatch.setattr(embeddings, "embed_query", fake_embed)
    monkeypatch.setattr(llm, "stream_answer", no_llm)
    r = api.post("/chat", headers={"X-Workspace-Id": WS}, json={"question": "What is the dose of drug X?"})
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == llm.NO_EVIDENCE and events[-1]["citations"] == []
