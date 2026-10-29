import json
from datetime import datetime, timedelta, timezone

from app import db, llm, main
from tests.conftest import OTHER_WS, WS, provider, ready_doc

H = {"X-Workspace-Id": WS}
CHUNK = {"id": 1, "document_id": "d1", "filename": "guide.pdf", "page_number": 4,
         "content": "First-line therapy is a thiazide diuretic.", "similarity": 0.82, "score": 0.03}


def chat(api, question: str, headers=H, **body) -> list[dict]:
    r = api.post("/chat", headers=headers, json={"question": question, **body})
    assert r.status_code == 200
    return [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]


def test_health(api):
    assert api.get("/health").json() == {"status": "ok"}


def test_requires_workspace_header(api):
    assert api.get("/documents").status_code == 400


def test_rejects_unsupported_empty_and_fake_pdfs(api):
    r = api.post("/documents", headers=H, files=[
        ("files", ("notes.txt", b"hi", "text/plain")),
        ("files", ("empty.pdf", b"", "application/pdf")),
        ("files", ("fake.pdf", b"not a pdf", "application/pdf")),
    ])
    assert [(x["status"], x["error"]) for x in r.json()] == [
        ("rejected", "Unsupported file type. Only PDF files are accepted."),
        ("rejected", "The file is empty."),
        ("rejected", "The file is not a valid PDF."),
    ]


def test_database_outage_returns_friendly_503(api, monkeypatch):
    async def boom(*a, **k):
        raise db.DatabaseError("connection refused on 10.0.0.1")
    monkeypatch.setattr(db, "select", boom)
    r = api.get("/documents", headers=H)
    assert r.status_code == 503 and "10.0.0.1" not in r.json()["detail"]


def test_no_documents_gives_gentle_reply_without_llm(api, fake_db):
    llm.PROVIDERS["anthropic"] = provider(fail=AssertionError("LLM must not be called"))
    events = chat(api, "hello")
    assert events[0]["type"] == "conversation"
    assert events[-1]["type"] == "done" and "don't have any documents" in events[-1]["answer"]


def add_processing(fake_db, progress: int, doc_id: str = "p1"):
    now = datetime.now(timezone.utc).isoformat()
    fake_db.tables["documents"].append({"id": doc_id, "workspace_id": WS, "filename": "book.pdf", "status": "processing",
                                        "progress": progress, "updated_at": now, "created_at": now})
    main._active_docs.add(doc_id)


def test_just_started_indexing_gives_progress_reply_without_llm(api, fake_db):
    add_processing(fake_db, progress=5)
    try:
        llm.PROVIDERS["anthropic"] = provider(fail=AssertionError("LLM must not be called"))
        answer = chat(api, "What does the book say about insulin?")[-1]["answer"]
    finally:
        main._active_docs.discard("p1")
    assert "still indexing **book.pdf**" in answer and "5%" in answer


def test_partly_indexed_document_is_already_answerable(api, fake_db):
    add_processing(fake_db, progress=45)
    fake_db.rpc_rows = [{**CHUNK, "filename": "book.pdf"}]
    calls: list = []
    try:
        llm.PROVIDERS["anthropic"] = provider("Thiazides [S1]. The book is still being indexed.", calls=calls)
        events = chat(api, "What is first-line therapy?")
    finally:
        main._active_docs.discard("p1")
    assert 'name="book.pdf" status="processing" progress="45%"' in calls[0]["messages"][-1]["content"]
    assert events[-1]["citations"][0]["filename"] == "book.pdf"


def test_stale_processing_document_is_marked_failed(api, fake_db):
    old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    fake_db.tables["documents"].append({"id": "s1", "workspace_id": WS, "filename": "stuck.pdf", "status": "processing",
                                        "progress": 30, "updated_at": old, "created_at": old})
    doc = api.get("/documents", headers=H).json()[0]
    assert doc["status"] == "failed" and "upload the file again" in doc["error"]


def test_meta_question_reaches_llm_with_library_even_without_matching_passages(api, fake_db):
    ready_doc(fake_db, filename="hypertension-guideline.pdf")
    fake_db.rpc_rows = []  # nothing retrieved for a vague question
    calls: list = []
    llm.PROVIDERS["anthropic"] = provider("You have one document: hypertension-guideline.pdf (12 pages).", calls=calls)
    events = chat(api, "What context do you have?")
    assert events[-1]["answer"].startswith("You have one document")
    prompt = calls[0]["messages"][-1]["content"]
    assert 'name="hypertension-guideline.pdf" status="ready" pages="12"' in prompt
    assert "Hypertension management guideline." in prompt and "(no passages matched this question)" in prompt


def test_low_similarity_passages_are_not_sent(api, fake_db):
    ready_doc(fake_db)
    fake_db.rpc_rows = [CHUNK, {**CHUNK, "id": 2, "similarity": 0.1, "content": "noise"}]
    calls: list = []
    llm.PROVIDERS["anthropic"] = provider("Thiazides are first line [S1].", calls=calls)
    events = chat(api, "What is first-line therapy?")
    assert "noise" not in calls[0]["messages"][-1]["content"]
    assert events[-1]["citations"][0] == {"label": "S1", "document_id": "d1", "filename": "guide.pdf",
                                          "page_number": 4, "excerpt": CHUNK["content"]}


def test_conversation_history_is_saved_and_listed(api, fake_db):
    ready_doc(fake_db)
    fake_db.rpc_rows = [CHUNK]
    llm.PROVIDERS["anthropic"] = provider("A thiazide diuretic [S1].")
    events = chat(api, "What is the first-line therapy for hypertension?")
    conv = events[0]
    assert conv["type"] == "conversation" and conv["title"] == "What is the first-line therapy for hypertension?"
    assert events[-1]["conversation_id"] == conv["id"]

    calls: list = []
    llm.PROVIDERS["anthropic"] = provider("Yes [S1].", calls=calls)
    follow_up = chat(api, "Is it safe?", conversation_id=conv["id"])
    assert follow_up[0]["type"] != "conversation"  # same conversation, no new one
    assert [m["role"] for m in calls[0]["messages"]] == ["user", "assistant", "user"]  # history sent to the model

    listed = api.get("/conversations", headers=H).json()
    assert [c["id"] for c in listed] == [conv["id"]]
    msgs = api.get(f"/conversations/{conv['id']}/messages", headers=H).json()
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[1]["citations"][0]["page_number"] == 4 and msgs[1]["provider"] == "anthropic"


def test_conversations_are_isolated_between_workspaces(api, fake_db):
    llm.PROVIDERS["anthropic"] = provider("hi")
    conv_id = chat(api, "hello")[0]["id"]
    other = {"X-Workspace-Id": OTHER_WS}
    assert api.get("/conversations", headers=other).json() == []
    assert api.get(f"/conversations/{conv_id}/messages", headers=other).status_code == 404
    assert chat(api, "hello", headers=other, conversation_id=conv_id)[-1]["type"] == "error"


def test_rename_and_delete_conversation(api, fake_db):
    conv_id = chat(api, "hello")[0]["id"]
    assert api.patch(f"/conversations/{conv_id}", headers=H, json={"title": "Dosing questions"}).status_code == 200
    assert api.get("/conversations", headers=H).json()[0]["title"] == "Dosing questions"
    api.delete(f"/conversations/{conv_id}", headers=H)
    assert api.get("/conversations", headers=H).json() == []
    assert fake_db.tables["messages"] == []


def test_selected_model_is_used_and_reported(api, fake_db):
    ready_doc(fake_db)
    fake_db.rpc_rows = [CHUNK]
    calls: list = []
    llm.PROVIDERS["openai"] = provider("Thiazide [S1].", calls=calls)
    events = chat(api, "First-line therapy?", provider="openai", model="gpt-5-mini")
    assert calls[0]["model"] == "gpt-5-mini"
    model_event = next(e for e in events if e["type"] == "model")
    assert model_event == {"type": "model", "provider": "openai", "model": "gpt-5-mini", "model_name": "GPT-5 mini"}
    assert events[-1]["model_name"] == "GPT-5 mini"


def test_unknown_model_falls_back_to_provider_default(api, fake_db):
    ready_doc(fake_db)
    calls: list = []
    llm.PROVIDERS["anthropic"] = provider("ok", calls=calls)
    chat(api, "hello there", provider="anthropic", model="gpt-unknown")
    assert calls[0]["model"] == llm.default_model("anthropic")


def test_models_catalog_lists_three_models_per_provider(api, monkeypatch):
    monkeypatch.setattr(main.settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(main.settings, "openai_api_key", "")
    data = api.get("/models").json()
    by_id = {p["id"]: p for p in data["providers"]}
    assert [len(by_id[p]["models"]) for p in ("anthropic", "openai")] == [3, 3]
    assert by_id["anthropic"]["configured"] and not by_id["openai"]["configured"]
    assert by_id["openai"]["status"] == "not_configured"
    assert data["default"]["provider"] == "anthropic"
