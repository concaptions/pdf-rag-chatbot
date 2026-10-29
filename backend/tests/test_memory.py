import asyncio
import json

from app import db, llm, memory
from tests.conftest import WS, provider, ready_doc

H = {"X-Workspace-Id": WS}
METFORMIN = {"id": 1, "document_id": "d1", "filename": "diabetes.pdf", "page_number": 4,
             "content": "Metformin starts at 500 mg daily.", "similarity": 0.8, "score": 0.03}
MALARIA = {"id": 2, "document_id": "d2", "filename": "tropical.pdf", "page_number": 9,
           "content": "Artemisinin combination therapy treats malaria.", "similarity": 0.8, "score": 0.03}


def seed(fake_db, n_messages: int, summary: str | None = None, summary_upto: int = 0) -> str:
    conv = asyncio.run(fake_db.insert("conversations", {"workspace_id": WS, "title": "Long chat",
                                                         "summary": summary, "summary_upto": summary_upto}))[0]
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        asyncio.run(fake_db.insert("messages", {"workspace_id": WS, "conversation_id": conv["id"], "role": role,
                                                "content": f"{role} message {i}", "citations": [], "provider": None, "model": None}))
    return conv["id"]


def chat(api, question: str, conv_id: str | None = None) -> list[dict]:
    r = api.post("/chat", headers=H, json={"question": question, "conversation_id": conv_id})
    return [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]


def test_model_receives_at_least_30_previous_messages(api, fake_db):
    ready_doc(fake_db)
    conv_id = seed(fake_db, 40)
    calls: list = []
    llm.PROVIDERS["anthropic"] = provider("ok", calls=calls)
    chat(api, "And the next question?", conv_id)
    sent = calls[0]["messages"]
    assert len(sent) == 41  # 40 earlier messages verbatim + the new question
    assert sent[0] == {"role": "user", "content": "user message 0"}
    assert sent[-2] == {"role": "assistant", "content": "assistant message 39"}


def test_history_never_starts_with_an_assistant_turn(api, fake_db):
    ready_doc(fake_db)
    conv_id = seed(fake_db, 60, summary="Earlier: dosing questions.", summary_upto=1)  # message 1 (user) folded
    calls: list = []
    llm.PROVIDERS["anthropic"] = provider("ok", calls=calls)
    chat(api, "Next?", conv_id)
    assert calls[0]["messages"][0]["role"] == "user"


def test_old_citation_labels_become_document_and_page(api, fake_db):
    ready_doc(fake_db)
    conv_id = seed(fake_db, 0)
    asyncio.run(fake_db.insert("messages", [
        {"workspace_id": WS, "conversation_id": conv_id, "role": "user", "content": "Dose?", "citations": [], "provider": None, "model": None},
        {"workspace_id": WS, "conversation_id": conv_id, "role": "assistant", "content": "500 mg daily [S1].",
         "citations": [{"label": "S1", "filename": "diabetes.pdf", "page_number": 4, "excerpt": "..."}], "provider": None, "model": None},
    ]))
    calls: list = []
    llm.PROVIDERS["anthropic"] = provider("ok", calls=calls)
    chat(api, "Thanks", conv_id)
    assert calls[0]["messages"][1]["content"] == "500 mg daily (diabetes.pdf, p. 4)."


def test_summary_and_only_unsummarised_messages_reach_the_model(api, fake_db):
    ready_doc(fake_db)
    conv_id = seed(fake_db, 40, summary="- User asked about metformin dosing (diabetes.pdf, p. 4).", summary_upto=20)
    calls: list = []
    llm.PROVIDERS["anthropic"] = provider("ok", calls=calls)
    chat(api, "Remind me what we discussed first", conv_id)
    assert calls[0]["summary"].startswith("- User asked about metformin")
    assert calls[0]["messages"][0]["content"] == "user message 20"  # ids 1..20 are folded into the summary
    assert len(calls[0]["messages"]) == 21


def test_compaction_folds_everything_older_than_the_window(fake_db, monkeypatch):
    conv_id = seed(fake_db, 50)
    seen: dict = {}

    async def fake_summarize(previous, messages):
        seen.update(previous=previous, count=len(messages), first=messages[0]["content"], last=messages[-1]["content"])
        return "- 10 questions about dosing."
    monkeypatch.setattr(llm, "summarize", fake_summarize)

    assert asyncio.run(memory.compact(conv_id)) is True
    assert seen == {"previous": "", "count": 20, "first": "user message 0", "last": "assistant message 19"}
    conv = next(c for c in fake_db.tables["conversations"] if c["id"] == conv_id)
    assert conv["summary"] == "- 10 questions about dosing." and conv["summary_upto"] == 20
    summary, recent = asyncio.run(memory.load(conv_id))
    assert summary == "- 10 questions about dosing." and len(recent) == 30 and recent[0]["role"] == "user"


def test_compaction_keeps_the_window_starting_on_a_user_turn(fake_db, monkeypatch):
    conv_id = seed(fake_db, 51)  # 51 - 30 = 21 would start the window on an assistant message

    async def fake_summarize(previous, messages):
        return "summary"
    monkeypatch.setattr(llm, "summarize", fake_summarize)
    asyncio.run(memory.compact(conv_id))
    _, recent = asyncio.run(memory.load(conv_id))
    assert recent[0]["role"] == "user" and len(recent) == 29


def test_compaction_is_skipped_below_threshold_and_on_summariser_failure(fake_db, monkeypatch):
    async def failing(previous, messages):
        return ""
    monkeypatch.setattr(llm, "summarize", failing)
    assert asyncio.run(memory.compact(seed(fake_db, 49))) is False
    full = seed(fake_db, 60)
    assert asyncio.run(memory.compact(full)) is False
    _, recent = asyncio.run(memory.load(full))
    assert len(recent) == 50  # nothing lost: the latest 50 are still sent verbatim


def test_chat_schedules_compaction_at_the_threshold(api, fake_db, monkeypatch):
    ready_doc(fake_db)
    scheduled: list = []
    monkeypatch.setattr(memory, "maybe_compact", lambda conv_id, n: scheduled.append(n))
    llm.PROVIDERS["anthropic"] = provider("ok")
    chat(api, "q", seed(fake_db, 48))
    assert scheduled == [50]


def test_missing_memory_columns_fall_back_to_the_window(fake_db, monkeypatch):
    conv_id = seed(fake_db, 10)
    real_select = fake_db.select

    async def select(table, params):
        if table == "conversations" and "summary" in params.get("select", ""):
            raise db.DatabaseError("Database error 400: column conversations.summary does not exist")
        return await real_select(table, params)
    monkeypatch.setattr(db, "select", select)
    summary, recent = asyncio.run(memory.load(conv_id))
    assert summary == "" and len(recent) == 10


def test_new_topic_after_a_long_thread_still_finds_its_passages(api, fake_db):
    ready_doc(fake_db, filename="diabetes.pdf")
    ready_doc(fake_db, filename="tropical.pdf")
    conv_id = seed(fake_db, 0)
    for i in range(15):  # 15 turns about metformin
        asyncio.run(fake_db.insert("messages", [
            {"workspace_id": WS, "conversation_id": conv_id, "role": "user", "content": f"metformin question {i}", "citations": [], "provider": None, "model": None},
            {"workspace_id": WS, "conversation_id": conv_id, "role": "assistant", "content": "metformin answer", "citations": [], "provider": None, "model": None},
        ]))
    fake_db.rpc_rows = lambda q: [METFORMIN] if "metformin" in q else [MALARIA]
    calls: list = []
    llm.PROVIDERS["anthropic"] = provider("Artemisinin combination therapy [S1].", calls=calls)
    events = chat(api, "What is the treatment for malaria?", conv_id)
    sources = calls[0]["messages"][-1]["content"].split("<sources>")[1]
    assert 'label="S1" document="tropical.pdf"' in sources  # the standalone search ranks first
    assert 'label="S2" document="diabetes.pdf"' in sources  # the follow-up search still contributes
    assert events[-1]["citations"][0]["filename"] == "tropical.pdf"
    assert [c["query_text"] for c in fake_db.rpc_calls] == [
        "What is the treatment for malaria?", "metformin question 14 What is the treatment for malaria?"]


def test_anthropic_history_gets_a_cache_breakpoint():
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "now"}]
    out = llm.with_history_cache(msgs)
    assert out[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert out[2] == msgs[2] and msgs[1]["content"] == "b"  # input untouched
    assert llm.with_history_cache([msgs[2]]) == [msgs[2]]
