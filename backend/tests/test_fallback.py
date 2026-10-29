import asyncio
import json

import anthropic
import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app import db, embeddings, llm, main

WS = "22222222-2222-2222-2222-222222222222"
CHUNKS = [{"document_id": "d1", "filename": "a.pdf", "page_number": 3, "content": "Metformin dose is 500 mg.", "similarity": 0.9}]


def status_error(cls, status: int, message: str):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.example.com"))
    return cls(message, response=response, body=None)


def provider(*chunks, fail: Exception | None = None):
    async def gen(_messages):
        for c in chunks:
            yield c
        if fail:
            raise fail
    return gen


def run(primary, fallback, order=("anthropic", "openai")):
    async def collect():
        return [e async for e in llm.stream_answer("q", CHUNKS, [])]
    llm.PROVIDERS["anthropic"], llm.PROVIDERS["openai"] = primary, fallback
    return asyncio.run(collect())


@pytest.fixture(autouse=True)
def restore_providers():
    saved = dict(llm.PROVIDERS)
    yield
    llm.PROVIDERS.update(saved)


def test_primary_success_does_not_touch_fallback():
    events = run(provider("Dose ", "500 mg [S1]."), provider(fail=AssertionError("must not be called")))
    assert events == [("provider", "anthropic"), ("text", "Dose "), ("text", "500 mg [S1].")]


def test_falls_back_to_openai_when_anthropic_fails_before_output():
    credit = status_error(anthropic.BadRequestError, 400, "Your credit balance is too low to access the Anthropic API.")
    events = run(provider(fail=credit), provider("From OpenAI [S1]."))
    assert events == [("provider", "openai"), ("text", "From OpenAI [S1].")]


def test_mid_stream_failure_resets_partial_text_then_falls_back():
    events = run(provider("Partial", fail=anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))),
                 provider("Full answer [S1]."))
    assert events == [("provider", "anthropic"), ("text", "Partial"), ("reset", ""),
                      ("provider", "openai"), ("text", "Full answer [S1].")]


def test_openai_primary_falls_back_to_anthropic(monkeypatch):
    monkeypatch.setattr(main.settings, "llm_primary", "openai")
    quota = status_error(openai.RateLimitError, 429, "You exceeded your current quota (insufficient_quota).")
    events = run(provider("Claude answer [S1]."), provider(fail=quota))
    assert events[0] == ("provider", "anthropic")


def test_both_fail_gives_friendly_non_technical_message():
    credit = status_error(anthropic.BadRequestError, 400, "Your credit balance is too low to access the Anthropic API.")
    bad_key = status_error(openai.AuthenticationError, 401, "Incorrect API key provided: sk-abc***")
    with pytest.raises(llm.AllProvidersFailed) as info:
        run(provider(fail=credit), provider(fail=bad_key))
    msg = info.value.user_message()
    assert "Claude (Anthropic): the account has run out of credits." in msg
    assert "OpenAI: the API key was not accepted." in msg
    assert "sk-" not in msg and "400" not in msg and "401" not in msg


@pytest.mark.parametrize("exc,reason", [
    (llm.ProviderNotConfigured("x"), "not set up on the server"),
    (status_error(anthropic.RateLimitError, 429, "rate limited"), "the service is busy right now"),
    (status_error(openai.NotFoundError, 404, "model gpt-x does not exist"), "the configured AI model is not available"),
    (status_error(anthropic.InternalServerError, 529, "Overloaded"), "the service is temporarily down or overloaded"),
    (openai.APITimeoutError(request=httpx.Request("POST", "https://x")), "the service took too long to respond"),
    (ValueError("weird"), "the service returned an unexpected error"),
])
def test_describe_failure(exc, reason):
    assert llm.describe_failure(exc) == reason


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(embeddings, "warmup", lambda: None)

    async def fake_select(*a, **k):
        return []

    async def fake_retrieve(*a, **k):
        return CHUNKS

    async def fake_insert(*a, **k):
        return None

    monkeypatch.setattr(db, "select", fake_select)
    monkeypatch.setattr(db, "insert", fake_insert)
    monkeypatch.setattr(main, "retrieve", fake_retrieve)
    with TestClient(main.app) as c:
        yield c


def chat_events(api):
    r = api.post("/chat", headers={"X-Workspace-Id": WS}, json={"question": "What is the metformin dose?"})
    return [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]


def test_chat_streams_full_text_even_when_it_starts_like_the_sentinel(api):
    llm.PROVIDERS["anthropic"] = provider("I", "n the documents, the dose is 500 mg [S1].")
    events = chat_events(api)
    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert streamed == "In the documents, the dose is 500 mg [S1]."
    assert events[-1]["provider"] == "anthropic" and events[-1]["citations"][0]["page_number"] == 3


def test_chat_sentinel_becomes_no_evidence_answer(api):
    llm.PROVIDERS["anthropic"] = provider("INSUFFICIENT", "_EVIDENCE")
    events = chat_events(api)
    assert not [e for e in events if e["type"] == "token"]
    assert events[-1]["answer"] == llm.NO_EVIDENCE and events[-1]["citations"] == []


def test_chat_both_providers_down_returns_friendly_error_event(api):
    llm.PROVIDERS["anthropic"] = provider(fail=llm.ProviderNotConfigured("x"))
    llm.PROVIDERS["openai"] = provider(fail=status_error(openai.AuthenticationError, 401, "bad key"))
    events = chat_events(api)
    assert events[-1]["type"] == "error"
    assert "Claude (Anthropic): not set up on the server." in events[-1]["message"]
    assert "OpenAI: the API key was not accepted." in events[-1]["message"]
