import asyncio
import json

import anthropic
import httpx
import openai
import pytest

from app import llm
from tests.conftest import WS, provider, ready_doc

CHUNKS = [{"document_id": "d1", "filename": "a.pdf", "page_number": 3, "content": "Metformin dose is 500 mg.", "similarity": 0.9}]


def status_error(cls, status: int, message: str):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.example.com"))
    return cls(message, response=response, body=None)


CREDIT = status_error(anthropic.BadRequestError, 400, "Your credit balance is too low to access the Anthropic API.")


def run(anthropic_gen, openai_gen, **choice):
    async def collect():
        return [e async for e in llm.stream_answer("q", CHUNKS, [], [], **choice)]
    llm.PROVIDERS["anthropic"], llm.PROVIDERS["openai"] = anthropic_gen, openai_gen
    return asyncio.run(collect())


@pytest.fixture(autouse=True)
def restore_providers():
    saved = dict(llm.PROVIDERS)
    llm._last_status.clear()
    yield
    llm.PROVIDERS.update(saved)
    llm._last_status.clear()


def start(p, m):
    return ("start", {"provider": p, "model": m})


def test_chosen_model_success_does_not_touch_fallback():
    events = run(provider("Dose ", "500 mg [S1]."), provider(fail=AssertionError("must not be called")),
                 provider="anthropic", model="claude-haiku-5-5")
    assert events == [start("anthropic", "claude-haiku-5-5"), ("text", "Dose "), ("text", "500 mg [S1].")]
    assert llm._last_status["anthropic"]["status"] == "ok"


def test_falls_back_to_openai_default_model_when_anthropic_fails_before_output():
    events = run(provider(fail=CREDIT), provider("From OpenAI [S1]."), provider="anthropic")
    assert events == [start("openai", llm.default_model("openai")), ("text", "From OpenAI [S1].")]
    assert llm._last_status["anthropic"] == {**llm._last_status["anthropic"], "status": "error",
                                             "reason": "the account has run out of credits"}


def test_mid_stream_failure_resets_partial_text_then_falls_back():
    events = run(provider("Partial", fail=anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))),
                 provider("Full answer [S1]."), provider="anthropic", model="claude-opus-5-5")
    assert events == [start("anthropic", "claude-opus-5-5"), ("text", "Partial"), ("reset", None),
                      start("openai", llm.default_model("openai")), ("text", "Full answer [S1].")]


def test_openai_choice_falls_back_to_anthropic():
    quota = status_error(openai.RateLimitError, 429, "You exceeded your current quota (insufficient_quota).")
    events = run(provider("Claude answer [S1]."), provider(fail=quota), provider="openai", model="gpt-5")
    assert events[0] == start("anthropic", llm.default_model("anthropic"))


def test_both_fail_gives_friendly_non_technical_message():
    bad_key = status_error(openai.AuthenticationError, 401, "Incorrect API key provided: sk-abc***")
    with pytest.raises(llm.AllProvidersFailed) as info:
        run(provider(fail=CREDIT), provider(fail=bad_key), provider="anthropic")
    msg = info.value.user_message()
    assert "Claude (Anthropic): the account has run out of credits." in msg
    assert "OpenAI: the API key was not accepted." in msg
    assert "sk-" not in msg and "400" not in msg and "401" not in msg


def test_not_configured_provider_is_not_reported_as_an_error():
    run(provider("ok"), provider(fail=llm.ProviderNotConfigured("x")), provider="openai")
    assert "openai" not in llm._last_status and llm._last_status["anthropic"]["status"] == "ok"


@pytest.mark.parametrize("exc,reason", [
    (llm.ProviderNotConfigured("x"), "not set up on the server"),
    (status_error(anthropic.RateLimitError, 429, "rate limited"), "the service is busy right now"),
    (status_error(openai.NotFoundError, 404, "model gpt-x does not exist"), "the selected AI model is not available"),
    (status_error(anthropic.InternalServerError, 529, "Overloaded"), "the service is temporarily down or overloaded"),
    (openai.APITimeoutError(request=httpx.Request("POST", "https://x")), "the service took too long to respond"),
    (ValueError("weird"), "the service returned an unexpected error"),
])
def test_describe_failure(exc, reason):
    assert llm.describe_failure(exc) == reason


def chat_events(api, **body):
    r = api.post("/chat", headers={"X-Workspace-Id": WS}, json={"question": "What is the metformin dose?", **body})
    return [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]


def test_chat_streams_every_token_including_the_first(api, fake_db):
    ready_doc(fake_db)
    fake_db.rpc_rows = [{**CHUNKS[0], "id": 1, "score": 0.03}]
    llm.PROVIDERS["anthropic"] = provider("I", "n the documents, the dose is 500 mg [S1].")
    events = chat_events(api, provider="anthropic")
    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert streamed == "In the documents, the dose is 500 mg [S1]."
    assert events[-1]["provider"] == "anthropic" and events[-1]["citations"][0]["page_number"] == 3


def test_chat_reset_event_is_sent_on_mid_answer_failover(api, fake_db):
    ready_doc(fake_db)
    llm.PROVIDERS["anthropic"] = provider("Part", fail=status_error(anthropic.InternalServerError, 500, "boom"))
    llm.PROVIDERS["openai"] = provider("Complete answer.")
    events = chat_events(api, provider="anthropic")
    kinds = [e["type"] for e in events]
    assert kinds.index("reset") < len(kinds) - 1 and events[-1]["answer"] == "Complete answer."
    assert events[-1]["provider"] == "openai"


def test_chat_both_providers_down_returns_friendly_error_event(api, fake_db):
    ready_doc(fake_db)
    llm.PROVIDERS["anthropic"] = provider(fail=llm.ProviderNotConfigured("x"))
    llm.PROVIDERS["openai"] = provider(fail=status_error(openai.AuthenticationError, 401, "bad key"))
    events = chat_events(api)
    assert events[-1]["type"] == "error"
    assert "Claude (Anthropic): not set up on the server." in events[-1]["message"]
    assert "OpenAI: the API key was not accepted." in events[-1]["message"]
