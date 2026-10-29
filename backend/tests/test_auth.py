import asyncio
import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException

from app import auth

PRIVATE = ec.generate_private_key(ec.SECP256R1())
JWK = jwt.PyJWK(json.loads(jwt.algorithms.ECAlgorithm.to_jwk(PRIVATE.public_key())) | {"kid": "k1", "alg": "ES256"})
USER = "44444444-4444-4444-4444-444444444444"


def token(**overrides) -> str:
    claims = {"sub": USER, "aud": "authenticated", "iss": f"{auth._base()}/auth/v1", "exp": int(time.time()) + 600} | overrides
    return jwt.encode(claims, PRIVATE, algorithm="ES256", headers={"kid": "k1"})


@pytest.fixture(autouse=True)
def local_jwks(monkeypatch):
    async def key(kid):
        return JWK if kid == "k1" else None
    monkeypatch.setattr(auth, "_signing_key", key)


def verify(t: str):
    return asyncio.run(auth.verify(t))


def test_valid_token_returns_the_user_id():
    assert verify(token()) == USER


@pytest.mark.parametrize("bad", [
    {"exp": int(time.time()) - 10},
    {"aud": "anon"},
    {"iss": "https://evil.example.com/auth/v1"},
])
def test_invalid_claims_are_rejected(bad):
    with pytest.raises(HTTPException) as e:
        verify(token(**bad))
    assert e.value.status_code == 401


def test_tampered_and_garbage_tokens_are_rejected():
    head, payload, sig = token().split(".")
    forged = jwt.utils.base64url_encode(json.dumps({"sub": "someone-else"}).encode()).decode()
    for t in (f"{head}.{forged}.{sig}", "not-a-jwt", ""):
        with pytest.raises(HTTPException) as e:
            verify(t)
        assert e.value.status_code == 401


def test_expired_token_gets_a_clear_message():
    with pytest.raises(HTTPException) as e:
        verify(token(exp=int(time.time()) - 10))
    assert "expired" in e.value.detail
