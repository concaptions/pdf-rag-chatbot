"""Local ONNX embeddings (fastembed, BAAI/bge-small-en-v1.5, 384 dims).

Runs in-process: no extra API key, no network hop, ~ms per query.
"""

import asyncio
from functools import lru_cache

from app.config import get_settings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(get_settings().embedding_model)


def warmup() -> None:
    list(_model().embed(["warmup"]))


async def embed_documents(texts: list[str]) -> list[list[float]]:
    def run():
        return [v.tolist() for v in _model().embed(texts, batch_size=64)]

    return await asyncio.to_thread(run)


async def embed_query(text: str) -> list[float]:
    def run():
        return next(iter(_model().embed([QUERY_PREFIX + text]))).tolist()

    return await asyncio.to_thread(run)
