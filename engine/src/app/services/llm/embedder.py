"""Embedding generator - OpenAI text-embedding-3-large (1024 dimensions).

Verbatim from old engine/embedder.py.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import tiktoken
from openai import AsyncOpenAI

log = logging.getLogger(__name__)

_encoding = tiktoken.get_encoding("cl100k_base")


def _chunk_by_tokens(
    texts: list[str],
    indices: list[int],
    max_tokens: int,
) -> Iterator[tuple[list[str], list[int]]]:
    """Yield (chunk_texts, chunk_indices) groups that fit within max_tokens.

    Args:
        texts: Non-empty texts to embed.
        indices: Original indices corresponding to each text.
        max_tokens: Maximum total token count per chunk.
    """
    chunk_texts: list[str] = []
    chunk_indices: list[int] = []
    chunk_tokens = 0

    for text, idx in zip(texts, indices, strict=True):
        text_tokens = len(_encoding.encode(text))
        # If a single text exceeds max_tokens, send it alone
        if text_tokens >= max_tokens:
            if chunk_texts:
                yield chunk_texts, chunk_indices
                chunk_texts, chunk_indices, chunk_tokens = [], [], 0
            yield [text], [idx]
            continue

        if chunk_tokens + text_tokens > max_tokens:
            yield chunk_texts, chunk_indices
            chunk_texts, chunk_indices, chunk_tokens = [], [], 0

        chunk_texts.append(text)
        chunk_indices.append(idx)
        chunk_tokens += text_tokens

    if chunk_texts:
        yield chunk_texts, chunk_indices


QUERY_INSTRUCTION = "Instruct: Given a product search query, retrieve technically relevant products.\nQuery: "


def as_query(text: str) -> str:
    """Wrap a search query with the Qwen3 retrieval instruction prefix.

    Qwen3-Embedding is instruction-aware: queries get an instruction prefix while
    documents are embedded raw. Applying this only to queries improves retrieval.
    """
    return f"{QUERY_INSTRUCTION}{text}"


async def embed(
    client: AsyncOpenAI,
    text: str,
    model: str = "text-embedding-3-large",
    dimensions: int | None = 1024,
) -> list[float]:
    """Generate embedding for a single text.

    Returns a zero vector if the text is empty. dimensions is only sent when set;
    local models (Qwen3 via LM Studio) reject the OpenAI-specific dimensions arg,
    so pass None for them (Qwen3-0.6B is natively 1024-dim).
    """
    fallback_dims = dimensions or 1024
    if not text or not text.strip():
        return [0.0] * fallback_dims

    kwargs: dict = {"model": model, "input": text}
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    response = await client.embeddings.create(**kwargs)
    return response.data[0].embedding


async def embed_batch(
    client: AsyncOpenAI,
    texts: list[str],
    model: str = "text-embedding-3-large",
    dimensions: int = 1024,
    max_tokens: int = 250_000,
) -> list[list[float]]:
    """Generate embeddings for a batch of texts.

    Empty texts get zero vectors. Splits into token-aware chunks
    to stay within the OpenAI API token limit.

    Args:
        client: OpenAI async client.
        texts: Texts to embed.
        model: Embedding model name.
        dimensions: Embedding dimensions.
        max_tokens: Max tokens per API call (default 250k, API limit is 300k).

    Returns:
        List of embedding vectors, one per input text.
    """
    if not texts:
        return []

    results: list[list[float]] = [[0.0] * dimensions for _ in range(len(texts))]
    valid_indices: list[int] = []
    valid_texts: list[str] = []

    for i, t in enumerate(texts):
        if t and t.strip():
            valid_indices.append(i)
            valid_texts.append(t)

    if not valid_texts:
        return results

    for chunk_num, (chunk_texts, chunk_indices) in enumerate(
        _chunk_by_tokens(valid_texts, valid_indices, max_tokens),
        start=1,
    ):
        token_count = sum(len(_encoding.encode(t)) for t in chunk_texts)
        log.info(
            "embedding chunk %d: %d texts, %d tokens",
            chunk_num,
            len(chunk_texts),
            token_count,
        )
        response = await client.embeddings.create(
            model=model,
            input=chunk_texts,
            dimensions=dimensions,
        )
        for emb_data in response.data:
            original_idx = chunk_indices[emb_data.index]
            results[original_idx] = emb_data.embedding

    return results
