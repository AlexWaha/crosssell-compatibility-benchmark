"""Inbound request models for the api service."""

from __future__ import annotations

from pydantic import BaseModel


class SearchRequest(BaseModel):
    """POST /api/search request body.

    Attributes:
        query: Search query string.
        limit: Maximum number of results to return.
    """

    query: str
    limit: int = 24


class EmbedCallBody(BaseModel):
    """Body sent to engine /embed for query vector delegation.

    Attributes:
        text: Text to embed.
    """

    text: str
