"""Search service: delegate query embedding to engine, call Typesense multi_search.

Falls back to keyword-only search if the engine embed call fails (log warning, never 500).
No LLM or embedding computation happens here - this service is a pure HTTP client.
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)


class SearchService:
    """Orchestrates hybrid search: engine embed -> Typesense multi_search.

    Args:
        http: Shared httpx.AsyncClient.
        engine_url: Base URL for the engine service (e.g. "http://engine:9000").
        typesense_base_url: Base URL for Typesense (e.g. "http://typesense:8108").
        typesense_api_key: Typesense API key header value.
        typesense_collection: Name of the Typesense collection to search.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        engine_url: str,
        typesense_base_url: str,
        typesense_api_key: str,
        typesense_collection: str,
    ) -> None:
        self._http = http
        self._engine_url = engine_url
        self._typesense_base_url = typesense_base_url
        self._typesense_api_key = typesense_api_key
        self._typesense_collection = typesense_collection

    async def search(self, query: str, limit: int) -> dict:
        """Perform a hybrid search and return the response dict.

        Attempts to get a query vector from the engine /embed endpoint. If that
        fails for any reason, falls back to keyword-only search. Never raises.

        Args:
            query: User search query string.
            limit: Maximum number of results to return.

        Returns:
            Dict with "query" and "items" keys matching the /api/search wire shape.
        """
        params: dict = {
            "collection": self._typesense_collection,
            "q": query,
            "query_by": "name,product_type,description,compatibility_tags",
            "exclude_fields": "embedding",
            "per_page": limit,
        }

        # Delegate query embedding to the engine (BFF does no compute).
        # Falls back to keyword-only search if engine is unavailable.
        try:
            r = await self._http.post(
                f"{self._engine_url}/embed",
                json={"text": query},
                timeout=15,
            )
            if r.status_code == 200:
                vec = r.json().get("embedding") or []
                if vec:
                    params["vector_query"] = (
                        f"embedding:([{','.join(str(v) for v in vec)}], k:{limit})"
                    )
        except Exception as exc:
            log.warning("engine embed unavailable, keyword-only: %s", exc)

        resp = await self._http.post(
            f"{self._typesense_base_url}/multi_search",
            json={"searches": [params]},
            headers={"X-TYPESENSE-API-KEY": self._typesense_api_key},
        )
        resp.raise_for_status()
        hits = resp.json().get("results", [{}])[0].get("hits", [])
        items = [
            {
                "id": d["document"].get("product_id"),
                "name": d["document"].get("name", ""),
                "brand": d["document"].get("brand", ""),
                "product_type": d["document"].get("product_type", ""),
                "price": d["document"].get("price"),
                "currency": "USD",
                "image": (d["document"].get("image") or "").removeprefix(
                    "catalog/product/"
                ),
            }
            for d in hits
        ]
        return {"query": query, "items": items}
