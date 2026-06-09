"""Outbound response models for the api service.

All response shapes here reproduce the exact current wire format consumed by the SPA.
Types are defined to match byte-for-byte the golden fixtures in docs/migration-fixtures/.
"""

from __future__ import annotations

# Response shapes are returned as typed dicts from service methods to guarantee
# byte-identical JSON with the golden fixtures. Pydantic models are kept here for
# documentation purposes and future tightening; routes return plain dicts from services.

# Wire shapes (documented, not enforced at serialization time to avoid any drift risk):
#
# GET /api/health      -> {"status": "ok"}
#
# GET /api/categories  -> {"items": [{"id", "parent_id", "name", "slug", "product_count"}]}
#
# GET /api/products/{id}
#   -> {"id", "slug", "name", "brand", "product_type", "price", "currency", "image",
#        "description", "attributes", "compatibility_tags", "category_path"}
#
# GET /api/products/{id}/recommendations
#   -> {"product_id": int, "items": [card + "context_code", "hybrid_score",
#        "semantic_score", "logical_score"]}
#
# GET /api/top-products -> {"items": [card + "reco_count"]}
#
# GET /api/summary
#   -> {"model", "catalog", "evaluated", "with_reco", "coverage",
#        "total_pairs", "verdict1_pairs", "verdict0_pairs", "verdict1_share",
#        "avg_semantic", "avg_logical", "avg_hybrid", "by_context"}
#
# GET /api/metrics
#   -> {"catalog", "coverage", "contextDist", "pAtK", "alpha", "best_alpha", "stats"}
#
# POST /api/search
#   -> {"query": str, "items": [{"id", "name", "brand", "product_type",
#        "price", "currency", "image"}]}
