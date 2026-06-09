"""Ground-truth judge for compatibility pairs.

judge_pair() calls an LLM (JUDGE_MODEL = gpt-5) with a strict JSON schema
to produce a binary compatibility label + rationale for one product pair.

select_sample() draws a stratified sample from recommendations for the judge.

MockJudge returns deterministic labels ($0) for use in tests and CI.

The judge label pipeline:
  select_sample() -> list of (product_i, product_j) pairs
  -> judge_pair() per pair
  -> write_ground_truth() -> ground_truth table
  -> load_ground_truth() -> loaded by run_eval -> non-null P/R/NDCG
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.db.metrics_repository import MetricsRepository

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge response schema (strict JSON schema)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You are a senior technical compatibility expert reviewing product pairs for an "
    "e-commerce cross-selling engine. Your task is to determine whether two products "
    "are technically compatible for cross-selling (i.e., a customer who buys product A "
    "would genuinely benefit from also buying product B).\n\n"
    "Output ONLY valid JSON. Do NOT add explanations outside the JSON object."
)

JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "compatibility_judgment",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "label": {
                    "type": "integer",
                    "description": "1 if compatible for cross-selling, 0 if not",
                    "enum": [0, 1],
                },
                "rationale": {
                    "type": "string",
                    "description": "One-sentence technical justification for the label",
                },
            },
            "required": ["label", "rationale"],
        },
    },
}


def _build_judge_prompt(source_doc: dict, cand_doc: dict) -> str:
    """Build the user prompt for a judge call.

    Args:
        source_doc: Source product document (from recommendations or Typesense).
        cand_doc: Candidate product document.

    Returns:
        Formatted user prompt string.
    """
    src_name = source_doc.get("name", f"product {source_doc.get('product_id', '?')}")
    cnd_name = cand_doc.get("name", f"product {cand_doc.get('product_id', '?')}")
    src_type = source_doc.get("product_type", "unknown")
    cnd_type = cand_doc.get("product_type", "unknown")

    src_attrs = source_doc.get("attributes") or {}
    cnd_attrs = cand_doc.get("attributes") or {}

    lines = [
        f"Source product: {src_name}",
        f"  Type: {src_type}",
    ]
    if src_attrs:
        lines.append(f"  Key attributes: {json.dumps(src_attrs, ensure_ascii=False)}")

    lines += [
        f"\nCandidate product: {cnd_name}",
        f"  Type: {cnd_type}",
    ]
    if cnd_attrs:
        lines.append(f"  Key attributes: {json.dumps(cnd_attrs, ensure_ascii=False)}")

    lines.append(
        "\nIs the candidate product technically compatible with the source for "
        "cross-selling? Respond with label=1 (compatible) or label=0 (not compatible) "
        "and a brief one-sentence rationale."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Real judge (delegates to LLM)
# ---------------------------------------------------------------------------


async def judge_pair(
    source_doc: dict,
    cand_doc: dict,
    llm: Any,
    judge_model: str = "gpt-5",
) -> dict:
    """Call the LLM judge for a single product pair.

    Args:
        source_doc: Source product document.
        cand_doc: Candidate product document.
        llm: LLM client with generate(system, user, response_format) interface.
        judge_model: Model identifier recorded in the ground_truth row.

    Returns:
        Dict with keys: product_i, product_j, context_code, label, source,
        judge_model, rationale.

    Raises:
        ValueError: If LLM response cannot be parsed or lacks required keys.
    """
    user_prompt = _build_judge_prompt(source_doc, cand_doc)
    raw_data, _ti, _to = await llm.generate(
        JUDGE_SYSTEM_PROMPT, user_prompt, JUDGE_RESPONSE_FORMAT
    )

    if isinstance(raw_data, str):
        try:
            raw_data = json.loads(raw_data)
        except json.JSONDecodeError:
            raise ValueError(f"judge: cannot parse LLM response: {raw_data[:200]}")

    if not isinstance(raw_data, dict):
        raise ValueError(f"judge: expected dict, got {type(raw_data).__name__}")

    label = raw_data.get("label")
    if label not in (0, 1):
        raise ValueError(f"judge: label must be 0 or 1, got {label!r}")

    rationale = str(raw_data.get("rationale", ""))
    product_i = source_doc.get("product_id", 0)
    product_j = cand_doc.get("product_id", 0)

    return {
        "product_i": int(product_i),
        "product_j": int(product_j),
        "context_code": source_doc.get("context_code"),
        "label": int(label),
        "source": "llm",
        "judge_model": judge_model,
        "rationale": rationale,
    }


JUDGE_BATCH_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "compatibility_judgments",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "candidate_id": {"type": "integer"},
                            "label": {"type": "integer", "enum": [0, 1]},
                            "rationale": {"type": "string"},
                        },
                        "required": ["candidate_id", "label", "rationale"],
                    },
                }
            },
            "required": ["results"],
        },
    },
}


def _cats(doc: dict) -> str:
    """Render a product's category path for grounding the judge in real taxonomy."""
    cats = doc.get("categories") or []
    return " / ".join(str(c) for c in cats) if cats else "uncategorized"


# Domain-agnostic decision policy. The judge reasons from the REAL category of each
# product (injected per pair), so it works for every vertical in the catalog
# (phones, computing, photo/video, cycling, automotive, tools, ...), not just a few.
JUDGE_POLICY = (
    "Decide, for EACH candidate, whether a customer buying the SOURCE would also "
    "reasonably buy the candidate AS A COMPANION and it is technically usable with "
    "the source. Output label=1 (good cross-sell companion) or label=0.\n"
    "\nReason from the REAL category of each product (given per item). Two classes:\n"
    "\nA) UNIVERSAL companions - fit ANY product of the source's platform; do NOT "
    "require exact brand/model match. Label=1 unless an explicit spec clash is shown. "
    "Examples across domains:\n"
    "   - computing: internal SSD/HDD/RAM (standard SATA/NVMe/DDR), USB flash, "
    "card readers/USB hubs, mice, keyboards, mouse pads, webcams, laptop bags/coolers, "
    "monitors, surge protectors, standard-connector chargers/power banks.\n"
    "   - photo/video: tripods, camera bags, cleaning kits, generic LED lights, "
    "memory cards (cameras DO have card slots).\n"
    "   - cycling: lights, bottles & cages, computers/speedometers, locks, pumps, "
    "generic mounts.\n"
    "   - tools: drill/driver bit & socket sets, tool boxes/kits, safety helmets/"
    "glasses, work gloves.\n"
    "   - phones/tablets: power banks, standard USB/USB-C chargers, wireless "
    "headphones/headsets, holders/stands.\n"
    "\nB) FIT-CRITICAL companions - compatible ONLY when a specific spec matches; "
    "judge from names/attributes and label=0 on a clear mismatch (and when the source "
    "plainly lacks the interface):\n"
    "   - phone/tablet CASE or screen protector -> must be for the SAME model/brand "
    "(an iPhone case does NOT fit a Samsung Galaxy).\n"
    "   - memory card -> only if the device has a card slot (most iPhones do NOT).\n"
    "   - camera LENS -> lens mount must match the camera body; lens FILTER -> "
    "filter thread diameter must match.\n"
    "   - battery / model-specific charger -> must match the device model.\n"
    "   - TYRES -> must match vehicle type & size (bike vs motorcycle vs car).\n"
    "   - CPU/motherboard -> socket/chipset must match.\n"
    "\nGuidance: be inclusive for class A (when in doubt about a universal companion "
    "of the right platform, prefer label=1); be strict for class B (when in doubt "
    "about a fit-critical item, prefer label=0). Never reject a universal companion "
    "merely because the brand differs from the source."
)


def _build_judge_batch_prompt(source_doc: dict, cand_docs: list[dict]) -> str:
    """Build a batched judge prompt: one source vs many candidates, taxonomy-grounded."""
    src_name = source_doc.get("name", f"product {source_doc.get('product_id', '?')}")
    lines = [
        f"SOURCE product: {src_name}",
        f"  category: {_cats(source_doc)}",
        f"  type: {source_doc.get('product_type', 'unknown')}",
    ]
    if source_doc.get("attributes"):
        lines.append(
            f"  attributes: {json.dumps(source_doc['attributes'], ensure_ascii=False)}"
        )
    lines.append("\n" + JUDGE_POLICY)
    lines.append("\nCANDIDATES:")
    for c in cand_docs:
        lines.append(
            f"\n[ID={c.get('product_id', 0)}] {c.get('name', 'Unknown')}"
            f"\n   category: {_cats(c)} | type: {c.get('product_type', 'unknown')}"
        )
        if c.get("attributes"):
            lines.append(
                f"   attributes: {json.dumps(c['attributes'], ensure_ascii=False)}"
            )
    return "\n".join(lines)


async def judge_batch(
    source_doc: dict,
    cand_docs: list[dict],
    llm: Any,
    judge_model: str = "gpt-5",
) -> tuple[list[dict], int, int]:
    """Judge one source against many candidates in a single LLM call.

    Cuts cost ~Nx vs judge_pair by batching candidates. Returns one ground_truth
    row dict per candidate plus the prompt/completion token counts so the caller
    can enforce a budget guard.

    Args:
        source_doc: Source product document (with real name/type/attributes).
        cand_docs: Candidate product documents.
        llm: LLM client exposing generate(system, user, response_format).
        judge_model: Model id recorded in the ground_truth rows.

    Returns:
        Tuple of (rows, tokens_in, tokens_out). rows keys match write_ground_truth:
        product_i, product_j, context_code, label, source, judge_model, rationale.
        Candidates the model omits are skipped.
    """
    user_prompt = _build_judge_batch_prompt(source_doc, cand_docs)
    raw, ti, to = await llm.generate(
        JUDGE_SYSTEM_PROMPT, user_prompt, JUDGE_BATCH_RESPONSE_FORMAT
    )
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"judge_batch: cannot parse response: {raw[:200]}")
    results = raw.get("results", []) if isinstance(raw, dict) else []

    product_i = int(source_doc.get("product_id", 0))
    by_id = {int(c.get("product_id", 0)): c for c in cand_docs}
    rows: list[dict] = []
    for r in results:
        cid = r.get("candidate_id")
        if cid not in by_id:
            continue
        label = r.get("label")
        if label not in (0, 1):
            continue
        rows.append(
            {
                "product_i": product_i,
                "product_j": int(cid),
                "context_code": source_doc.get("context_code"),
                "label": int(label),
                "source": "llm",
                "judge_model": judge_model,
                "rationale": str(r.get("rationale", ""))[:500],
            }
        )
    return rows, ti, to


# ---------------------------------------------------------------------------
# Mock judge (deterministic, $0)
# ---------------------------------------------------------------------------


class MockJudge:
    """Deterministic mock judge for $0 testing.

    Label is derived from a hash of (product_i, product_j) so results are
    stable across reruns and deterministically mixed (not all-zero, not all-one).

    Attributes:
        judge_model: Model identifier recorded in ground_truth rows.
        positive_rate: Fraction of pairs labelled as compatible (0.0-1.0).
    """

    def __init__(
        self, judge_model: str = "mock_judge", positive_rate: float = 0.4
    ) -> None:
        self.judge_model = judge_model
        self.positive_rate = positive_rate

    def _deterministic_label(self, product_i: int, product_j: int) -> int:
        """Hash-based deterministic label in [0, 1]."""
        digest = hashlib.md5(f"{product_i}:{product_j}".encode()).hexdigest()
        frac = int(digest[:4], 16) / 0xFFFF
        return 1 if frac < self.positive_rate else 0

    async def judge_pair(self, source_doc: dict, cand_doc: dict) -> dict:
        """Return a deterministic mock judgment for a product pair.

        source is set to "llm" (same as real judge) so the row lands in the
        ground_truth table under the unique key (product_i, product_j, source).
        load_ground_truth filters by source='human', so mock GT rows written
        here are isolated and do NOT pollute real evaluation metrics.

        Args:
            source_doc: Source product document.
            cand_doc: Candidate product document.

        Returns:
            Dict with keys: product_i, product_j, context_code, label,
            source, judge_model, rationale.
        """
        product_i = int(source_doc.get("product_id", 0))
        product_j = int(cand_doc.get("product_id", 0))
        label = self._deterministic_label(product_i, product_j)
        rationale = (
            f"Mock judgment: {'compatible' if label else 'not compatible'} "
            f"(deterministic hash of ids {product_i},{product_j})"
        )
        return {
            "product_i": product_i,
            "product_j": product_j,
            "context_code": source_doc.get("context_code"),
            "label": label,
            "source": "llm",
            "judge_model": self.judge_model,
            "rationale": rationale,
        }


# ---------------------------------------------------------------------------
# Sample selector
# ---------------------------------------------------------------------------


async def select_sample(
    repo: "MetricsRepository",
    experiment_id: str,
    n: int = 500,
    stratify_by: str = "category",
) -> list[dict]:
    """Select a stratified sample of pairs from recommendations for judging.

    Delegates to MetricsRepository.sample_pairs_for_judge() which filters
    out pairs already present in ground_truth.

    Args:
        repo: MetricsRepository connected to avtc_metrics.
        experiment_id: Experiment whose recommendations to sample from.
        n: Maximum number of pairs to return.
        stratify_by: Stratification field (currently informational only;
            the underlying query uses ORDER BY RAND() with a LIMIT).

    Returns:
        List of pair dicts with keys: product_i, product_j, context_code,
        semantic_score, logical_score, hybrid_score.
    """
    pairs = await repo.sample_pairs_for_judge(experiment_id, n, stratify_by)
    log.info(
        "selected %d pairs for judging (experiment=%s, n=%d)",
        len(pairs),
        experiment_id,
        n,
    )
    return pairs
