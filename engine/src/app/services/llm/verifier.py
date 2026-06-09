"""LLM adapter with a real provider and a mock for test-first smoke testing.

One seam (LLMClient.verify) routes to OpenAI in real mode or to MockLLM in mock mode,
selected by Settings.LLM_MODE. The mock is parametrizable per scenario so the whole
pipeline can be validated end-to-end with $0 spend before any real call:

  valid     -> well-formed JSON that passes validation (verdict per candidate)
  slow      -> sleeps beyond the timeout, then responds (tests "monitor until response")
  error     -> raises an API-style error (tests retry / dead-letter)
  truncated -> returns truncated/invalid JSON (tests parse-fail handling, no crash)

Verbatim logic from old engine/llm.py with updated imports.
"""

from __future__ import annotations

import asyncio
import json
import logging

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a technical compatibility verification expert.\n"
    "Given a source product and a batch of candidate products, evaluate whether each\n"
    "candidate is technically compatible with the source for cross-selling.\n\n"
    'Output ONLY valid JSON, no markdown. Shape: {"results": [ ... ]} where each element has:\n'
    "- candidate_id: int\n- verdict: bool\n- logical_score: float 0-1\n"
    "- context_code: str (cable, mount, accessory, case, charger, adapter)\n"
    "- rules_evaluated: int\n- rules_passed: int\n- rules_failed: int\n- rules_undefined: int\n"
)

_RESULT_PROPS = {
    "candidate_id": {"type": "integer"},
    "verdict": {"type": "boolean"},
    "logical_score": {"type": "number"},
    "context_code": {"type": "string"},
    "rules_evaluated": {"type": "integer"},
    "rules_passed": {"type": "integer"},
    "rules_failed": {"type": "integer"},
    "rules_undefined": {"type": "integer"},
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "compatibility_results",
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
                        "properties": _RESULT_PROPS,
                        "required": list(_RESULT_PROPS),
                    },
                }
            },
            "required": ["results"],
        },
    },
}


class LLMError(Exception):
    """Raised by the LLM layer on a provider error (retryable by the worker)."""


def build_user_prompt(source: dict, candidates: list[dict]) -> str:
    lines = [f"Source product: {source.get('name', 'Unknown')}"]
    if source.get("product_type"):
        lines.append(f"Type: {source['product_type']}")
    if source.get("attributes"):
        lines.append(
            f"Attributes: {json.dumps(source['attributes'], ensure_ascii=False)}"
        )
    lines.append("\nCandidate products to evaluate:")
    for i, c in enumerate(candidates, 1):
        lines.append(f"\n{i}. [ID={c.get('product_id', 0)}] {c.get('name', 'Unknown')}")
        if c.get("product_type"):
            lines.append(f"   Type: {c['product_type']}")
    return "\n".join(lines)


def parse_results(raw: str) -> list[dict]:
    """Parse a verification response; tolerates fences, returns [] on bad/truncated JSON."""
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(text)
    except (json.JSONDecodeError, IndexError):
        log.error("failed to parse verification response: %.160s", raw)
        return []
    if isinstance(data, dict):
        for k in ("results", "candidates", "items"):
            if isinstance(data.get(k), list):
                return data[k]
        return []
    return data if isinstance(data, list) else []


class RealLLM:
    """Calls the OpenAI-compatible chat completions endpoint."""

    def __init__(self, cfg, client) -> None:
        self.cfg = cfg
        self.client = client

    async def verify(
        self, source: dict, candidates: list[dict]
    ) -> tuple[list[dict], int, int]:
        kwargs = {
            "model": self.cfg.PRIMARY_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(source, candidates)},
            ],
            "max_completion_tokens": self.cfg.MAX_TOKENS,
            "response_format": RESPONSE_FORMAT,
        }
        if self.cfg.REASONING_EFFORT:
            kwargs["reasoning_effort"] = self.cfg.REASONING_EFFORT
        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # network / API error -> retryable
            raise LLMError(str(exc)) from exc
        if not getattr(resp, "choices", None):
            raise LLMError("no choices in response")
        raw = resp.choices[0].message.content or ""
        usage = resp.usage
        ti = usage.prompt_tokens if usage else 0
        to = usage.completion_tokens if usage else 0
        return parse_results(raw), ti, to

    async def generate(
        self, system_prompt: str, user_prompt: str, response_format: dict
    ) -> tuple[dict, int, int]:
        """Generic generation call for rule-gen, judge, and other structured tasks.

        Uses the same OpenAI chat completions endpoint as verify(), but accepts
        arbitrary system/user prompts and a response_format schema.

        Args:
            system_prompt: System role message content.
            user_prompt: User role message content.
            response_format: OpenAI response_format dict (json_schema).

        Returns:
            Tuple of (parsed_dict, tokens_in, tokens_out).

        Raises:
            LLMError: On provider errors or empty response.
        """
        kwargs = {
            "model": self.cfg.PRIMARY_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": self.cfg.MAX_TOKENS,
            "response_format": response_format,
        }
        if self.cfg.REASONING_EFFORT:
            kwargs["reasoning_effort"] = self.cfg.REASONING_EFFORT
        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(str(exc)) from exc
        if not getattr(resp, "choices", None):
            raise LLMError("no choices in generate response")
        raw = resp.choices[0].message.content or ""
        usage = resp.usage
        ti = usage.prompt_tokens if usage else 0
        to = usage.completion_tokens if usage else 0
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            log.error("generate: failed to parse JSON response: %.160s", raw)
            parsed = {}
        return parsed, ti, to


class MockLLM:
    """No network. Behaviour controlled by scenario (valid|slow|error|truncated)."""

    def __init__(self, scenario: str = "valid", slow_seconds: float = 5.0) -> None:
        self.scenario = scenario
        self.slow_seconds = slow_seconds

    def _valid_json(self, candidates: list[dict]) -> str:
        results = []
        for idx, c in enumerate(candidates):
            results.append(
                {
                    "candidate_id": c.get("product_id", 0),
                    "verdict": idx % 2 == 0,  # deterministic mix
                    "logical_score": 0.9 if idx % 2 == 0 else 0.2,
                    "context_code": "accessory",
                    "rules_evaluated": 1,
                    "rules_passed": 1 if idx % 2 == 0 else 0,
                    "rules_failed": 0 if idx % 2 == 0 else 1,
                    "rules_undefined": 0,
                }
            )
        return json.dumps({"results": results})

    async def verify(
        self, source: dict, candidates: list[dict]
    ) -> tuple[list[dict], int, int]:
        if self.scenario == "error":
            raise LLMError("simulated provider error (mock)")
        if self.scenario == "slow":
            await asyncio.sleep(self.slow_seconds)
            return parse_results(self._valid_json(candidates)), 10, 10
        if self.scenario == "truncated":
            raw = self._valid_json(candidates)[
                : max(20, len(candidates) * 3)
            ]  # cut mid-JSON
            return parse_results(raw), 10, 5  # parse_results -> [] (handled, no crash)
        return parse_results(self._valid_json(candidates)), 10, 10  # valid

    async def generate(
        self, system_prompt: str, user_prompt: str, response_format: dict
    ) -> tuple[dict, int, int]:
        """Mock generate for rule-gen and other structured tasks.

        In the 'valid' scenario, delegates to MockRuleGen to return realistic
        rule sets for known fixture type-pairs.  In 'error' it raises LLMError.
        In 'truncated'/'slow' it returns an empty rules dict (handled by callers).

        Args:
            system_prompt: Ignored in mock mode.
            user_prompt: Parsed to extract type_a/type_b for MockRuleGen.
            response_format: Ignored in mock mode.

        Returns:
            Tuple of (dict, tokens_in, tokens_out).

        Raises:
            LLMError: In the 'error' scenario.
        """
        if self.scenario == "error":
            raise LLMError("simulated provider error (mock generate)")
        if self.scenario in ("truncated", "slow"):
            return {"rules": []}, 10, 5
        # valid - delegate to MockRuleGen for realistic rules
        from app.services.compatibility.ontology import MockRuleGen

        gen = MockRuleGen()
        return await gen.generate(system_prompt, user_prompt, response_format)


def get_llm(cfg, client=None):
    """Factory: MockLLM in mock mode, RealLLM otherwise."""
    if cfg.LLM_MODE == "mock":
        return MockLLM(scenario=cfg.MOCK_SCENARIO)
    return RealLLM(cfg, client)


# Aliases for the Batch-API path (compatibility with the ported batch_run module).
build_verification_prompt = build_user_prompt
parse_verification_response = parse_results
