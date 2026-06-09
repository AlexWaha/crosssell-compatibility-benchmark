"""Unit tests for the rulegen-collect parse+validate+write path.

Tests the pure _parse_rulegen_line helper and the overall collect logic using a
mock metrics repo. No I/O against real OpenAI or MySQL - $0, no network calls.

Run: docker exec avtc_engine python -m pytest tests/test_rulegen_collect.py -v
"""

from __future__ import annotations

import json

import pytest

from batch_run import _parse_rulegen_line


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jsonl_line(custom_id: str, rules: list[dict]) -> str:
    """Build a synthetic batch output JSONL line matching the OpenAI Batch API shape."""
    content = json.dumps({"rules": rules}, ensure_ascii=False)
    obj = {
        "custom_id": custom_id,
        "response": {"body": {"choices": [{"message": {"content": content}}]}},
    }
    return json.dumps(obj, ensure_ascii=False)


def _valid_rule(idx: int = 1) -> dict:
    return {
        "id": f"rule_{idx}",
        "type": "exact_match",
        "attribute_a": "brand",
        "attribute_b": "brand",
        "weight": 0.9,
        "description": f"Brand compatibility rule {idx}",
    }


def _valid_rules(n: int = 2) -> list[dict]:
    return [_valid_rule(i) for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# _parse_rulegen_line: correct parsing
# ---------------------------------------------------------------------------


def test_parse_rulegen_line_valid():
    """Parses a well-formed line and returns (type_a, type_b, raw_rules)."""
    rules = _valid_rules(2)
    line = _make_jsonl_line("rulegen_laptop__charger", rules)
    result = _parse_rulegen_line(line)
    assert result is not None
    type_a, type_b, raw_rules = result
    assert type_a == "laptop"
    assert type_b == "charger"
    assert raw_rules == rules


def test_parse_rulegen_line_type_with_underscores():
    """type_a containing underscores is handled (split on first __, not all __)."""
    rules = _valid_rules(2)
    # custom_id: rulegen_coffee_machine__descaler
    # after stripping prefix: coffee_machine__descaler
    # split on first __: type_a=coffee_machine, type_b=descaler
    line = _make_jsonl_line("rulegen_coffee_machine__descaler", rules)
    result = _parse_rulegen_line(line)
    assert result is not None
    type_a, type_b, _ = result
    assert type_a == "coffee_machine"
    assert type_b == "descaler"


def test_parse_rulegen_line_type_b_with_underscores():
    """type_b containing underscores is preserved after the first __ separator."""
    rules = _valid_rules(2)
    line = _make_jsonl_line("rulegen_laptop__power_adapter", rules)
    result = _parse_rulegen_line(line)
    assert result is not None
    type_a, type_b, _ = result
    assert type_a == "laptop"
    assert type_b == "power_adapter"


def test_parse_rulegen_line_content_with_markdown_fence():
    """Content wrapped in markdown fences is stripped before JSON parse."""
    rules = _valid_rules(2)
    fenced_content = "```json\n" + json.dumps({"rules": rules}) + "\n```"
    obj = {
        "custom_id": "rulegen_laptop__charger",
        "response": {"body": {"choices": [{"message": {"content": fenced_content}}]}},
    }
    line = json.dumps(obj)
    result = _parse_rulegen_line(line)
    assert result is not None
    _, _, raw_rules = result
    assert raw_rules == rules


# ---------------------------------------------------------------------------
# _parse_rulegen_line: failure cases return None
# ---------------------------------------------------------------------------


def test_parse_rulegen_line_empty_string_returns_none():
    assert _parse_rulegen_line("") is None


def test_parse_rulegen_line_whitespace_only_returns_none():
    assert _parse_rulegen_line("   \n") is None


def test_parse_rulegen_line_invalid_json_returns_none():
    assert _parse_rulegen_line("{not valid json}") is None


def test_parse_rulegen_line_wrong_prefix_returns_none():
    """Lines with custom_id not starting with 'rulegen_' are ignored."""
    line = _make_jsonl_line("compat_p123b0", _valid_rules(2))
    assert _parse_rulegen_line(line) is None


def test_parse_rulegen_line_missing_double_underscore_returns_none():
    """custom_id with no __ separator after stripping prefix returns None."""
    line = _make_jsonl_line("rulegen_laptop_charger", _valid_rules(2))
    assert _parse_rulegen_line(line) is None


def test_parse_rulegen_line_no_choices_returns_none():
    """Response with empty choices list returns None."""
    obj = {
        "custom_id": "rulegen_laptop__charger",
        "response": {"body": {"choices": []}},
    }
    assert _parse_rulegen_line(json.dumps(obj)) is None


def test_parse_rulegen_line_bad_content_json_returns_none():
    """Bad JSON in message content returns None."""
    obj = {
        "custom_id": "rulegen_laptop__charger",
        "response": {
            "body": {"choices": [{"message": {"content": "not-json-at-all"}}]}
        },
    }
    assert _parse_rulegen_line(json.dumps(obj)) is None


# ---------------------------------------------------------------------------
# parse + validate + write path using mock metrics repo
# ---------------------------------------------------------------------------


class MockMetricsRepo:
    """In-memory mock for MetricsRepository.set_rule_cache calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[dict], str, str]] = []

    async def set_rule_cache(
        self,
        type_a: str,
        type_b: str,
        rules: list[dict],
        generated_by: str = "llm",
        source_hash: str = "",
    ) -> None:
        self.calls.append((type_a, type_b, rules, generated_by, source_hash))


async def _run_collect_loop(lines: list[str], repo: MockMetricsRepo) -> dict:
    """Run the parse-validate-write loop against a list of synthetic JSONL lines.

    Mirrors the inner loop in cmd_rulegen_collect without any I/O.
    Returns summary dict: {written, skipped}.
    """
    from app.services.compatibility.ontology import _validate_rules

    written = 0
    skipped = 0
    for line in lines:
        parsed = _parse_rulegen_line(line)
        if parsed is None:
            continue
        type_a, type_b, raw_rules = parsed
        try:
            rules = _validate_rules(raw_rules)
        except ValueError:
            skipped += 1
            continue
        await repo.set_rule_cache(
            type_a, type_b, rules, generated_by="batch_rulegen", source_hash=""
        )
        written += 1
    return {"written": written, "skipped": skipped}


def test_collect_loop_writes_valid_pairs(event_loop):
    """Two valid lines produce two set_rule_cache calls with correct types."""
    repo = MockMetricsRepo()
    lines = [
        _make_jsonl_line("rulegen_laptop__charger", _valid_rules(2)),
        _make_jsonl_line("rulegen_coffee_machine__descaler", _valid_rules(3)),
    ]
    result = event_loop.run_until_complete(_run_collect_loop(lines, repo))
    assert result["written"] == 2
    assert result["skipped"] == 0
    pairs = {(c[0], c[1]) for c in repo.calls}
    assert ("laptop", "charger") in pairs
    assert ("coffee_machine", "descaler") in pairs
    # generated_by is always batch_rulegen
    assert all(c[3] == "batch_rulegen" for c in repo.calls)


def test_collect_loop_drops_malformed_rules(event_loop):
    """A line with rules that fail _validate_rules is skipped, not written."""
    repo = MockMetricsRepo()
    malformed_rules = [
        # only 1 rule with an invalid type -> _validate_rules raises ValueError
        {
            "id": "rule_1",
            "type": "unknown_type",
            "attribute_a": "brand",
            "attribute_b": "brand",
            "weight": 0.5,
            "description": "bad rule",
        }
    ]
    lines = [
        _make_jsonl_line("rulegen_laptop__charger", _valid_rules(2)),  # valid
        _make_jsonl_line("rulegen_tv__remote", malformed_rules),  # invalid
    ]
    result = event_loop.run_until_complete(_run_collect_loop(lines, repo))
    assert result["written"] == 1
    assert result["skipped"] == 1
    # Only the valid pair is written
    assert len(repo.calls) == 1
    assert repo.calls[0][0] == "laptop"
    assert repo.calls[0][1] == "charger"


def test_collect_loop_skips_parse_failures(event_loop):
    """Lines with wrong prefix or bad JSON are silently ignored."""
    repo = MockMetricsRepo()
    lines = [
        _make_jsonl_line("rulegen_laptop__charger", _valid_rules(2)),  # valid
        "{not-json}",  # bad JSON
        _make_jsonl_line("compat_p1b0", _valid_rules(2)),  # wrong prefix
        "",  # empty
    ]
    result = event_loop.run_until_complete(_run_collect_loop(lines, repo))
    assert result["written"] == 1
    assert result["skipped"] == 0  # parse failures don't count as skipped
    assert len(repo.calls) == 1


def test_collect_loop_idempotent(event_loop):
    """Running the same lines twice produces two write calls per pair (upsert in real repo)."""
    repo = MockMetricsRepo()
    lines = [_make_jsonl_line("rulegen_laptop__charger", _valid_rules(2))]
    event_loop.run_until_complete(_run_collect_loop(lines, repo))
    event_loop.run_until_complete(_run_collect_loop(lines, repo))
    # Two calls to set_rule_cache with same args - real repo upserts, so safe
    assert len(repo.calls) == 2
    assert repo.calls[0][:2] == repo.calls[1][:2]


def test_collect_loop_rule_count_clamped_to_max(event_loop):
    """Rules beyond _MAX_RULES=8 are truncated by _validate_rules before write."""
    repo = MockMetricsRepo()
    # 10 rules - _validate_rules truncates to 8
    many_rules = _valid_rules(10)
    lines = [_make_jsonl_line("rulegen_laptop__charger", many_rules)]
    result = event_loop.run_until_complete(_run_collect_loop(lines, repo))
    assert result["written"] == 1
    written_rules = repo.calls[0][2]
    assert len(written_rules) <= 8


@pytest.fixture
def event_loop():
    """Provide a fresh event loop for each async test."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
