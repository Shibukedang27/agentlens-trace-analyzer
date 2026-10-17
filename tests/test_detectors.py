from __future__ import annotations

import pytest

from agentlens.detectors import (
    DetectorConfig,
    IdleRunDetector,
    OversizedModelDetector,
    PromptBloatDetector,
    RetryStormDetector,
    ToolLoopDetector,
    run_detectors,
)

CFG = DetectorConfig()


def test_tool_loop_detected_with_savings(builder, prices) -> None:
    b = builder()
    for _ in range(4):
        b.llm(inp=1000, out=50).tool("get_order", {"id": 7})
    run = b.build()
    [finding] = ToolLoopDetector().detect(run, prices, CFG)
    assert finding.detector == "tool_loop"
    assert "4x" in finding.title
    # 3 redundant tool calls -> the 3 LLM calls that decided them are wasted
    expected = 3 * prices.cost("orion-large", 1000, 50)
    assert finding.est_savings == pytest.approx(expected)
    assert finding.wasted_tokens == 3 * 1050


def test_tool_loop_ignores_different_args(builder, prices) -> None:
    b = builder()
    for i in range(5):
        b.llm().tool("get_order", {"id": i})
    assert ToolLoopDetector().detect(b.build(), prices, CFG) == []


def test_tool_loop_ignores_failed_repeats(builder, prices) -> None:
    b = builder()
    for _ in range(4):
        b.tool("get_order", {"id": 1}, status="error")
    assert ToolLoopDetector().detect(b.build(), prices, CFG) == []


def test_tool_loop_args_are_order_insensitive(builder, prices) -> None:
    b = (
        builder()
        .tool("t", {"a": 1, "b": 2})
        .tool("t", {"b": 2, "a": 1})
        .tool("t", {"a": 1, "b": 2})
    )
    assert len(ToolLoopDetector().detect(b.build(), prices, CFG)) == 1


def test_retry_storm_detected(builder, prices) -> None:
    b = builder().llm()
    for _ in range(4):
        b.tool("fetch", {"u": 1}, status="error", error_type="TimeoutError")
    [finding] = RetryStormDetector().detect(b.build(), prices, CFG)
    assert "4 consecutive failures" in finding.title
    assert "TimeoutError" in finding.evidence
    assert "no backoff" in finding.evidence


def test_retry_streak_resets_on_success(builder, prices) -> None:
    b = builder()
    for _ in range(3):
        b.tool("fetch", status="error").tool("fetch", status="error").tool("fetch")
    assert RetryStormDetector().detect(b.build(), prices, CFG) == []


def test_retry_storm_on_llm_rate_limits(builder, prices) -> None:
    b = builder()
    for _ in range(3):
        b.llm(inp=0, out=0, status="error", error_type="RateLimitError")
    b.llm()
    [finding] = RetryStormDetector().detect(b.build(), prices, CFG)
    assert finding.title.endswith("llm:orion-large")
    assert finding.est_savings == 0.0  # rate-limited calls are not billed


def test_idle_run_by_low_output(builder, prices) -> None:
    run = builder().llm(model="vega-pro", inp=3000, out=12).build()
    [finding] = IdleRunDetector().detect(run, prices, CFG)
    assert finding.est_savings == pytest.approx(prices.cost("vega-pro", 3000, 12))


def test_idle_run_by_marker_text(builder, prices) -> None:
    run = builder().llm(inp=2000, out=80, output_text="Nothing happened. HEARTBEAT_OK").build()
    [finding] = IdleRunDetector().detect(run, prices, CFG)
    assert "nothing happened" in finding.evidence


@pytest.mark.parametrize(
    "make",
    [
        lambda b: b.llm(inp=2000, out=10).tool("act"),  # did something
        lambda b: b.llm(inp=100, out=10),  # too small to matter
        lambda b: b.llm(inp=2000, out=300),  # substantive answer
    ],
)
def test_idle_run_negatives(builder, prices, make) -> None:
    assert IdleRunDetector().detect(make(builder()).build(), prices, CFG) == []


def test_prompt_bloat_detected(builder, prices) -> None:
    b = builder()
    for inp in (1000, 2500, 4000, 6000, 9000):
        b.llm(inp=inp).tool("search", {"i": inp})
    [finding] = PromptBloatDetector().detect(b.build(), prices, CFG)
    assert "9.0x" in finding.title
    expected_tokens = sum(int((x - 1000) * 0.5) for x in (2500, 4000, 6000, 9000))
    assert finding.wasted_tokens == expected_tokens


def test_prompt_bloat_ignores_modest_growth_and_short_runs(builder, prices) -> None:
    modest = builder()
    for inp in (2000, 2300, 2600, 2900, 3200):
        modest.llm(inp=inp)
    short = builder().llm(inp=1000).llm(inp=9000)
    assert PromptBloatDetector().detect(modest.build(), prices, CFG) == []
    assert PromptBloatDetector().detect(short.build(), prices, CFG) == []


def test_prompt_bloat_ignores_small_side_prompts(builder, prices) -> None:
    # a tiny routing call must not be the baseline for the growth ratio
    b = builder().llm(model="orion-mini", inp=200, out=5)
    for inp in (2000, 2200, 2400, 2600):
        b.llm(inp=inp)
    assert PromptBloatDetector().detect(b.build(), prices, CFG) == []


def test_oversized_model_detected(builder, prices) -> None:
    b = builder().llm(inp=300, out=5).llm(inp=400, out=10).llm(inp=3000, out=300)
    [finding] = OversizedModelDetector().detect(b.build(), prices, CFG)
    big = prices.cost("orion-large", 300, 5) + prices.cost("orion-large", 400, 10)
    small = prices.cost("orion-mini", 300, 5) + prices.cost("orion-mini", 400, 10)
    assert finding.est_savings == pytest.approx(big - small)
    assert "orion-mini" in finding.evidence


def test_oversized_model_negatives(builder, prices) -> None:
    small_model = (
        builder().llm(model="orion-mini", inp=300, out=5).llm(model="orion-mini", inp=300, out=5)
    )
    single = builder().llm(inp=300, out=5).llm(inp=3000, out=300)
    assert OversizedModelDetector().detect(small_model.build(), prices, CFG) == []
    assert OversizedModelDetector().detect(single.build(), prices, CFG) == []


def test_config_overrides_change_thresholds(builder, prices) -> None:
    b = builder()
    for _ in range(3):
        b.llm().tool("x", {"a": 1})
    run = b.build()
    assert len(run_detectors([run], prices)) == 1
    strict = DetectorConfig.from_dict({"loop_min_repeats": 4})
    assert run_detectors([run], prices, strict) == []


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown detector settings"):
        DetectorConfig.from_dict({"not_a_setting": 1})


def test_clean_run_has_no_findings(builder, prices) -> None:
    b = builder().llm(model="orion-mini", inp=300, out=8)
    for i, inp in enumerate((2000, 2300, 2600)):
        b.llm(inp=inp, out=120).tool("search_kb", {"q": i})
    b.llm(inp=2900, out=250)
    assert run_detectors([b.build()], prices) == []
