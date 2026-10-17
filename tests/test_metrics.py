from __future__ import annotations

import pytest

from agentlens.analyzer import analyze_runs
from agentlens.metrics import (
    agent_frame,
    input_growth_ratio,
    max_tool_repeat,
    percentile,
    run_frame,
    span_frame,
    summarize,
)


def _two_runs(builder):
    a = builder(agent="alpha", trace_id="a").llm(inp=1000, out=100).tool("s", status="error")
    a.llm(inp=1500, out=50)
    b = builder(agent="beta", trace_id="b").llm(model="orion-mini", inp=500, out=20)
    return [a.build(), b.build()]


def test_percentile_basics() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0
    assert percentile([0, 100], 95) == pytest.approx(95.0)


def test_run_frame_values(builder, prices) -> None:
    df = run_frame(_two_runs(builder), prices).set_index("trace_id")
    a = df.loc["a"]
    assert a["n_llm_calls"] == 2 and a["n_tool_calls"] == 1
    assert a["input_tokens"] == 2500 and a["output_tokens"] == 150
    assert a["cost"] == pytest.approx(prices.cost("orion-large", 2500, 150))
    assert a["error_rate"] == pytest.approx(1 / 3)
    assert df.loc["b"]["cost"] == pytest.approx(prices.cost("orion-mini", 500, 20))


def test_agent_frame_sorted_by_cost(builder, prices) -> None:
    runs = _two_runs(builder)
    agents = agent_frame(run_frame(runs, prices), span_frame(runs, prices))
    assert list(agents["agent"]) == ["alpha", "beta"]
    assert agents.loc[0, "runs"] == 1
    assert agents.loc[0, "p95_llm_ms"] > 0


def test_summary_kpis(builder, prices) -> None:
    runs = _two_runs(builder)
    summary = summarize(run_frame(runs, prices), span_frame(runs, prices))
    assert summary["runs"] == 2 and summary["agents"] == 2
    assert summary["llm_calls"] == 3 and summary["tool_calls"] == 1
    assert summary["error_rate"] == pytest.approx(1 / 4)


def test_repeat_and_growth_features(builder) -> None:
    b = builder().tool("x", {"a": 1}).tool("x", {"a": 1}).tool("x", {"a": 2})
    for inp in (1000, 2000, 3000):
        b.llm(inp=inp)
    run = b.build()
    assert max_tool_repeat(run) == 2
    assert input_growth_ratio(run) == pytest.approx(3.0)


def test_savings_capped_at_run_cost(builder, prices) -> None:
    # tool loop + idle-like low output in the same run: total savings can't exceed cost
    b = builder()
    for inp in (1000, 3000, 6000, 9000):
        b.llm(inp=inp, out=5).tool("t", {"a": 1})
    result = analyze_runs([b.build()], prices=prices, anomaly=False)
    row = result.run_df.iloc[0]
    assert row["est_savings"] <= row["cost"] + 1e-12
    assert set(row["flags"].split(",")) >= {"tool_loop", "prompt_bloat"}


def test_analyze_empty_input(prices) -> None:
    result = analyze_runs([], prices=prices)
    assert result.summary["runs"] == 0
    assert result.findings_df().empty
