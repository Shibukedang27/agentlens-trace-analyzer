from __future__ import annotations

from agentlens.analyzer import analyze_runs
from agentlens.anomaly import anomaly_scores
from agentlens.evaluate import evaluate_detection, measure_throughput, prf, to_markdown
from agentlens.generator import ANOMALY_TYPES, GeneratorConfig, generate, load_labels, write_dataset
from agentlens.parser import group_runs, load_runs, parse_span


def test_generator_is_deterministic() -> None:
    a = generate(GeneratorConfig(n_runs=50, seed=3))
    b = generate(GeneratorConfig(n_runs=50, seed=3))
    c = generate(GeneratorConfig(n_runs=50, seed=4))
    assert a == b
    assert a != c


def test_labels_match_traces_and_types() -> None:
    records, labels = generate(GeneratorConfig(n_runs=200, seed=1))
    trace_ids = {r["trace_id"] for r in records}
    assert set(labels) == trace_ids
    assert all(set(v) <= set(ANOMALY_TYPES) for v in labels.values())
    assert sum(1 for v in labels.values() if v) > 20


def test_otel_style_parses_to_same_metrics(prices) -> None:
    simple, _ = generate(GeneratorConfig(n_runs=40, seed=9))
    otel, _ = generate(GeneratorConfig(n_runs=40, seed=9, style="otel"))
    assert "attributes" in otel[1]
    r1 = analyze_runs(group_runs(parse_span(r) for r in simple), prices=prices, anomaly=False)
    r2 = analyze_runs(group_runs(parse_span(r) for r in otel), prices=prices, anomaly=False)
    assert r1.summary["total_cost"] == r2.summary["total_cost"]
    assert len(r1.findings) == len(r2.findings)


def test_write_and_reload_dataset(tmp_path) -> None:
    records, labels = generate(GeneratorConfig(n_runs=20, seed=2))
    write_dataset(records, labels, tmp_path / "t.jsonl", tmp_path / "l.json", meta={"seed": 2})
    runs, errors = load_runs(tmp_path / "t.jsonl")
    assert len(runs) == 20 and errors == []
    assert load_labels(tmp_path / "l.json") == labels


def test_prf() -> None:
    m = prf({"a", "b", "c"}, {"b", "c", "d", "e"})
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 2)
    assert m["precision"] == round(2 / 3, 4) and m["recall"] == 0.5
    assert prf(set(), set())["f1"] == 0.0


def test_evaluate_detection_sanity() -> None:
    res = evaluate_detection(n_runs=300, seed=11)
    assert set(res["detectors"]) == set(ANOMALY_TYPES)
    # loose floor: detectors must be far better than chance on their own labels
    for name, m in res["detectors"].items():
        assert m["f1"] > 0.6, name
    assert 0.5 < res["isolation_forest"]["roc_auc"] <= 1.0


def test_throughput_and_markdown() -> None:
    tp = measure_throughput(n_runs=50, repeats=1)
    assert tp["spans_per_sec_full"] > 0
    md = to_markdown(
        {"detection": evaluate_detection(n_runs=100, seed=5), "robustness": {}, "throughput": tp}
    )
    assert "| tool_loop |" in md and "Spans/sec" in md


def test_anomaly_scores_range_and_small_input(prices) -> None:
    records, _ = generate(GeneratorConfig(n_runs=120, seed=5))
    result = analyze_runs(group_runs(parse_span(r) for r in records), prices=prices)
    score = result.run_df["anomaly_score"]
    assert score.between(0, 1).all() and score.max() == 1.0
    tiny = result.run_df.head(5)
    s, flags = anomaly_scores(tiny)
    assert (s == 0).all() and not flags.any()
