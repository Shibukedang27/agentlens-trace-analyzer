from __future__ import annotations

import json

from click.testing import CliRunner

from agentlens.analyzer import analyze_file
from agentlens.cli import main
from agentlens.report import to_html, to_markdown


def _gen(tmp_path, runs: int = 80):
    traces = tmp_path / "t.jsonl"
    labels = tmp_path / "l.json"
    res = CliRunner().invoke(
        main,
        [
            "generate",
            "--runs",
            str(runs),
            "--seed",
            "7",
            "--out",
            str(traces),
            "--labels",
            str(labels),
        ],
    )
    assert res.exit_code == 0, res.output
    return traces, labels


def test_cli_version() -> None:
    res = CliRunner().invoke(main, ["--version"])
    assert res.exit_code == 0 and "agentlens" in res.output


def test_cli_generate_and_analyze_writes_reports(tmp_path) -> None:
    traces, labels = _gen(tmp_path)
    assert json.loads(labels.read_text())["meta"]["synthetic"] is True
    md, html, js, csv = (tmp_path / n for n in ("r.md", "r.html", "r.json", "f.csv"))
    res = CliRunner().invoke(
        main,
        [
            "analyze",
            str(traces),
            "--report",
            str(md),
            "--html",
            str(html),
            "--json",
            str(js),
            "--findings-csv",
            str(csv),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Waste by detector" in res.output
    assert "## Waste by detector" in md.read_text()
    assert "<svg" in html.read_text()
    data = json.loads(js.read_text())
    assert data["summary"]["runs"] == 80 and "disclaimer" in data
    assert csv.read_text().startswith("detector,trace_id")


def test_cli_fail_on_waste_gate(tmp_path) -> None:
    traces, _ = _gen(tmp_path)
    res = CliRunner().invoke(main, ["analyze", str(traces), "--no-anomaly", "--fail-on-waste", "0"])
    assert res.exit_code == 2


def test_cli_custom_prices_and_bad_prices(tmp_path) -> None:
    traces, _ = _gen(tmp_path, runs=30)
    good = tmp_path / "p.yaml"
    good.write_text("currency: EUR\nmodels:\n  orion-large: {input: 1, output: 1}\n")
    res = CliRunner().invoke(main, ["analyze", str(traces), "--prices", str(good), "--top", "0"])
    assert res.exit_code == 0 and "Unknown models" in res.output
    bad = tmp_path / "bad.yaml"
    bad.write_text("nothing: here\n")
    res = CliRunner().invoke(main, ["analyze", str(traces), "--prices", str(bad)])
    assert res.exit_code != 0 and "could not load prices" in res.output


def test_cli_rejects_file_without_spans(tmp_path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("{bad json\n")
    res = CliRunner().invoke(main, ["analyze", str(empty)])
    assert res.exit_code != 0 and "no valid spans" in res.output
    res = CliRunner().invoke(main, ["analyze", str(empty), "--strict"])
    assert res.exit_code != 0 and "line 1" in res.output


def test_cli_trace_tree(tmp_path) -> None:
    traces, _ = _gen(tmp_path, runs=10)
    trace_id = json.loads(traces.read_text().splitlines()[0])["trace_id"]
    res = CliRunner().invoke(main, ["trace", str(traces), trace_id[:8]])
    assert res.exit_code == 0 and trace_id in res.output
    res = CliRunner().invoke(main, ["trace", str(traces), "nope"])
    assert res.exit_code != 0


def test_cli_eval_writes_results(tmp_path) -> None:
    out = tmp_path / "results"
    res = CliRunner().invoke(
        main,
        [
            "eval",
            "--runs",
            "150",
            "--robustness-seeds",
            "1,2",
            "--throughput-runs",
            "40",
            "--out",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads((out / "eval_results.json").read_text())
    assert data["detection"]["n_runs"] == 150
    assert set(data["robustness"]["seeds"]) == {1, 2}
    assert "Throughput" in (out / "eval_results.md").read_text()


def test_reports_mention_synthetic_disclaimer(tmp_path) -> None:
    traces, _ = _gen(tmp_path, runs=40)
    result = analyze_file(traces)
    assert "Synthetic traces, example prices" in to_markdown(result)
    assert "Synthetic traces, example prices" in to_html(result)
