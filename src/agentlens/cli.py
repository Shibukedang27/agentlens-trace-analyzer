"""Command line interface: ``agentlens generate | analyze | trace | eval``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from agentlens import __version__
from agentlens.analyzer import AnalysisResult, analyze_file
from agentlens.detectors import DetectorConfig
from agentlens.generator import GeneratorConfig, generate, write_dataset
from agentlens.metrics import span_cost
from agentlens.parser import TraceParseError, load_runs
from agentlens.pricing import PriceConfigError, PriceTable
from agentlens.report import DISCLAIMER, to_html, to_json, to_markdown

console = Console()


def _load_prices(path: str | None) -> PriceTable:
    try:
        return PriceTable.from_yaml(path) if path else PriceTable.default_table()
    except (OSError, PriceConfigError) as exc:
        raise click.ClickException(f"could not load prices: {exc}") from exc


def _print_result(result: AnalysisResult, top: int) -> None:
    s = result.summary
    cur = "$" if result.currency == "USD" else result.currency + " "
    console.print(
        f"[bold red]agentlens[/] · {s['runs']:,} runs · {s['spans']:,} spans · "
        f"{s['agents']} agents   [dim]{DISCLAIMER}[/]"
    )
    kpi = Table(show_header=False, box=None, padding=(0, 2))
    kpi.add_row(
        "Total cost",
        f"[bold]{cur}{s['total_cost']:,.2f}[/]",
        "Est. savings",
        f"[bold yellow]{cur}{s['est_savings']:,.2f} ({s['savings_pct']:.1f}%)[/]",
    )
    kpi.add_row(
        "Total tokens",
        f"{s['total_tokens']:,}",
        "Flagged runs",
        f"{s['flagged_runs']:,} ({s['findings']:,} findings)",
    )
    kpi.add_row(
        "LLM p50/p95",
        f"{s['p50_llm_ms']:.0f} / {s['p95_llm_ms']:.0f} ms",
        "Span error rate",
        f"{s['error_rate'] * 100:.1f}%",
    )
    console.print(kpi)

    det = Table(title="Waste by detector", title_style="bold yellow", header_style="bold red")
    for col in ("Detector", "Findings", "Runs", "Wasted tokens", "Est. savings"):
        det.add_column(col, justify="left" if col == "Detector" else "right")
    for r in result.savings_by_detector().to_dict(orient="records"):
        det.add_row(
            r["label"],
            str(r["findings"]),
            str(r["runs"]),
            f"{r['wasted_tokens']:,}",
            f"{cur}{r['est_savings']:,.2f}",
        )
    console.print(det)

    agents = Table(title="Cost by agent", title_style="bold yellow", header_style="bold red")
    for col in ("Agent", "Runs", "Cost", "Avg/run", "p95 run", "Errors"):
        agents.add_column(col, justify="left" if col == "Agent" else "right")
    for r in result.agent_df.to_dict(orient="records"):
        agents.add_row(
            r["agent"],
            str(r["runs"]),
            f"{cur}{r['total_cost']:,.2f}",
            f"{cur}{r['avg_cost_per_run']:.4f}",
            f"{r['p95_run_ms'] / 1000:.1f}s",
            f"{r['error_rate'] * 100:.1f}%",
        )
    console.print(agents)

    if top and result.findings:
        tops = sorted(result.findings, key=lambda f: f.est_savings, reverse=True)[:top]
        ft = Table(
            title=f"Top {len(tops)} findings", title_style="bold yellow", header_style="bold red"
        )
        for col in ("Detector", "Agent", "Trace", "Finding", "Savings"):
            ft.add_column(col, justify="right" if col == "Savings" else "left")
        for f in tops:
            ft.add_row(f.detector, f.agent, f.trace_id[:10], f.title, f"{cur}{f.est_savings:.4f}")
        console.print(ft)
    if result.parse_errors:
        console.print(
            f"[yellow]{len(result.parse_errors)} line(s) skipped; first: "
            f"{result.parse_errors[0]}[/]"
        )
    if result.unknown_models:
        console.print(
            f"[yellow]Unknown models priced at default rate: {', '.join(result.unknown_models)}[/]"
        )


@click.group()
@click.version_option(__version__, prog_name="agentlens")
def main() -> None:
    """agentlens: find the waste (cost, loops, retries, idle runs) in LLM agent traces."""


@main.command("generate")
@click.option("--runs", "n_runs", default=600, show_default=True, help="Number of runs.")
@click.option("--seed", default=42, show_default=True)
@click.option("--out", default="data/demo_traces.jsonl", show_default=True, type=click.Path())
@click.option(
    "--labels", "labels_path", default="data/demo_labels.json", show_default=True, type=click.Path()
)
@click.option(
    "--style",
    type=click.Choice(["simple", "otel"]),
    default="simple",
    show_default=True,
    help="Field names: simple or OpenTelemetry GenAI attributes.",
)
def generate_cmd(n_runs: int, seed: int, out: str, labels_path: str, style: str) -> None:
    """Generate a synthetic trace dataset with labeled, injected anomalies."""
    records, labels = generate(GeneratorConfig(n_runs=n_runs, seed=seed, style=style))
    write_dataset(
        records,
        labels,
        out,
        labels_path,
        meta={"seed": seed, "n_runs": n_runs, "style": style, "synthetic": True},
    )
    anomalous = sum(1 for v in labels.values() if v)
    console.print(
        f"Wrote [bold]{len(records):,}[/] spans / {n_runs:,} runs to {out} "
        f"({anomalous} runs with injected anomalies; labels in {labels_path})"
    )


@main.command("analyze")
@click.argument("traces", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--prices",
    type=click.Path(exists=True, dir_okay=False),
    help="Price table YAML (default: bundled EXAMPLE prices).",
)
@click.option("--report", "report_md", type=click.Path(), help="Write a Markdown report.")
@click.option(
    "--html", "report_html", type=click.Path(), help="Write a self-contained HTML report."
)
@click.option("--json", "report_json", type=click.Path(), help="Write full results as JSON.")
@click.option("--findings-csv", type=click.Path(), help="Write all findings as CSV.")
@click.option("--top", default=10, show_default=True, help="Findings to print.")
@click.option("--no-anomaly", is_flag=True, help="Skip the IsolationForest score.")
@click.option("--strict", is_flag=True, help="Fail on the first malformed line.")
@click.option(
    "--fail-on-waste",
    type=float,
    default=None,
    help="Exit 2 if estimated savings exceed this % of cost (for CI gates).",
)
def analyze_cmd(
    traces: str,
    prices: str | None,
    report_md: str | None,
    report_html: str | None,
    report_json: str | None,
    findings_csv: str | None,
    top: int,
    no_anomaly: bool,
    strict: bool,
    fail_on_waste: float | None,
) -> None:
    """Analyze a JSONL trace file: metrics, waste findings and estimated savings."""
    table = _load_prices(prices)
    try:
        config = DetectorConfig.from_dict(table.detector_overrides)
        result = analyze_file(
            traces, prices=table, config=config, anomaly=not no_anomaly, strict=strict
        )
    except (TraceParseError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if not result.runs:
        raise click.ClickException(f"no valid spans found in {traces}")
    _print_result(result, top)
    for path, content in (
        (report_md, lambda: to_markdown(result)),
        (report_html, lambda: to_html(result, table)),
        (report_json, lambda: json.dumps(to_json(result), indent=1, default=str)),
    ):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content(), encoding="utf-8")
            console.print(f"[green]wrote {path}[/]")
    if findings_csv:
        result.findings_df().to_csv(findings_csv, index=False)
        console.print(f"[green]wrote {findings_csv}[/]")
    if fail_on_waste is not None and result.summary["savings_pct"] > fail_on_waste:
        console.print(f"[red]waste {result.summary['savings_pct']:.1f}% > {fail_on_waste}%[/]")
        sys.exit(2)


@main.command("trace")
@click.argument("traces", type=click.Path(exists=True, dir_okay=False))
@click.argument("trace_id")
@click.option("--prices", type=click.Path(exists=True, dir_okay=False))
def trace_cmd(traces: str, trace_id: str, prices: str | None) -> None:
    """Print one run as a span tree (prefix of the trace id is enough)."""
    table = _load_prices(prices)
    runs, _ = load_runs(traces)
    matches = [r for r in runs if r.trace_id.startswith(trace_id)]
    if not matches:
        raise click.ClickException(f"trace {trace_id!r} not found")
    run = matches[0]
    tree = Tree(f"[bold red]{run.agent}[/] {run.trace_id} · {run.duration_ms / 1000:.1f}s")
    for span in run.spans:
        if span.parent_id is None:
            continue
        status = "[red]ERROR[/]" if span.is_error else "[green]ok[/]"
        if span.kind == "llm":
            cost = span_cost(span, table)
            tree.add(
                f"[magenta]llm[/] {span.name} {span.model} {span.input_tokens}→"
                f"{span.output_tokens} tok ${cost:.4f} {span.latency_ms:.0f}ms {status}"
            )
        else:
            tree.add(
                f"[yellow]tool[/] {span.tool_name} {json.dumps(span.tool_args)[:60]} "
                f"{span.latency_ms:.0f}ms {status}"
            )
    console.print(tree)


@main.command("eval")
@click.option("--runs", "n_runs", default=1000, show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option(
    "--robustness-seeds",
    default="1,2,3,4,5",
    show_default=True,
    help="Comma-separated extra seeds for F1 mean/std ('' to skip).",
)
@click.option(
    "--throughput-runs",
    default=5000,
    show_default=True,
    help="Runs in the throughput dataset (0 to skip).",
)
@click.option("--out", default="results", show_default=True, type=click.Path(file_okay=False))
def eval_cmd(n_runs: int, seed: int, robustness_seeds: str, throughput_runs: int, out: str) -> None:
    """Benchmark detectors against injected labels and measure throughput."""
    from agentlens.evaluate import run_benchmark
    from agentlens.evaluate import to_markdown as eval_markdown

    seeds = tuple(int(x) for x in robustness_seeds.split(",") if x.strip())
    results = run_benchmark(
        n_runs=n_runs, seed=seed, robustness_seeds=seeds, throughput_runs=throughput_runs
    )
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (out_dir / "eval_results.md").write_text(eval_markdown(results), encoding="utf-8")

    det = results["detection"]
    t = Table(
        title=f"Detector accuracy · {det['n_runs']} synthetic runs · seed {det['seed']}",
        title_style="bold yellow",
        header_style="bold red",
    )
    for col in ("Detector", "Support", "Precision", "Recall", "F1"):
        t.add_column(col, justify="left" if col == "Detector" else "right")
    rows = [
        *det["detectors"].items(),
        ("rules (any)", det["rules_any"]),
        ("IsolationForest (any)", det["isolation_forest"]),
    ]
    for name, m in rows:
        t.add_row(
            name,
            str(m["support"]),
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"[bold]{m['f1']:.3f}[/]",
        )
    console.print(t)
    if results["throughput"]:
        tp = results["throughput"]
        console.print(
            f"Throughput: [bold]{tp['spans_per_sec_full']:,}[/] spans/sec full pipeline, "
            f"{tp['spans_per_sec_rules_only']:,} spans/sec rules only "
            f"({tp['n_spans']:,} spans)"
        )
    console.print(
        f"[green]wrote {out_dir / 'eval_results.json'} and {out_dir / 'eval_results.md'}[/]"
    )
