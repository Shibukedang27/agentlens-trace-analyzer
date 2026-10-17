"""Rewrite the generated sections of README.md from results/*.json (no hand-typed numbers).

agentlens eval && agentlens analyze data/demo_traces.jsonl --json results/demo_report.json
python scripts/update_readme.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace(text: str, tag: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- {tag}:START -->\n).*?(<!-- {tag}:END -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"marker {tag} not found in README.md")
    return pattern.sub(lambda m: m.group(1) + body.strip() + "\n" + m.group(2), text)


def main() -> None:
    ev = json.loads((ROOT / "results" / "eval_results.json").read_text())
    demo = json.loads((ROOT / "results" / "demo_report.json").read_text())
    det = ev["detection"]
    rob = ev["robustness"]
    tp = ev["throughput"]
    s = demo["summary"]

    rows = [
        "| Detector | Support | Precision | Recall | F1 | F1 mean ± std (5 other seeds) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in det["detectors"].items():
        r = rob.get(name, {})
        rows.append(
            f"| `{name}` | {m['support']} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"**{m['f1']:.3f}** | {r.get('f1_mean', 0):.3f} ± {r.get('f1_std', 0):.3f} |"
        )
    ra, iso = det["rules_any"], det["isolation_forest"]
    rows.append(
        f"| any rule (run has any anomaly) | {ra['support']} | {ra['precision']:.3f} | "
        f"{ra['recall']:.3f} | **{ra['f1']:.3f}** | |"
    )
    rows.append(
        f"| IsolationForest (any anomaly) | {iso['support']} | {iso['precision']:.3f} | "
        f"{iso['recall']:.3f} | **{iso['f1']:.3f}** | ROC-AUC {iso['roc_auc']:.3f} |"
    )
    m = tp["machine"]
    results = "\n".join(
        [
            f"Benchmark dataset: **{det['n_runs']:,} synthetic runs / {det['n_spans']:,} spans** "
            f"(seed {det['seed']}), {det['anomalous_runs']} runs with injected anomalies. "
            "Scores are run-level: a run counts as a true positive for a detector if the "
            "detector flags it and the generator injected that pattern.",
            "",
            *rows,
            "",
            f"**Throughput** ({tp['n_spans']:,} spans / {tp['n_runs']:,} runs, median of "
            f"{tp['repeats']} runs, Python {m['python']}, {m['cpu_count']} CPUs): "
            f"**{tp['spans_per_sec_full']:,} spans/sec** for the full file pipeline "
            f"(parse + metrics + 5 detectors + IsolationForest), "
            f"{tp['spans_per_sec_rules_only']:,} spans/sec without IsolationForest.",
        ]
    )
    demo_md = "\n".join(
        [
            "| Demo dataset (`agentlens generate`, seed 42) | Value |",
            "|---|---:|",
            f"| Runs / spans / agents | {s['runs']:,} / {s['spans']:,} / {s['agents']} |",
            f"| Total tokens | {s['total_tokens']:,} |",
            f"| Total cost (example prices) | ${s['total_cost']:,.2f} |",
            f"| Estimated savings if flagged waste were fixed | ${s['est_savings']:,.2f} "
            f"({s['savings_pct']:.1f}%) |",
            f"| Runs flagged / findings | {s['flagged_runs']} / {s['findings']} |",
            f"| LLM latency p50 / p95 | {s['p50_llm_ms'] / 1000:.2f}s / {s['p95_llm_ms'] / 1000:.2f}s |",
            f"| Span error rate | {s['error_rate'] * 100:.1f}% |",
        ]
    )
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "--cov=agentlens",
            "--cov-report=term",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    n_tests = int(re.search(r"(\d+) passed", run).group(1))
    coverage = re.search(r"^TOTAL\s+\d+\s+\d+\s+(\d+)%", run, re.M).group(1)
    readme = ROOT / "README.md"
    text = readme.read_text()
    text = replace(text, "RESULTS", results)
    text = replace(text, "DEMO", demo_md)
    text = replace(
        text,
        "TESTS",
        f"**{n_tests} pytest tests, all passing, {coverage}% line coverage** (parser, pricing, "
        "metrics, every detector, generator, eval, CLI, reports and a Streamlit smoke test).",
    )
    readme.write_text(text)
    print("README.md updated")


if __name__ == "__main__":
    main()
