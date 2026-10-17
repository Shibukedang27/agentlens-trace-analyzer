"""High-level entry point: parse -> metrics -> detectors -> savings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from agentlens.anomaly import anomaly_scores
from agentlens.detectors import DETECTORS, DetectorConfig, Finding, run_detectors
from agentlens.metrics import agent_frame, model_frame, run_frame, span_frame, summarize
from agentlens.models import Run
from agentlens.parser import load_runs
from agentlens.pricing import PriceTable

FINDING_COLUMNS = [
    "detector",
    "trace_id",
    "agent",
    "severity",
    "title",
    "evidence",
    "wasted_tokens",
    "est_savings",
]


@dataclass
class AnalysisResult:
    runs: list[Run]
    run_df: pd.DataFrame
    span_df: pd.DataFrame
    agent_df: pd.DataFrame
    model_df: pd.DataFrame
    findings: list[Finding]
    summary: dict[str, Any]
    currency: str = "USD"
    parse_errors: list[str] = field(default_factory=list)
    unknown_models: list[str] = field(default_factory=list)

    def findings_df(self) -> pd.DataFrame:
        if not self.findings:
            return pd.DataFrame(columns=FINDING_COLUMNS)
        return pd.DataFrame([f.to_dict() for f in self.findings])[FINDING_COLUMNS]

    def savings_by_detector(self) -> pd.DataFrame:
        """Findings, affected runs, wasted tokens and (uncapped) savings per detector."""
        rows = []
        for det in DETECTORS:
            items = [f for f in self.findings if f.detector == det.name]
            rows.append(
                {
                    "detector": det.name,
                    "label": det.label,
                    "findings": len(items),
                    "runs": len({f.trace_id for f in items}),
                    "wasted_tokens": sum(f.wasted_tokens for f in items),
                    "est_savings": sum(f.est_savings for f in items),
                }
            )
        return pd.DataFrame(rows)

    def run(self, trace_id: str) -> Run:
        for run in self.runs:
            if run.trace_id == trace_id:
                return run
        raise KeyError(trace_id)


def analyze_runs(
    runs: list[Run],
    prices: PriceTable | None = None,
    config: DetectorConfig | None = None,
    anomaly: bool = True,
    contamination: float = 0.1,
    seed: int = 0,
) -> AnalysisResult:
    prices = prices or PriceTable.default_table()
    if config is None:
        config = DetectorConfig.from_dict(prices.detector_overrides)
    span_df = span_frame(runs, prices)
    run_df = run_frame(runs, prices)
    findings = run_detectors(runs, prices, config)

    savings: dict[str, float] = {}
    flags: dict[str, list[str]] = {}
    for f in findings:
        savings[f.trace_id] = savings.get(f.trace_id, 0.0) + f.est_savings
        flags.setdefault(f.trace_id, [])
        if f.detector not in flags[f.trace_id]:
            flags[f.trace_id].append(f.detector)
    if not run_df.empty:
        # a run can never save more than it cost (avoids double counting overlaps)
        raw = run_df["trace_id"].map(savings).fillna(0.0)
        run_df["est_savings"] = raw.where(raw <= run_df["cost"], run_df["cost"])
        run_df["flags"] = run_df["trace_id"].map(lambda t: ",".join(flags.get(t, [])))
        if anomaly:
            score, outlier = anomaly_scores(run_df, contamination=contamination, seed=seed)
            run_df["anomaly_score"] = score
            run_df["anomaly_outlier"] = outlier
        else:
            run_df["anomaly_score"] = 0.0
            run_df["anomaly_outlier"] = False
    else:
        for col in ("est_savings", "flags", "anomaly_score", "anomaly_outlier"):
            run_df[col] = []

    summary = summarize(run_df, span_df)
    total_savings = float(run_df["est_savings"].sum()) if not run_df.empty else 0.0
    summary.update(
        {
            "findings": len(findings),
            "flagged_runs": int((run_df["flags"] != "").sum()) if not run_df.empty else 0,
            "est_savings": total_savings,
            "savings_pct": (total_savings / summary["total_cost"] * 100.0)
            if summary.get("total_cost")
            else 0.0,
            "anomaly_outliers": int(run_df["anomaly_outlier"].sum()) if not run_df.empty else 0,
            "currency": prices.currency,
        }
    )
    return AnalysisResult(
        runs=runs,
        run_df=run_df,
        span_df=span_df,
        agent_df=agent_frame(run_df, span_df),
        model_df=model_frame(span_df),
        findings=findings,
        summary=summary,
        currency=prices.currency,
        unknown_models=sorted(prices.unknown_models),
    )


def analyze_file(
    path: str | Path,
    prices: PriceTable | None = None,
    config: DetectorConfig | None = None,
    anomaly: bool = True,
    strict: bool = False,
) -> AnalysisResult:
    runs, errors = load_runs(path, strict=strict)
    result = analyze_runs(runs, prices=prices, config=config, anomaly=anomaly)
    result.parse_errors = errors
    return result
