"""Cost, token, latency and error metrics per span, run, agent and model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from agentlens.models import Run, Span
from agentlens.pricing import PriceTable

RUN_COLUMNS = [
    "trace_id",
    "agent",
    "start_time",
    "duration_ms",
    "n_spans",
    "n_llm_calls",
    "n_tool_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost",
    "n_errors",
    "error_rate",
    "llm_latency_ms",
    "max_tool_repeat",
    "input_growth_ratio",
]


def percentile(values: Sequence[float] | np.ndarray | pd.Series, q: float) -> float:
    """Linear-interpolated percentile; returns 0.0 for empty input."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def span_cost(span: Span, prices: PriceTable) -> float:
    if span.kind != "llm":
        return 0.0
    return prices.cost(span.model, span.input_tokens, span.output_tokens)


def run_cost(run: Run, prices: PriceTable) -> float:
    return sum(span_cost(s, prices) for s in run.spans)


def tool_signature(span: Span) -> str:
    import json

    try:
        args = json.dumps(span.tool_args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        args = str(span.tool_args)
    return f"{span.tool_name or span.name}|{args}"


def max_tool_repeat(run: Run) -> int:
    counts: dict[str, int] = {}
    for span in run.tool_spans:
        if span.is_error:
            continue
        sig = tool_signature(span)
        counts[sig] = counts.get(sig, 0) + 1
    return max(counts.values(), default=0)


def main_thread(run: Run, min_input_tokens: int = 600) -> list[Span]:
    """LLM calls of the run's main conversation.

    The main conversation is the model with the most LLM calls (ties broken by input
    tokens); tiny side prompts (routing, classification, <= ``min_input_tokens``)
    are excluded because they don't carry the accumulated context.
    """
    by_model: dict[str, list[Span]] = {}
    for span in run.llm_spans:
        if span.input_tokens > min_input_tokens:
            by_model.setdefault(span.model or "unknown", []).append(span)
    if not by_model:
        return []
    return max(by_model.values(), key=lambda g: (len(g), sum(s.input_tokens for s in g)))


def input_growth_ratio(run: Run) -> float:
    inputs = [s.input_tokens for s in main_thread(run)]
    if len(inputs) < 2:
        return 1.0
    return inputs[-1] / inputs[0]


def span_frame(runs: Sequence[Run], prices: PriceTable) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        t0 = run.start
        for span in run.spans:
            rows.append(
                {
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "parent_id": span.parent_id,
                    "agent": run.agent,
                    "kind": span.kind,
                    "name": span.name,
                    "model": span.model,
                    "tool_name": span.tool_name,
                    "status": span.status,
                    "input_tokens": span.input_tokens,
                    "output_tokens": span.output_tokens,
                    "latency_ms": span.latency_ms,
                    "start_time": span.start_time,
                    "offset_ms": (span.start_time - t0).total_seconds() * 1000.0,
                    "cost": span_cost(span, prices),
                }
            )
    return pd.DataFrame(rows)


def run_frame(runs: Sequence[Run], prices: PriceTable) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        llm = run.llm_spans
        n_errors = sum(1 for s in run.spans if s.is_error and s.kind != "agent")
        work = [s for s in run.spans if s.kind != "agent"]
        rows.append(
            {
                "trace_id": run.trace_id,
                "agent": run.agent,
                "start_time": run.start,
                "duration_ms": run.duration_ms,
                "n_spans": len(run.spans),
                "n_llm_calls": len(llm),
                "n_tool_calls": len(run.tool_spans),
                "input_tokens": run.input_tokens,
                "output_tokens": run.output_tokens,
                "total_tokens": run.input_tokens + run.output_tokens,
                "cost": run_cost(run, prices),
                "n_errors": n_errors,
                "error_rate": n_errors / len(work) if work else 0.0,
                "llm_latency_ms": sum(s.latency_ms for s in llm),
                "max_tool_repeat": max_tool_repeat(run),
                "input_growth_ratio": input_growth_ratio(run),
            }
        )
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def agent_frame(run_df: pd.DataFrame, span_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate run metrics per agent (cost, tokens, p50/p95 latency, error rate)."""
    if run_df.empty:
        return pd.DataFrame()
    llm = span_df[span_df["kind"] == "llm"] if not span_df.empty else span_df
    rows = []
    for agent, group in run_df.groupby("agent", sort=True):
        agent_llm = llm[llm["agent"] == agent] if not llm.empty else llm
        work_spans = int(group["n_llm_calls"].sum() + group["n_tool_calls"].sum())
        rows.append(
            {
                "agent": agent,
                "runs": len(group),
                "total_cost": float(group["cost"].sum()),
                "avg_cost_per_run": float(group["cost"].mean()),
                "total_tokens": int(group["total_tokens"].sum()),
                "avg_tokens_per_run": float(group["total_tokens"].mean()),
                "p50_run_ms": percentile(group["duration_ms"], 50),
                "p95_run_ms": percentile(group["duration_ms"], 95),
                "p50_llm_ms": percentile(agent_llm["latency_ms"], 50) if len(agent_llm) else 0.0,
                "p95_llm_ms": percentile(agent_llm["latency_ms"], 95) if len(agent_llm) else 0.0,
                "error_rate": float(group["n_errors"].sum() / work_spans) if work_spans else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("total_cost", ascending=False).reset_index(drop=True)


def model_frame(span_df: pd.DataFrame) -> pd.DataFrame:
    if span_df.empty:
        return pd.DataFrame()
    llm = span_df[span_df["kind"] == "llm"]
    rows = []
    for model, group in llm.groupby(llm["model"].fillna("unknown"), sort=True):
        rows.append(
            {
                "model": model,
                "calls": len(group),
                "input_tokens": int(group["input_tokens"].sum()),
                "output_tokens": int(group["output_tokens"].sum()),
                "cost": float(group["cost"].sum()),
                "p50_latency_ms": percentile(group["latency_ms"], 50),
                "p95_latency_ms": percentile(group["latency_ms"], 95),
                "error_rate": float((group["status"] == "error").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("cost", ascending=False).reset_index(drop=True)


def summarize(run_df: pd.DataFrame, span_df: pd.DataFrame) -> dict[str, Any]:
    if run_df.empty:
        return {"runs": 0, "spans": 0, "agents": 0, "total_cost": 0.0, "total_tokens": 0}
    work = span_df[span_df["kind"] != "agent"]
    llm = span_df[span_df["kind"] == "llm"]
    return {
        "runs": len(run_df),
        "spans": len(span_df),
        "agents": int(run_df["agent"].nunique()),
        "llm_calls": len(llm),
        "tool_calls": int((span_df["kind"] == "tool").sum()),
        "total_cost": float(run_df["cost"].sum()),
        "avg_cost_per_run": float(run_df["cost"].mean()),
        "input_tokens": int(run_df["input_tokens"].sum()),
        "output_tokens": int(run_df["output_tokens"].sum()),
        "total_tokens": int(run_df["total_tokens"].sum()),
        "error_rate": float((work["status"] == "error").mean()) if len(work) else 0.0,
        "p50_run_ms": percentile(run_df["duration_ms"], 50),
        "p95_run_ms": percentile(run_df["duration_ms"], 95),
        "p50_llm_ms": percentile(llm["latency_ms"], 50),
        "p95_llm_ms": percentile(llm["latency_ms"], 95),
    }
