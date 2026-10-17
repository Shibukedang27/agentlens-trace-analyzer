"""agentlens Streamlit dashboard.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agentlens.analyzer import AnalysisResult, analyze_runs
from agentlens.detectors import DETECTORS, DetectorConfig
from agentlens.generator import GeneratorConfig, generate
from agentlens.parser import group_runs, load_runs, parse_span
from agentlens.pricing import PriceTable
from agentlens.report import DISCLAIMER

ROOT = Path(__file__).resolve().parents[1]
RED, ORANGE, GOLD, GREY = "#e0115f", "#ff8c1a", "#ffd166", "#5a5a66"
PLOT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
}

st.set_page_config(page_title="agentlens · agent trace waste", page_icon="🔥", layout="wide")
st.markdown(
    """<style>
    [data-testid="stMetric"]{background:linear-gradient(160deg,#1b1b21,#121215);
      border:1px solid #2a2a31;border-left:4px solid #e0115f;border-radius:10px;padding:12px 16px}
    [data-testid="stMetricValue"]{font-size:1.7rem}
    h1 span.red{color:#e0115f}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def demo_records(n_runs: int, seed: int) -> list[dict]:
    records, _ = generate(GeneratorConfig(n_runs=n_runs, seed=seed))
    return records


@st.cache_resource(show_spinner="Analyzing traces...")
def analyze_cached(
    source_key: str,
    payload: bytes | None,
    n_runs: int,
    seed: int,
    prices_yaml: bytes | None,
    cfg_json: str,
) -> AnalysisResult:
    if prices_yaml:
        import yaml

        prices = PriceTable.from_dict(yaml.safe_load(prices_yaml))
    else:
        prices = PriceTable.default_table()
    if payload is not None:
        with tempfile.NamedTemporaryFile("wb", suffix=".jsonl", delete=False) as tmp:
            tmp.write(payload)
        runs, errors = load_runs(tmp.name)
    else:
        runs, errors = group_runs(parse_span(r) for r in demo_records(n_runs, seed)), []
    result = analyze_runs(runs, prices=prices, config=DetectorConfig(**json.loads(cfg_json)))
    result.parse_errors = errors
    return result


# ---------------------------------------------------------------- sidebar
st.sidebar.title("agentlens")
source = st.sidebar.radio("Data", ["Demo dataset (synthetic)", "Upload JSONL"])
payload = None
n_runs, seed = 600, 42
if source == "Upload JSONL":
    upload = st.sidebar.file_uploader("Trace file (.jsonl)", type=["jsonl", "json"])
    if upload is None:
        st.info("Upload a JSONL file of spans, or switch to the demo dataset.")
        st.stop()
    payload = upload.getvalue()
else:
    n_runs = st.sidebar.slider("Demo runs", 100, 2000, 600, step=100)
    seed = int(st.sidebar.number_input("Seed", value=42, step=1))
price_file = st.sidebar.file_uploader("Price table (.yaml, optional)", type=["yaml", "yml"])
with st.sidebar.expander("Detector thresholds"):
    cfg = {
        "loop_min_repeats": st.slider("Loop: min identical calls", 2, 8, 3),
        "retry_min_consecutive_failures": st.slider(
            "Retry storm: min consecutive failures", 2, 8, 3
        ),
        "bloat_min_growth_ratio": st.slider("Prompt bloat: min growth ratio", 1.5, 6.0, 2.5, 0.5),
        "idle_max_output_tokens": st.slider("Idle: max output tokens", 10, 120, 40, 10),
        "oversized_min_calls": st.slider("Oversized: min trivial calls", 1, 5, 2),
    }
st.sidebar.caption(DISCLAIMER)

result = analyze_cached(
    source, payload, n_runs, seed, price_file.getvalue() if price_file else None, json.dumps(cfg)
)
s = result.summary
cur = "$" if result.currency == "USD" else result.currency + " "

st.markdown(
    "<h1>agent<span class='red'>lens</span> · where are your agents burning tokens?</h1>",
    unsafe_allow_html=True,
)
st.caption(f"{s['runs']:,} runs · {s['spans']:,} spans · {s['agents']} agents — {DISCLAIMER}")

tab_overview, tab_findings, tab_trace, tab_eval = st.tabs(
    ["Overview", "Waste findings", "Trace timeline", "Benchmark"]
)

# ---------------------------------------------------------------- overview
with tab_overview:
    c = st.columns(4)
    c[0].metric("Total cost", f"{cur}{s['total_cost']:,.2f}")
    c[1].metric(
        "Estimated savings", f"{cur}{s['est_savings']:,.2f}", f"{s['savings_pct']:.1f}% of spend"
    )
    c[2].metric(
        "Flagged runs",
        f"{s['flagged_runs']:,} / {s['runs']:,}",
        f"{s['findings']} findings",
        delta_color="off",
    )
    c[3].metric("Total tokens", f"{s['total_tokens'] / 1e6:,.2f}M")
    c = st.columns(4)
    c[0].metric("LLM latency p50", f"{s['p50_llm_ms'] / 1000:.2f}s")
    c[1].metric("LLM latency p95", f"{s['p95_llm_ms'] / 1000:.2f}s")
    c[2].metric("Run duration p95", f"{s['p95_run_ms'] / 1000:.1f}s")
    c[3].metric("Span error rate", f"{s['error_rate'] * 100:.1f}%")

    left, right = st.columns(2)
    agents = result.agent_df
    savings_by_agent = result.run_df.groupby("agent")["est_savings"].sum()
    fig = go.Figure()
    fig.add_bar(
        y=agents["agent"],
        x=agents["total_cost"] - agents["agent"].map(savings_by_agent),
        name="Useful spend",
        orientation="h",
        marker_color=RED,
    )
    fig.add_bar(
        y=agents["agent"],
        x=agents["agent"].map(savings_by_agent),
        name="Estimated waste",
        orientation="h",
        marker_color=GOLD,
    )
    fig.update_layout(
        barmode="stack",
        title="Cost by agent (useful vs. waste)",
        height=360,
        xaxis_title=f"cost ({result.currency})",
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        **PLOT,
    )
    fig.update_yaxes(autorange="reversed")
    left.plotly_chart(fig)

    sav = result.savings_by_detector()
    fig = go.Figure(
        go.Bar(
            x=sav["label"],
            y=sav["est_savings"],
            marker_color=ORANGE,
            text=[f"{cur}{v:,.2f}" for v in sav["est_savings"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Estimated savings by detector",
        height=360,
        yaxis_title=f"savings ({result.currency})",
        **PLOT,
    )
    right.plotly_chart(fig)

    runs_df = result.run_df.assign(flagged=result.run_df["flags"] != "")
    fig = go.Figure()
    for flagged, color, label in ((False, RED, "clean run"), (True, GOLD, "flagged run")):
        part = runs_df[runs_df["flagged"] == flagged]
        fig.add_scatter(
            x=part["total_tokens"],
            y=part["cost"],
            mode="markers",
            name=label,
            marker={"color": color, "size": 7, "opacity": 0.75},
            text=part["agent"] + " · " + part["flags"],
            hovertemplate="%{text}<br>%{x:,} tokens · $%{y:.4f}<extra></extra>",
        )
    fig.update_layout(
        title="Every run: tokens vs. cost (gold = flagged by a detector)",
        height=320,
        xaxis_title="tokens per run",
        yaxis_title=f"cost ({result.currency})",
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        **PLOT,
    )
    st.plotly_chart(fig)

    st.subheader("Per-agent metrics")
    st.dataframe(
        agents.rename(columns=lambda col: col.replace("_", " ")),
        hide_index=True,
        column_config={
            "total cost": st.column_config.NumberColumn(format="$%.2f"),
            "avg cost per run": st.column_config.NumberColumn(format="$%.4f"),
            "error rate": st.column_config.NumberColumn(format="%.3f"),
        },
    )

# ---------------------------------------------------------------- findings
with tab_findings:
    fdf = result.findings_df()
    c1, c2 = st.columns(2)
    dets = c1.multiselect(
        "Detector", [d.name for d in DETECTORS], default=[d.name for d in DETECTORS]
    )
    ags = c2.multiselect(
        "Agent",
        sorted(result.run_df["agent"].unique()),
        default=sorted(result.run_df["agent"].unique()),
    )
    view = fdf[fdf["detector"].isin(dets) & fdf["agent"].isin(ags)].sort_values(
        "est_savings", ascending=False
    )
    c = st.columns(3)
    c[0].metric("Findings", f"{len(view):,}")
    c[1].metric("Runs affected", f"{view['trace_id'].nunique():,}")
    c[2].metric("Est. savings (uncapped)", f"{cur}{view['est_savings'].sum():,.2f}")
    st.dataframe(
        view.assign(trace_id=view["trace_id"].str[:12]),
        hide_index=True,
        height=520,
        column_config={
            "est_savings": st.column_config.NumberColumn("est. savings", format="$%.4f"),
            "evidence": st.column_config.TextColumn(width="large"),
        },
    )
    buf = io.StringIO()
    view.to_csv(buf, index=False)
    st.download_button(
        "Download findings CSV", buf.getvalue(), "agentlens_findings.csv", "text/csv"
    )

# ---------------------------------------------------------------- trace view
with tab_trace:
    ranked = result.run_df.sort_values(["est_savings", "cost"], ascending=False)
    options = ranked["trace_id"].tolist()
    labels = {
        r.trace_id: f"{r.trace_id[:10]} · {r.agent} · {cur}{r.cost:.4f} · {r.flags or 'clean'}"
        for r in ranked.itertuples()
    }
    trace_id = st.selectbox("Run (sorted by estimated waste)", options, format_func=labels.get)
    run = result.run(trace_id)
    row = ranked.set_index("trace_id").loc[trace_id]
    c = st.columns(5)
    c[0].metric("Agent", run.agent)
    c[1].metric("Duration", f"{run.duration_ms / 1000:.1f}s")
    c[2].metric("Cost", f"{cur}{row['cost']:.4f}")
    c[3].metric("Est. waste", f"{cur}{row['est_savings']:.4f}")
    c[4].metric("Anomaly score", f"{row['anomaly_score']:.2f}")
    flagged = {sid for f in result.findings if f.trace_id == trace_id for sid in f.span_ids}
    spans = result.span_df[result.span_df["trace_id"] == trace_id].reset_index(drop=True)
    names = [
        f"{i:02d} {(t if isinstance(t, str) else n)}" + (f" · {m}" if isinstance(m, str) else "")
        for i, (t, n, m) in enumerate(zip(spans["tool_name"], spans["name"], spans["model"]))
    ]
    colors = [
        "#ff3b3b" if st_ == "error" else {"llm": RED, "tool": ORANGE}.get(k, GREY)
        for k, st_ in zip(spans["kind"], spans["status"])
    ]
    fig = go.Figure(
        go.Bar(
            y=names,
            x=spans["latency_ms"] / 1000,
            base=spans["offset_ms"] / 1000,
            orientation="h",
            marker={
                "color": colors,
                "line": {
                    "color": [
                        GOLD if sid in flagged else "rgba(0,0,0,0)" for sid in spans["span_id"]
                    ],
                    "width": 2,
                },
            },
            customdata=spans[["input_tokens", "output_tokens", "cost", "status"]],
            hovertemplate="%{y}<br>start %{base:.2f}s, %{x:.2f}s<br>tokens %{customdata[0]}→"
            "%{customdata[1]}<br>cost $%{customdata[2]:.4f} · %{customdata[3]}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Span waterfall (crimson = LLM, orange = tool, red = error, gold outline = flagged)",
        height=max(320, 30 * len(spans) + 80),
        xaxis_title="seconds since run start",
        **PLOT,
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig)
    for f in result.findings:
        if f.trace_id == trace_id:
            st.warning(
                f"**{f.detector}** — {f.title}  \n{f.evidence}  \nEst. savings {cur}{f.est_savings:.4f}"
            )
    st.dataframe(
        spans[
            [
                "kind",
                "name",
                "model",
                "tool_name",
                "status",
                "input_tokens",
                "output_tokens",
                "latency_ms",
                "cost",
            ]
        ],
        hide_index=True,
    )

# ---------------------------------------------------------------- benchmark
with tab_eval:
    path = ROOT / "results" / "eval_results.json"
    if not path.exists():
        st.info("Run `agentlens eval` to produce results/eval_results.json.")
    else:
        res = json.loads(path.read_text())
        det = res["detection"]
        st.caption(
            f"Synthetic benchmark: {det['n_runs']} runs / {det['n_spans']:,} spans, seed "
            f"{det['seed']}, {det['anomalous_runs']} runs with injected anomalies."
        )
        rows = [{"detector": k, **v} for k, v in det["detectors"].items()]
        rows.append({"detector": "rules: any", **det["rules_any"]})
        rows.append({"detector": "IsolationForest", **det["isolation_forest"]})
        table = pd.DataFrame(rows)
        fig = go.Figure()
        for metric, color in (("precision", RED), ("recall", ORANGE), ("f1", GOLD)):
            fig.add_bar(
                x=table["detector"],
                y=table[metric],
                name=metric,
                marker_color=color,
                text=[f"{v:.2f}" for v in table[metric]],
                textposition="outside",
            )
        fig.update_layout(
            barmode="group",
            title="Detector precision / recall / F1 vs injected labels",
            height=420,
            yaxis_range=[0, 1.12],
            **PLOT,
        )
        st.plotly_chart(fig)
        st.dataframe(table, hide_index=True)
        tp = res.get("throughput") or {}
        if tp:
            c = st.columns(3)
            c[0].metric("Throughput (full pipeline)", f"{tp['spans_per_sec_full']:,} spans/s")
            c[1].metric("Throughput (rules only)", f"{tp['spans_per_sec_rules_only']:,} spans/s")
            c[2].metric("Benchmark size", f"{tp['n_spans']:,} spans")
