"""Markdown, JSON and self-contained HTML reports (no external assets, no JS)."""

from __future__ import annotations

import html
from typing import Any

from agentlens.analyzer import AnalysisResult
from agentlens.metrics import span_cost
from agentlens.models import Run
from agentlens.pricing import PriceTable

DISCLAIMER = "Synthetic traces, example prices: numbers illustrate the method, not real billing."

COLORS = {
    "llm": "#e0115f",
    "tool": "#ff8c1a",
    "agent": "#5a5a66",
    "error": "#ff3b3b",
    "flag": "#ffd166",
}


def _money(value: float, currency: str = "USD") -> str:
    symbol = "$" if currency == "USD" else f"{currency} "
    return f"{symbol}{value:,.2f}" if abs(value) >= 1 else f"{symbol}{value:,.4f}"


def _ms(value: float) -> str:
    return f"{value / 1000:.1f}s" if value >= 1000 else f"{value:.0f}ms"


def top_findings(result: AnalysisResult, n: int = 10) -> list[dict[str, Any]]:
    items = sorted(result.findings, key=lambda f: f.est_savings, reverse=True)[:n]
    return [f.to_dict() for f in items]


def most_wasteful_run(result: AnalysisResult) -> Run | None:
    if result.run_df.empty or result.run_df["est_savings"].max() <= 0:
        return None
    trace_id = result.run_df.sort_values("est_savings", ascending=False).iloc[0]["trace_id"]
    return result.run(str(trace_id))


def to_json(result: AnalysisResult) -> dict[str, Any]:
    return {
        "summary": result.summary,
        "savings_by_detector": result.savings_by_detector().to_dict(orient="records"),
        "agents": result.agent_df.to_dict(orient="records"),
        "models": result.model_df.to_dict(orient="records"),
        "findings": [f.to_dict() for f in result.findings],
        "parse_errors": result.parse_errors,
        "unknown_models": result.unknown_models,
        "disclaimer": DISCLAIMER,
    }


def to_markdown(result: AnalysisResult, top: int = 10) -> str:
    s = result.summary
    cur = result.currency
    lines = [
        "# agentlens report",
        "",
        f"> {DISCLAIMER}",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Runs | {s['runs']:,} |",
        f"| Spans | {s['spans']:,} |",
        f"| Agents | {s['agents']} |",
        f"| Total tokens | {s['total_tokens']:,} |",
        f"| Total cost | {_money(s['total_cost'], cur)} |",
        f"| Estimated savings | {_money(s['est_savings'], cur)} ({s['savings_pct']:.1f}%) |",
        f"| Runs flagged by detectors | {s['flagged_runs']:,} ({s['findings']:,} findings) |",
        f"| Span error rate | {s['error_rate'] * 100:.1f}% |",
        f"| LLM latency p50 / p95 | {_ms(s['p50_llm_ms'])} / {_ms(s['p95_llm_ms'])} |",
        f"| Run duration p50 / p95 | {_ms(s['p50_run_ms'])} / {_ms(s['p95_run_ms'])} |",
        "",
        "## Waste by detector",
        "",
        "| Detector | Findings | Runs | Wasted tokens | Est. savings |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result.savings_by_detector().to_dict(orient="records"):
        lines.append(
            f"| {row['label']} | {row['findings']} | {row['runs']} | {row['wasted_tokens']:,} | "
            f"{_money(row['est_savings'], cur)} |"
        )
    lines += [
        "",
        "Per-detector savings are not capped; the overview total caps each run at its own cost.",
        "",
        "## Cost by agent",
        "",
        "| Agent | Runs | Total cost | Avg cost/run | p50 run | p95 run | p95 LLM | Error rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.agent_df.to_dict(orient="records"):
        lines.append(
            f"| {row['agent']} | {row['runs']} | {_money(row['total_cost'], cur)} | "
            f"{_money(row['avg_cost_per_run'], cur)} | {_ms(row['p50_run_ms'])} | "
            f"{_ms(row['p95_run_ms'])} | {_ms(row['p95_llm_ms'])} | {row['error_rate'] * 100:.1f}% |"
        )
    lines += ["", f"## Top {top} findings by estimated savings", ""]
    lines += ["| Detector | Agent | Trace | Finding | Est. savings |", "|---|---|---|---|---:|"]
    for f in top_findings(result, top):
        lines.append(
            f"| {f['detector']} | {f['agent']} | `{f['trace_id'][:12]}` | {f['title']} | "
            f"{_money(f['est_savings'], cur)} |"
        )
    if result.parse_errors:
        lines += ["", f"## Parse errors ({len(result.parse_errors)})", ""]
        lines += [f"- {e}" for e in result.parse_errors[:20]]
    if result.unknown_models:
        lines += [
            "",
            "Unknown models (priced with the default rate): " + ", ".join(result.unknown_models),
        ]
    return "\n".join(lines) + "\n"


def svg_hbar(
    items: list[tuple[str, float]], fmt: Any, color: str = "#e0115f", width: int = 560
) -> str:
    if not items:
        return "<p>No data.</p>"
    row_h, label_w = 30, 150
    height = row_h * len(items) + 10
    peak = max(v for _, v in items) or 1.0
    bar_w = width - label_w - 110
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for i, (label, value) in enumerate(items):
        y = 5 + i * row_h
        w = max(2.0, bar_w * value / peak)
        parts.append(
            f'<text x="{label_w - 8}" y="{y + 19}" text-anchor="end" class="lbl">'
            f"{html.escape(label)}</text>"
            f'<rect x="{label_w}" y="{y + 4}" width="{w:.1f}" height="20" rx="3" fill="{color}"/>'
            f'<text x="{label_w + w + 8:.1f}" y="{y + 19}" class="val">{html.escape(fmt(value))}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def svg_waterfall(run: Run, prices: PriceTable, flagged: set[str], width: int = 1100) -> str:
    spans = run.spans
    total = max(run.duration_ms, 1.0)
    t0 = run.start
    row_h, label_w = 22, 230
    height = row_h * len(spans) + 30
    plot_w = width - label_w - 200
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for i, span in enumerate(spans):
        y = 5 + i * row_h
        x = label_w + plot_w * (span.start_time - t0).total_seconds() * 1000.0 / total
        w = max(2.0, plot_w * span.latency_ms / total)
        color = COLORS["error"] if span.is_error else COLORS.get(span.kind, "#888")
        label = span.tool_name or (f"{span.name} · {span.model}" if span.model else span.name)
        stroke = f' stroke="{COLORS["flag"]}" stroke-width="2"' if span.span_id in flagged else ""
        cost = span_cost(span, prices)
        info = f"{span.input_tokens}→{span.output_tokens} tok" if span.kind == "llm" else ""
        parts.append(
            f'<text x="{label_w - 8}" y="{y + 15}" text-anchor="end" class="lbl">'
            f"{html.escape(label[:34])}</text>"
            f'<rect x="{x:.1f}" y="{y + 3}" width="{w:.1f}" height="16" rx="2" fill="{color}"'
            f' opacity="{0.45 if span.kind == "agent" else 0.95}"{stroke}/>'
            f'<text x="{min(x + w + 6, width - 195):.1f}" y="{y + 15}" class="val">'
            f"{html.escape(info)}{' · $' + format(cost, '.4f') if cost else ''}</text>"
        )
    parts.append(
        f'<text x="{label_w}" y="{height - 6}" class="lbl">0</text>'
        f'<text x="{label_w + plot_w}" y="{height - 6}" text-anchor="end" class="lbl">'
        f"{_ms(total)}</text></svg>"
    )
    return "".join(parts)


CSS = """
:root{--bg:#0b0b0d;--panel:#16161a;--line:#2a2a31;--txt:#f4f4f6;--mut:#a3a3ad;
--red:#e0115f;--crimson:#b3122e;--orange:#ff8c1a;--gold:#ffd166}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);
font:15px/1.5 "Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:32px}
header{border-bottom:3px solid var(--crimson);padding-bottom:14px;margin-bottom:22px}
h1{margin:0;font-size:30px;letter-spacing:.5px}h1 span{color:var(--red)}
h2{font-size:19px;margin:28px 0 12px;color:var(--gold);text-transform:uppercase;letter-spacing:1px}
.note{color:var(--mut);font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.kpi{background:linear-gradient(160deg,#1b1b21,#121215);border:1px solid var(--line);
border-left:4px solid var(--red);border-radius:10px;padding:14px 16px}
.kpi.save{border-left-color:var(--gold)}.kpi .v{font-size:26px;font-weight:700}
.kpi .k{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:1px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;color:var(--gold);border-bottom:2px solid var(--crimson);padding:7px 8px}
td{border-bottom:1px solid var(--line);padding:6px 8px;vertical-align:top}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:1px 8px;border-radius:9px;font-size:12px;
background:#2a1016;color:#ff6b8b;border:1px solid #5a1a28}
.sev-high{color:#ff5c5c}.sev-medium{color:var(--orange)}.sev-low{color:var(--gold)}
svg .lbl{fill:#c9c9d1;font-size:12px}svg .val{fill:#f4f4f6;font-size:12px}
code{color:var(--orange)}
"""


def to_html(result: AnalysisResult, prices: PriceTable | None = None, top: int = 15) -> str:
    prices = prices or PriceTable.default_table()
    s = result.summary
    cur = result.currency
    esc = html.escape

    def kpi(label: str, value: str, cls: str = "") -> str:
        return f'<div class="kpi {cls}"><div class="k">{esc(label)}</div><div class="v">{esc(value)}</div></div>'

    kpis = "".join(
        [
            kpi("Runs analysed", f"{s['runs']:,}"),
            kpi("Spans", f"{s['spans']:,}"),
            kpi("Total cost", _money(s["total_cost"], cur)),
            kpi(
                "Est. savings", f"{_money(s['est_savings'], cur)} ({s['savings_pct']:.1f}%)", "save"
            ),
            kpi("Flagged runs", f"{s['flagged_runs']:,} / {s['runs']:,}"),
            kpi("Findings", f"{s['findings']:,}"),
            kpi("Span error rate", f"{s['error_rate'] * 100:.1f}%"),
            kpi("LLM p50 / p95", f"{_ms(s['p50_llm_ms'])} / {_ms(s['p95_llm_ms'])}"),
        ]
    )
    sav = result.savings_by_detector()
    sav_chart = svg_hbar(
        [(r["label"], r["est_savings"]) for r in sav.to_dict(orient="records")],
        lambda v: _money(v, cur),
        color="#ff8c1a",
    )
    cost_chart = svg_hbar(
        [(r["agent"], r["total_cost"]) for r in result.agent_df.to_dict(orient="records")],
        lambda v: _money(v, cur),
    )
    agent_rows = "".join(
        f"<tr><td>{esc(r['agent'])}</td><td class=n>{r['runs']}</td>"
        f"<td class=n>{_money(r['total_cost'], cur)}</td><td class=n>{_money(r['avg_cost_per_run'], cur)}</td>"
        f"<td class=n>{r['total_tokens']:,}</td><td class=n>{_ms(r['p50_run_ms'])}</td>"
        f"<td class=n>{_ms(r['p95_run_ms'])}</td><td class=n>{_ms(r['p95_llm_ms'])}</td>"
        f"<td class=n>{r['error_rate'] * 100:.1f}%</td></tr>"
        for r in result.agent_df.to_dict(orient="records")
    )
    finding_rows = "".join(
        f"<tr><td><span class=badge>{esc(f['detector'])}</span></td><td>{esc(f['agent'])}</td>"
        f"<td><code>{esc(f['trace_id'][:10])}</code></td>"
        f"<td class='sev-{esc(f['severity'])}'>{esc(f['severity'])}</td>"
        f"<td><b>{esc(f['title'])}</b><br><span class=note>{esc(f['evidence'])}</span></td>"
        f"<td class=n>{_money(f['est_savings'], cur)}</td></tr>"
        for f in top_findings(result, top)
    )
    waste_run = most_wasteful_run(result)
    waterfall = ""
    if waste_run is not None:
        flagged = {
            sid for f in result.findings if f.trace_id == waste_run.trace_id for sid in f.span_ids
        }
        dets = sorted({f.detector for f in result.findings if f.trace_id == waste_run.trace_id})
        waterfall = (
            f"<h2>Most wasteful run: timeline</h2><div class=panel><p class=note>Trace "
            f"<code>{esc(waste_run.trace_id)}</code> · agent {esc(waste_run.agent)} · flagged by "
            f"{esc(', '.join(dets))}. Crimson = LLM call, orange = tool call, red = error, "
            f"gold outline = span named in a finding.</p>{svg_waterfall(waste_run, prices, flagged)}</div>"
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>agentlens report</title>
<style>{CSS}</style></head><body><div class="wrap">
<header><h1>agent<span>lens</span> · trace waste report</h1>
<div class="note">{esc(DISCLAIMER)}</div></header>
<div class="kpis">{kpis}</div>
<div class="grid2" style="margin-top:18px">
<div class="panel"><h2 style="margin-top:0">Est. savings by detector</h2>{sav_chart}</div>
<div class="panel"><h2 style="margin-top:0">Cost by agent</h2>{cost_chart}</div></div>
<h2>Agents</h2><div class="panel"><table><tr><th>Agent</th><th>Runs</th><th>Total cost</th>
<th>Avg/run</th><th>Tokens</th><th>p50 run</th><th>p95 run</th><th>p95 LLM</th><th>Errors</th></tr>
{agent_rows}</table></div>
<h2>Top {top} findings</h2><div class="panel"><table><tr><th>Detector</th><th>Agent</th><th>Trace</th>
<th>Severity</th><th>Finding &amp; evidence</th><th>Est. savings</th></tr>{finding_rows}</table></div>
{waterfall}
<p class="note" style="margin-top:24px">Generated by agentlens. Per-detector savings are not capped;
the total caps each run at its own cost.</p>
</div></body></html>"""
