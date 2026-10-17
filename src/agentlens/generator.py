"""Seeded synthetic trace generator with labeled, injected waste patterns.

Everything here is synthetic. Normal runs deliberately include *benign
look-alikes* (legit status polling, a transient error retried once, very short
FAQ answers, long-but-healthy research context, a single short call on a large
model) and some injected anomalies are deliberately *subtle* (below the default
detector thresholds), so the benchmark is not trivially solvable.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ANOMALY_TYPES: tuple[str, ...] = (
    "tool_loop",
    "retry_storm",
    "idle_run",
    "prompt_bloat",
    "oversized_model",
)


@dataclass(frozen=True)
class AgentProfile:
    name: str
    weight: float
    main_model: str
    small_model: str
    tools: tuple[str, ...]
    steps: tuple[int, int]
    base_input: tuple[int, int]
    growth: tuple[int, int]
    eligible: tuple[str, ...]


PROFILES: tuple[AgentProfile, ...] = (
    AgentProfile(
        "support-agent",
        0.28,
        "orion-large",
        "orion-mini",
        ("search_kb", "get_order", "create_ticket", "send_reply"),
        (2, 4),
        (1400, 2600),
        (120, 450),
        ("tool_loop", "retry_storm", "prompt_bloat", "oversized_model"),
    ),
    AgentProfile(
        "research-agent",
        0.20,
        "vega-pro",
        "vega-flash",
        ("web_search", "fetch_page", "summarize_doc"),
        (3, 7),
        (2500, 4000),
        (150, 600),
        ("tool_loop", "retry_storm", "idle_run", "prompt_bloat", "oversized_model"),
    ),
    AgentProfile(
        "sql-analyst",
        0.20,
        "orion-large",
        "orion-mini",
        ("list_tables", "describe_table", "run_sql", "render_chart"),
        (2, 4),
        (1800, 3000),
        (150, 500),
        ("tool_loop", "retry_storm", "prompt_bloat", "oversized_model"),
    ),
    AgentProfile(
        "ops-heartbeat",
        0.22,
        "vega-pro",
        "vega-flash",
        ("list_alerts", "check_status", "restart_service", "notify_oncall"),
        (1, 3),
        (1600, 2800),
        (100, 300),
        ("tool_loop", "retry_storm", "idle_run", "oversized_model"),
    ),
    AgentProfile(
        "faq-bot",
        0.10,
        "vega-flash",
        "vega-flash",
        (),
        (0, 0),
        (250, 900),
        (0, 0),
        (),
    ),
)

DEFAULT_RATES: dict[str, float] = {
    "tool_loop": 0.07,
    "retry_storm": 0.06,
    "idle_run": 0.14,
    "prompt_bloat": 0.07,
    "oversized_model": 0.08,
}

IDLE_TEXTS = (
    "Nothing happened since the last check. HEARTBEAT_OK",
    "No new items. Nothing to report.",
    "All quiet, no action needed.",
)
IDLE_NEUTRAL_TEXTS = ("Checked. Status unchanged.", "Done for now.")


@dataclass
class GeneratorConfig:
    n_runs: int = 600
    seed: int = 42
    start: datetime = field(default_factory=lambda: datetime(2026, 10, 1, tzinfo=timezone.utc))
    rates: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RATES))
    style: str = "simple"  # "simple" or "otel"


@dataclass
class _Step:
    kind: str
    name: str
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    tool_name: str | None = None
    tool_args: Any = None
    status: str = "ok"
    error_type: str | None = None
    output_text: str | None = None
    latency_ms: float = 0.0
    gap_ms: float = 40.0


class _Builder:
    def __init__(self, rng: random.Random, profile: AgentProfile) -> None:
        self.rng = rng
        self.p = profile
        self.steps: list[_Step] = []
        self.ctx = rng.randint(*profile.base_input)

    def llm(
        self,
        model: str,
        inp: int,
        out: int,
        name: str = "chat",
        text: str | None = None,
        status: str = "ok",
        error_type: str | None = None,
        gap: float | None = None,
    ) -> None:
        latency = 250 + out * self.rng.uniform(12, 28) + inp * 0.015
        if status == "error":
            latency = self.rng.uniform(80, 400) if error_type == "RateLimitError" else 30000.0
        self.steps.append(
            _Step(
                "llm",
                name,
                model,
                inp,
                out,
                status=status,
                error_type=error_type,
                output_text=text,
                latency_ms=round(latency, 1),
                gap_ms=self.rng.uniform(15, 120) if gap is None else gap,
            )
        )

    def tool(
        self,
        tool: str,
        args: Any,
        status: str = "ok",
        error_type: str | None = None,
        gap: float | None = None,
    ) -> None:
        latency = self.rng.lognormvariate(5.5, 0.6)
        if status == "error" and error_type == "TimeoutError":
            latency = 10000.0
        self.steps.append(
            _Step(
                "tool",
                f"tool.{tool}",
                tool_name=tool,
                tool_args=args,
                status=status,
                error_type=error_type,
                latency_ms=round(latency, 1),
                gap_ms=self.rng.uniform(10, 80) if gap is None else gap,
            )
        )

    def unique_args(self, tool: str) -> dict[str, Any]:
        return {"query": f"{tool}-{self.rng.randint(1, 10**9)}"}

    def think(self, lo: int = 40, hi: int = 260) -> int:
        out = self.rng.randint(lo, hi)
        self.llm(self.p.main_model, self.ctx, out)
        return out

    def grow(self, out: int, growth: tuple[int, int] | None = None) -> None:
        self.ctx += out + self.rng.randint(*(growth or self.p.growth))


def _build_run(rng: random.Random, profile: AgentProfile, anomalies: set[str]) -> list[_Step]:
    b = _Builder(rng, profile)
    p = profile

    if p.name == "faq-bot":
        inp = rng.randint(*p.base_input)
        b.llm(p.main_model, inp, rng.randint(20, 220), text="Here is the answer from our FAQ.")
        return b.steps

    if "idle_run" in anomalies:
        if rng.random() < 0.3:
            b.llm(p.small_model, rng.randint(150, 400), rng.randint(2, 8), name="route")
        out = rng.randint(41, 90) if rng.random() < 0.2 else rng.randint(4, 35)
        text = rng.choice(IDLE_TEXTS) if rng.random() < 0.7 else rng.choice(IDLE_NEUTRAL_TEXTS)
        b.llm(p.main_model, rng.randint(1800, 6000), out, text=text)
        return b.steps

    oversized = "oversized_model" in anomalies
    n_trivial = 0
    if oversized:
        n_trivial = 1 if rng.random() < 0.15 else rng.randint(2, 5)

    # routing step: normally on the small model
    router = p.main_model if oversized else p.small_model
    b.llm(router, rng.randint(200, 500), rng.randint(3, 15), name="route")
    trivial_left = n_trivial - 1 if oversized else 0

    bloat = "prompt_bloat" in anomalies
    growth = None
    n_steps = rng.randint(*p.steps)
    if bloat:
        growth = (1500, 4000) if rng.random() < 0.85 else (500, 1000)
        n_steps = max(n_steps, rng.randint(4, 7))

    if "tool_loop" in anomalies and "retry_storm" in anomalies:
        n_steps = max(n_steps, 2)
    loop_at = rng.randrange(n_steps) if "tool_loop" in anomalies else -1
    storm_at = -1
    if "retry_storm" in anomalies:
        storm_at = rng.choice([i for i in range(n_steps) if i != loop_at])
    benign_poll = p.name == "ops-heartbeat" and rng.random() < 0.06
    benign_trivial = 0
    if not oversized and p.main_model != p.small_model:
        r = rng.random()
        benign_trivial = 2 if r < 0.03 else (1 if r < 0.18 else 0)

    for i in range(n_steps):
        tool = rng.choice(p.tools)
        if trivial_left > 0 and rng.random() < 0.6:
            b.llm(p.main_model, rng.randint(200, 550), rng.randint(3, 40), name="classify")
            trivial_left -= 1
        if benign_trivial > 0 and rng.random() < 0.5:
            b.llm(p.main_model, rng.randint(250, 550), rng.randint(5, 40), name="format")
            benign_trivial -= 1
        out = b.think()
        if i == loop_at:
            repeats = 2 if rng.random() < 0.15 else rng.randint(3, 8)
            args = b.unique_args(tool)
            b.tool(tool, args)
            for _ in range(repeats - 1):
                b.grow(out, (80, 300))
                out = b.think(20, 80)
                b.tool(tool, args)
        elif i == storm_at:
            failures = 2 if rng.random() < 0.15 else rng.randint(3, 7)
            if rng.random() < 0.7:
                error = rng.choice(("TimeoutError", "HTTP 503", "ConnectionError"))
                args = b.unique_args(tool)
                for k in range(failures):
                    if k:
                        b.llm(p.main_model, b.ctx, rng.randint(15, 50), gap=rng.uniform(20, 300))
                    b.tool(tool, args, status="error", error_type=error, gap=rng.uniform(20, 300))
                if rng.random() < 0.6:
                    b.tool(tool, args, gap=rng.uniform(20, 300))
            else:
                error = rng.choice(("RateLimitError", "Timeout"))
                for _ in range(failures):
                    billed = b.ctx if error == "Timeout" else 0
                    b.llm(
                        p.main_model,
                        billed,
                        0,
                        status="error",
                        error_type=error,
                        gap=rng.uniform(20, 300),
                    )
                b.llm(p.main_model, b.ctx, rng.randint(20, 80))
                b.tool(tool, b.unique_args(tool))
        elif benign_poll and i == 0:
            args = {"service": "payments-api"}
            for _ in range(3):
                b.tool("check_status", args, gap=rng.uniform(800, 2000))
        else:
            args = b.unique_args(tool)
            if rng.random() < 0.12:  # benign transient error, retried with backoff
                for _ in range(1 if rng.random() < 0.7 else 2):
                    b.tool(tool, args, status="error", error_type="HTTP 503")
                b.tool(tool, args, gap=rng.uniform(500, 2000))
            else:
                b.tool(tool, args)
        b.grow(out, growth)

    while trivial_left > 0:
        b.llm(p.main_model, rng.randint(200, 550), rng.randint(3, 40), name="classify")
        trivial_left -= 1
    b.llm(p.main_model, b.ctx, rng.randint(80, 400), text="Final answer delivered.")
    return b.steps


def _hex(rng: random.Random, bits: int) -> str:
    return f"{rng.getrandbits(bits):0{bits // 4}x}"


def _iso(ts: datetime) -> str:
    return ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _record(span: dict[str, Any], style: str) -> dict[str, Any]:
    if style == "simple":
        return {k: v for k, v in span.items() if v is not None}
    op = {"llm": "chat", "tool": "execute_tool", "agent": "invoke_agent"}[span["kind"]]
    attrs: dict[str, Any] = {"gen_ai.operation.name": op, "gen_ai.agent.name": span["agent"]}
    if span["kind"] == "llm":
        attrs["gen_ai.request.model"] = span["model"]
        attrs["gen_ai.usage.input_tokens"] = span["input_tokens"]
        attrs["gen_ai.usage.output_tokens"] = span["output_tokens"]
    if span["kind"] == "tool":
        attrs["gen_ai.tool.name"] = span["tool_name"]
        attrs["gen_ai.tool.call.arguments"] = json.dumps(span["tool_args"], sort_keys=True)
    if span.get("error_type"):
        attrs["error.type"] = span["error_type"]
    if span.get("output_text"):
        attrs["gen_ai.completion"] = span["output_text"]
    return {
        "trace_id": span["trace_id"],
        "span_id": span["span_id"],
        "parent_id": span["parent_id"],
        "name": span["name"],
        "start_time": span["start_time"],
        "end_time": span["end_time"],
        "status_code": "ERROR" if span["status"] == "error" else "OK",
        "attributes": attrs,
    }


def generate(
    config: GeneratorConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Return ``(span_records, labels)`` where labels maps trace_id -> anomaly types."""
    config = config or GeneratorConfig()
    rng = random.Random(config.seed)
    weights = [p.weight for p in PROFILES]
    records: list[dict[str, Any]] = []
    labels: dict[str, list[str]] = {}
    clock = config.start
    for _ in range(config.n_runs):
        profile = rng.choices(PROFILES, weights=weights)[0]
        anomalies = {a for a in profile.eligible if rng.random() < config.rates.get(a, 0.0)}
        if "idle_run" in anomalies:
            anomalies = {"idle_run"}  # an idle run has no tool steps to loop/retry/bloat
        steps = _build_run(rng, profile, anomalies)
        trace_id = _hex(rng, 128)
        root_id = _hex(rng, 64)
        clock += timedelta(seconds=rng.randint(30, 900))
        cursor = clock + timedelta(milliseconds=5)
        spans: list[dict[str, Any]] = []
        for step in steps:
            cursor += timedelta(milliseconds=step.gap_ms)
            end = cursor + timedelta(milliseconds=step.latency_ms)
            spans.append(
                {
                    "trace_id": trace_id,
                    "span_id": _hex(rng, 64),
                    "parent_id": root_id,
                    "kind": step.kind,
                    "name": step.name,
                    "agent": profile.name,
                    "model": step.model,
                    "input_tokens": step.input_tokens if step.kind == "llm" else None,
                    "output_tokens": step.output_tokens if step.kind == "llm" else None,
                    "latency_ms": step.latency_ms,
                    "status": step.status,
                    "tool_name": step.tool_name,
                    "tool_args": step.tool_args,
                    "error_type": step.error_type,
                    "output_text": step.output_text,
                    "start_time": _iso(cursor),
                    "end_time": _iso(end),
                }
            )
            cursor = end
        root_end = cursor + timedelta(milliseconds=5)
        root = {
            "trace_id": trace_id,
            "span_id": root_id,
            "parent_id": None,
            "kind": "agent",
            "name": f"agent.{profile.name}",
            "agent": profile.name,
            "model": None,
            "input_tokens": None,
            "output_tokens": None,
            "latency_ms": round((root_end - clock).total_seconds() * 1000.0, 1),
            "status": "ok",
            "tool_name": None,
            "tool_args": None,
            "error_type": None,
            "output_text": None,
            "start_time": _iso(clock),
            "end_time": _iso(root_end),
        }
        records.append(_record(root, config.style))
        records.extend(_record(s, config.style) for s in spans)
        labels[trace_id] = sorted(anomalies)
        clock = root_end
    return records, labels


def write_dataset(
    records: list[dict[str, Any]],
    labels: dict[str, list[str]],
    traces_path: str | Path,
    labels_path: str | Path | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    traces_path = Path(traces_path)
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    with open(traces_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    if labels_path is not None:
        labels_path = Path(labels_path)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": meta or {}, "labels": labels}
        labels_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_labels(path: str | Path) -> dict[str, list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("labels", data)
