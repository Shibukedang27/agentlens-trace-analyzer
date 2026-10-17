"""Parse JSONL span files into :class:`Span` and :class:`Run` objects.

Each line is one JSON object (one span). Fields can be given with simple names
(``input_tokens``) or OpenTelemetry GenAI attribute names
(``gen_ai.usage.input_tokens``), either at the top level or inside an
``attributes`` object.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agentlens.models import SPAN_KINDS, Run, Span

ALIASES: dict[str, tuple[str, ...]] = {
    "trace_id": ("trace_id", "traceId", "trace.id"),
    "span_id": ("span_id", "spanId", "span.id"),
    "parent_id": ("parent_id", "parent_span_id", "parentSpanId", "parent.id"),
    "kind": ("kind", "span_kind", "type"),
    "operation": ("gen_ai.operation.name", "operation"),
    "name": ("name", "span_name"),
    "agent": ("agent", "agent_name", "gen_ai.agent.name"),
    "model": ("model", "gen_ai.request.model", "gen_ai.response.model", "llm.model"),
    "input_tokens": (
        "input_tokens",
        "gen_ai.usage.input_tokens",
        "prompt_tokens",
        "gen_ai.usage.prompt_tokens",
    ),
    "output_tokens": (
        "output_tokens",
        "gen_ai.usage.output_tokens",
        "completion_tokens",
        "gen_ai.usage.completion_tokens",
    ),
    "latency_ms": ("latency_ms", "duration_ms"),
    "status": ("status", "status_code", "otel.status_code"),
    "tool_name": ("tool_name", "tool", "gen_ai.tool.name"),
    "tool_args": ("tool_args", "arguments", "gen_ai.tool.call.arguments"),
    "start_time": ("start_time", "startTime", "start", "timestamp"),
    "end_time": ("end_time", "endTime", "end"),
    "output_text": ("output_text", "output", "gen_ai.completion"),
    "error_type": ("error_type", "error.type"),
}

OPERATION_TO_KIND = {
    "chat": "llm",
    "text_completion": "llm",
    "generate_content": "llm",
    "completion": "llm",
    "embeddings": "llm",
    "execute_tool": "tool",
    "invoke_agent": "agent",
    "create_agent": "agent",
}

KIND_ALIASES = {"llm": "llm", "tool": "tool", "agent": "agent", "chain": "agent", "model": "llm"}


_FRACTION = re.compile(r"\.(\d+)(?=[+-]\d\d:?\d\d$|$)")


class TraceParseError(ValueError):
    """Raised when a span record cannot be parsed."""


@dataclass
class ParseResult:
    spans: list[Span]
    errors: list[str] = field(default_factory=list)
    lines_read: int = 0


def _lookup(record: dict[str, Any], key: str) -> Any:
    attrs = record.get("attributes") if isinstance(record.get("attributes"), dict) else {}
    for alias in ALIASES[key]:
        if alias in record and record[alias] is not None:
            return record[alias]
        if alias in attrs and attrs[alias] is not None:
            return attrs[alias]
    return None


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1e17:  # nanoseconds (OTLP)
            seconds /= 1e9
        elif seconds > 1e14:  # microseconds
            seconds /= 1e6
        elif seconds > 1e11:  # milliseconds
            seconds /= 1e3
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        # Python 3.10's fromisoformat only accepts 3 or 6 fractional digits
        text = _FRACTION.sub(lambda m: "." + (m.group(1) + "000000")[:6], text)
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise TraceParseError(f"invalid timestamp {value!r}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    raise TraceParseError(f"invalid timestamp {value!r}")


def _normalize_kind(record: dict[str, Any]) -> str:
    raw = _lookup(record, "kind")
    if isinstance(raw, str) and raw.lower() in KIND_ALIASES:
        return KIND_ALIASES[raw.lower()]
    op = _lookup(record, "operation")
    if isinstance(op, str) and op.lower() in OPERATION_TO_KIND:
        return OPERATION_TO_KIND[op.lower()]
    raise TraceParseError(f"unknown span kind {raw!r} (expected one of {SPAN_KINDS})")


def _normalize_status(value: Any) -> str:
    if value is None:
        return "ok"
    text = str(value).lower()
    if text in ("error", "err", "failed", "failure", "status_code_error", "2"):
        return "error"
    return "ok"


def _to_int(value: Any, name: str) -> int:
    if value is None:
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise TraceParseError(f"{name} must be an integer, got {value!r}") from exc
    if number < 0:
        raise TraceParseError(f"{name} must be >= 0, got {number}")
    return number


def parse_span(record: dict[str, Any]) -> Span:
    """Convert one JSON object into a :class:`Span`."""
    if not isinstance(record, dict):
        raise TraceParseError("span record must be a JSON object")
    trace_id = _lookup(record, "trace_id")
    span_id = _lookup(record, "span_id")
    if not trace_id or not span_id:
        raise TraceParseError("span is missing trace_id or span_id")
    kind = _normalize_kind(record)

    start = _parse_time(_lookup(record, "start_time"))
    end = _parse_time(_lookup(record, "end_time"))
    latency = _lookup(record, "latency_ms")
    if start is None:
        raise TraceParseError("span is missing start_time")
    if latency is None:
        latency = (end - start).total_seconds() * 1000.0 if end is not None else 0.0
    latency = float(latency)
    if latency < 0:
        raise TraceParseError(f"latency_ms must be >= 0, got {latency}")
    if end is None:
        end = start + timedelta(milliseconds=latency)

    tool_name = _lookup(record, "tool_name")
    name = _lookup(record, "name") or tool_name or kind
    attrs = record.get("attributes") if isinstance(record.get("attributes"), dict) else {}
    tool_args = _lookup(record, "tool_args")
    if isinstance(tool_args, str):
        with contextlib.suppress(ValueError):
            tool_args = json.loads(tool_args)
    output_text = _lookup(record, "output_text")
    return Span(
        trace_id=str(trace_id),
        span_id=str(span_id),
        parent_id=(str(p) if (p := _lookup(record, "parent_id")) else None),
        kind=kind,
        name=str(name),
        start_time=start,
        end_time=end,
        latency_ms=latency,
        status=_normalize_status(_lookup(record, "status")),
        agent=(str(a) if (a := _lookup(record, "agent")) else None),
        model=(str(m) if (m := _lookup(record, "model")) else None),
        input_tokens=_to_int(_lookup(record, "input_tokens"), "input_tokens"),
        output_tokens=_to_int(_lookup(record, "output_tokens"), "output_tokens"),
        tool_name=str(tool_name) if tool_name else None,
        tool_args=tool_args,
        output_text=str(output_text) if output_text is not None else None,
        error_type=(str(e) if (e := _lookup(record, "error_type")) else None),
        attributes=dict(attrs),
    )


def iter_lines(path: str | Path) -> Iterator[str]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        yield from handle


def parse_lines(lines: Iterable[str], strict: bool = False) -> ParseResult:
    """Parse JSONL lines. Bad lines are collected in ``errors`` unless ``strict``."""
    result = ParseResult(spans=[])
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        result.lines_read += 1
        try:
            record = json.loads(line)
            result.spans.append(parse_span(record))
        except (ValueError, TraceParseError) as exc:
            message = f"line {line_no}: {exc}"
            if strict:
                raise TraceParseError(message) from exc
            result.errors.append(message)
    return result


def load_spans(path: str | Path, strict: bool = False) -> ParseResult:
    """Load spans from a ``.jsonl`` (or ``.jsonl.gz``) file."""
    return parse_lines(iter_lines(path), strict=strict)


def group_runs(spans: Iterable[Span]) -> list[Run]:
    """Group spans by trace id into runs and propagate the agent name."""
    by_trace: dict[str, list[Span]] = defaultdict(list)
    for span in spans:
        by_trace[span.trace_id].append(span)
    runs: list[Run] = []
    for trace_id, trace_spans in by_trace.items():
        trace_spans.sort(key=lambda s: (s.start_time, s.span_id))
        agent = None
        for span in trace_spans:
            if span.parent_id is None and span.agent:
                agent = span.agent
                break
        if agent is None:
            agent = next((s.agent for s in trace_spans if s.agent), None)
        if agent is None:
            root = next((s for s in trace_spans if s.kind == "agent"), None)
            agent = root.name if root else "unknown"
        for span in trace_spans:
            if span.agent is None:
                span.agent = agent
        runs.append(Run(trace_id=trace_id, agent=agent, spans=trace_spans))
    runs.sort(key=lambda r: (r.start, r.trace_id))
    return runs


def load_runs(path: str | Path, strict: bool = False) -> tuple[list[Run], list[str]]:
    """Convenience: load a file and return ``(runs, parse_errors)``."""
    parsed = load_spans(path, strict=strict)
    return group_runs(parsed.spans), parsed.errors
