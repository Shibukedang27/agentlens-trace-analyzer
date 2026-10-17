from __future__ import annotations

import gzip
import json

import pytest

from agentlens.parser import TraceParseError, group_runs, load_spans, parse_lines, parse_span

BASE = {
    "trace_id": "t1",
    "span_id": "s1",
    "parent_id": "root",
    "kind": "llm",
    "model": "orion-mini",
    "input_tokens": 120,
    "output_tokens": 30,
    "start_time": "2026-10-01T00:00:00Z",
    "end_time": "2026-10-01T00:00:01.5Z",
}


def test_parse_simple_span() -> None:
    span = parse_span(BASE)
    assert span.kind == "llm"
    assert span.input_tokens == 120 and span.output_tokens == 30
    assert span.latency_ms == pytest.approx(1500.0)
    assert span.status == "ok"
    assert span.total_tokens == 150


def test_parse_otel_genai_attributes() -> None:
    record = {
        "trace_id": "t1",
        "span_id": "s2",
        "start_time": "2026-10-01T00:00:00Z",
        "end_time": "2026-10-01T00:00:00.200Z",
        "status_code": "ERROR",
        "attributes": {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "web_search",
            "gen_ai.tool.call.arguments": '{"q": "hi"}',
            "gen_ai.agent.name": "research-agent",
            "error.type": "TimeoutError",
        },
    }
    span = parse_span(record)
    assert span.kind == "tool"
    assert span.tool_name == "web_search"
    assert span.tool_args == {"q": "hi"}
    assert span.agent == "research-agent"
    assert span.is_error and span.error_type == "TimeoutError"


def test_otel_chat_operation_maps_to_llm() -> None:
    record = {**BASE, "kind": None, "attributes": {"gen_ai.operation.name": "chat"}}
    record.pop("kind")
    assert parse_span(record).kind == "llm"


def test_epoch_milliseconds_and_latency_fallback() -> None:
    record = {**BASE, "start_time": 1790812800000, "latency_ms": 250}
    record.pop("end_time")
    span = parse_span(record)
    assert span.start_time.year == 2026
    assert (span.end_time - span.start_time).total_seconds() == pytest.approx(0.25)


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"trace_id": None}, "trace_id"),
        ({"kind": "banana"}, "unknown span kind"),
        ({"input_tokens": -5}, ">= 0"),
        ({"input_tokens": "lots"}, "integer"),
        ({"start_time": "yesterday"}, "timestamp"),
    ],
)
def test_invalid_records_raise(patch: dict, message: str) -> None:
    with pytest.raises(TraceParseError, match=message):
        parse_span({**BASE, **patch})


def test_parse_lines_collects_errors_and_skips_blank() -> None:
    lines = [json.dumps(BASE), "", "{not json", json.dumps({**BASE, "span_id": "s9"})]
    result = parse_lines(lines)
    assert len(result.spans) == 2
    assert len(result.errors) == 1 and result.errors[0].startswith("line 3")


def test_parse_lines_strict_raises() -> None:
    with pytest.raises(TraceParseError, match="line 1"):
        parse_lines(["{broken"], strict=True)


def test_group_runs_propagates_agent_and_sorts() -> None:
    root = {
        **BASE,
        "span_id": "root",
        "parent_id": None,
        "kind": "agent",
        "agent": "bot",
        "start_time": "2026-10-01T00:00:00Z",
    }
    late = {
        **BASE,
        "span_id": "b",
        "start_time": "2026-10-01T00:00:05Z",
        "end_time": "2026-10-01T00:00:06Z",
    }
    early = {**BASE, "span_id": "a", "start_time": "2026-10-01T00:00:01Z"}
    other = {**BASE, "trace_id": "t2", "span_id": "x"}
    runs = group_runs(parse_span(r) for r in (late, root, early, other))
    by_id = {r.trace_id: r for r in runs}
    assert [s.span_id for s in by_id["t1"].spans] == ["root", "a", "b"]
    assert all(s.agent == "bot" for s in by_id["t1"].spans)
    assert by_id["t2"].agent == "unknown"


def test_load_gzip_file(tmp_path) -> None:
    path = tmp_path / "traces.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(BASE) + "\n")
    result = load_spans(path)
    assert len(result.spans) == 1 and result.lines_read == 1


@pytest.mark.parametrize(
    "stamp, micros",
    [
        ("2026-10-01T00:00:01.5Z", 500000),
        ("2026-10-01T00:00:01.123456789Z", 123456),
        ("2026-10-01T05:30:01.25+05:30", 250000),
    ],
)
def test_iso_fractions_any_precision(stamp: str, micros: int) -> None:
    span = parse_span({**BASE, "start_time": stamp, "end_time": None, "latency_ms": 1})
    assert span.start_time.microsecond == micros
    assert span.start_time.utcoffset() is not None
