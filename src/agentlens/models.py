"""Core data model: spans and runs (one run = one trace)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

SPAN_KINDS: tuple[str, ...] = ("agent", "llm", "tool")


@dataclass
class Span:
    """A single unit of work inside an agent run.

    Field names loosely follow the OpenTelemetry GenAI semantic conventions;
    see :mod:`agentlens.parser` for the accepted aliases.
    """

    trace_id: str
    span_id: str
    parent_id: str | None
    kind: str
    name: str
    start_time: datetime
    end_time: datetime
    latency_ms: float
    status: str = "ok"
    agent: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    tool_name: str | None = None
    tool_args: Any = None
    output_text: str | None = None
    error_type: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.status == "error"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Run:
    """All spans of one trace, sorted by start time."""

    trace_id: str
    agent: str
    spans: list[Span]

    @property
    def llm_spans(self) -> list[Span]:
        return [s for s in self.spans if s.kind == "llm"]

    @property
    def tool_spans(self) -> list[Span]:
        return [s for s in self.spans if s.kind == "tool"]

    @property
    def root(self) -> Span | None:
        roots = [s for s in self.spans if s.parent_id is None]
        return roots[0] if roots else None

    @property
    def start(self) -> datetime:
        return min(s.start_time for s in self.spans)

    @property
    def end(self) -> datetime:
        return max(s.end_time for s in self.spans)

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start).total_seconds() * 1000.0

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.spans)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.spans)
