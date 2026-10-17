from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from agentlens.models import Run, Span
from agentlens.pricing import PriceTable

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


class RunBuilder:
    """Tiny DSL to build runs for detector tests."""

    def __init__(self, agent: str = "test-agent", trace_id: str = "t1") -> None:
        self.agent = agent
        self.trace_id = trace_id
        self.cursor = T0
        self.spans: list[Span] = []
        self.n = 0

    def _add(self, kind: str, name: str, latency: float = 100.0, **kw: Any) -> RunBuilder:
        self.n += 1
        start = self.cursor + timedelta(milliseconds=10)
        end = start + timedelta(milliseconds=latency)
        self.spans.append(
            Span(
                trace_id=self.trace_id,
                span_id=f"s{self.n}",
                parent_id="root",
                kind=kind,
                name=name,
                start_time=start,
                end_time=end,
                latency_ms=latency,
                agent=self.agent,
                **kw,
            )
        )
        self.cursor = end
        return self

    def llm(
        self, model: str = "orion-large", inp: int = 1000, out: int = 100, **kw: Any
    ) -> RunBuilder:
        return self._add("llm", "chat", model=model, input_tokens=inp, output_tokens=out, **kw)

    def tool(self, name: str = "search", args: Any = None, **kw: Any) -> RunBuilder:
        return self._add(
            "tool",
            f"tool.{name}",
            tool_name=name,
            tool_args=args if args is not None else {"q": "x"},
            **kw,
        )

    def build(self) -> Run:
        root = Span(
            trace_id=self.trace_id,
            span_id="root",
            parent_id=None,
            kind="agent",
            name=f"agent.{self.agent}",
            start_time=T0,
            end_time=self.cursor,
            latency_ms=(self.cursor - T0).total_seconds() * 1000,
            agent=self.agent,
        )
        return Run(trace_id=self.trace_id, agent=self.agent, spans=[root, *self.spans])


@pytest.fixture
def builder() -> type[RunBuilder]:
    return RunBuilder


@pytest.fixture
def prices() -> PriceTable:
    return PriceTable.default_table()
