"""Explainable, rule-based waste detectors.

Every detector looks at one :class:`Run` and returns zero or more
:class:`Finding` objects. A finding carries the evidence (which spans, what
threshold was crossed) and an *estimated* saving: the cost that would not
have been spent if the pattern were fixed. Savings use the configured price
table, so they are only as real as the prices you give it.
"""

from __future__ import annotations

import itertools
import statistics
from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar

from agentlens.metrics import main_thread, span_cost, tool_signature
from agentlens.models import Run, Span
from agentlens.pricing import PriceTable

DEFAULT_IDLE_MARKERS: tuple[str, ...] = (
    "nothing happened",
    "nothing to report",
    "nothing new",
    "no new",
    "no action",
    "no changes",
    "heartbeat_ok",
    "all quiet",
)


@dataclass
class DetectorConfig:
    """Thresholds for all detectors. Override any field from YAML or code."""

    loop_min_repeats: int = 3
    retry_min_consecutive_failures: int = 3
    idle_max_output_tokens: int = 40
    idle_min_input_tokens: int = 300
    idle_markers: tuple[str, ...] = DEFAULT_IDLE_MARKERS
    bloat_min_llm_calls: int = 4
    bloat_min_growth_ratio: float = 2.5
    bloat_min_increasing_frac: float = 0.8
    bloat_removable_frac: float = 0.5
    trivial_max_input_tokens: int = 600
    trivial_max_output_tokens: int = 50
    oversized_min_calls: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DetectorConfig:
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown detector settings: {sorted(unknown)}")
        values = dict(data)
        if "idle_markers" in values:
            values["idle_markers"] = tuple(str(m).lower() for m in values["idle_markers"])
        return cls(**values)


@dataclass
class Finding:
    detector: str
    trace_id: str
    agent: str
    severity: str
    title: str
    evidence: str
    span_ids: list[str] = field(default_factory=list)
    wasted_tokens: int = 0
    est_savings: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decision_span(run: Run, span: Span) -> Span | None:
    """The LLM call that most recently started before ``span`` (the one that chose it)."""
    best: Span | None = None
    for candidate in run.llm_spans:
        if candidate.start_time <= span.start_time and candidate.span_id != span.span_id:
            best = candidate
        elif candidate.start_time > span.start_time:
            break
    return best


def _waste_of(run: Run, spans: list[Span], prices: PriceTable) -> tuple[int, float, list[str]]:
    """Tokens/cost of ``spans`` plus the LLM calls that decided them (deduplicated)."""
    seen: dict[str, Span] = {}
    for span in spans:
        if span.kind == "llm":
            seen[span.span_id] = span
        else:
            decision = _decision_span(run, span)
            if decision is not None:
                seen[decision.span_id] = decision
    tokens = sum(s.total_tokens for s in seen.values())
    cost = sum(span_cost(s, prices) for s in seen.values())
    return tokens, cost, list(seen)


class Detector:
    name: ClassVar[str] = "base"
    label: ClassVar[str] = "Base"
    description: ClassVar[str] = ""

    def detect(self, run: Run, prices: PriceTable, config: DetectorConfig) -> list[Finding]:
        raise NotImplementedError


class ToolLoopDetector(Detector):
    name = "tool_loop"
    label = "Tool-call loop"
    description = "Same tool succeeded with identical arguments >= N times in one run."

    def detect(self, run: Run, prices: PriceTable, config: DetectorConfig) -> list[Finding]:
        groups: dict[str, list[Span]] = {}
        for span in run.tool_spans:
            if not span.is_error:  # failed repeats are the retry-storm detector's job
                groups.setdefault(tool_signature(span), []).append(span)
        findings = []
        for sig, spans in groups.items():
            if len(spans) < config.loop_min_repeats:
                continue
            repeats = spans[1:]
            tokens, cost, decision_ids = _waste_of(run, repeats, prices)
            tool, _, args = sig.partition("|")
            findings.append(
                Finding(
                    detector=self.name,
                    trace_id=run.trace_id,
                    agent=run.agent,
                    severity="high" if len(spans) >= 6 else "medium",
                    title=f"{tool} called {len(spans)}x with identical arguments",
                    evidence=(
                        f"tool={tool} args={args[:80]} repeated {len(spans)} times "
                        f"(threshold {config.loop_min_repeats}); {len(repeats)} calls and "
                        f"{len(decision_ids)} deciding LLM calls are redundant"
                    ),
                    span_ids=[s.span_id for s in spans],
                    wasted_tokens=tokens,
                    est_savings=cost,
                )
            )
        return findings


class RetryStormDetector(Detector):
    name = "retry_storm"
    label = "Retry storm"
    description = "N or more consecutive failed attempts against the same tool or model."

    def detect(self, run: Run, prices: PriceTable, config: DetectorConfig) -> list[Finding]:
        streaks: dict[str, list[Span]] = {}
        worst: dict[str, list[Span]] = {}
        for span in run.spans:
            if span.kind == "agent":
                continue
            target = (
                f"tool:{span.tool_name or span.name}"
                if span.kind == "tool"
                else f"llm:{span.model}"
            )
            if span.is_error:
                streaks.setdefault(target, []).append(span)
                if len(streaks[target]) > len(worst.get(target, [])):
                    worst[target] = list(streaks[target])
            else:
                streaks[target] = []
        findings = []
        for target, failed in worst.items():
            if len(failed) < config.retry_min_consecutive_failures:
                continue
            tokens, cost, _ = _waste_of(run, failed[1:], prices)
            gaps = [
                (b.start_time - a.end_time).total_seconds() * 1000.0
                for a, b in itertools.pairwise(failed)
            ]
            median_gap = statistics.median(gaps) if gaps else 0.0
            errors = sorted({s.error_type for s in failed if s.error_type})
            findings.append(
                Finding(
                    detector=self.name,
                    trace_id=run.trace_id,
                    agent=run.agent,
                    severity="high" if len(failed) >= 5 else "medium",
                    title=f"{len(failed)} consecutive failures on {target}",
                    evidence=(
                        f"{len(failed)} failed attempts in a row on {target} "
                        f"(threshold {config.retry_min_consecutive_failures}); "
                        f"median gap between attempts {median_gap:.0f} ms"
                        + (f"; errors: {', '.join(errors)}" if errors else "")
                        + ("; looks like no backoff" if median_gap < 500 else "")
                    ),
                    span_ids=[s.span_id for s in failed],
                    wasted_tokens=tokens,
                    est_savings=cost,
                )
            )
        return findings


class IdleRunDetector(Detector):
    name = "idle_run"
    label = "Idle / no-op run"
    description = "Run spends tokens but calls no tools and returns (almost) nothing."

    def detect(self, run: Run, prices: PriceTable, config: DetectorConfig) -> list[Finding]:
        llm = run.llm_spans
        if not llm or run.tool_spans:
            return []
        inp = sum(s.input_tokens for s in llm)
        out = sum(s.output_tokens for s in llm)
        if inp < config.idle_min_input_tokens:
            return []
        texts = " ".join((s.output_text or "") for s in run.spans).lower()
        marker = next((m for m in config.idle_markers if m in texts), None)
        if out > config.idle_max_output_tokens and marker is None:
            return []
        cost = sum(span_cost(s, prices) for s in llm)
        reason = f"output says '{marker}'" if marker else f"only {out} output tokens"
        return [
            Finding(
                detector=self.name,
                trace_id=run.trace_id,
                agent=run.agent,
                severity="medium",
                title=f"Idle run: {inp + out} tokens, no tool calls",
                evidence=(
                    f"{len(llm)} LLM call(s), {inp} input tokens, 0 tool calls, {reason}. "
                    "A cheap non-LLM pre-check (e.g. 'anything changed since last run?') "
                    "could skip this run entirely"
                ),
                span_ids=[s.span_id for s in llm],
                wasted_tokens=inp + out,
                est_savings=cost,
            )
        ]


class PromptBloatDetector(Detector):
    name = "prompt_bloat"
    label = "Prompt bloat"
    description = "Input tokens keep growing across LLM steps (context never compacted)."

    def detect(self, run: Run, prices: PriceTable, config: DetectorConfig) -> list[Finding]:
        llm = main_thread(run, config.trivial_max_input_tokens)
        if len(llm) < config.bloat_min_llm_calls:
            return []
        inputs = [s.input_tokens for s in llm]
        ratio = inputs[-1] / inputs[0]
        increases = sum(1 for a, b in itertools.pairwise(inputs) if b > a)
        frac = increases / (len(inputs) - 1)
        if ratio < config.bloat_min_growth_ratio or frac < config.bloat_min_increasing_frac:
            return []
        wasted = 0
        cost = 0.0
        for span in llm[1:]:
            excess = int(max(0, span.input_tokens - inputs[0]) * config.bloat_removable_frac)
            wasted += excess
            cost += prices.price_for(span.model).cost(excess, 0)
        return [
            Finding(
                detector=self.name,
                trace_id=run.trace_id,
                agent=run.agent,
                severity="high" if ratio >= 5 else "medium",
                title=f"Prompt grew {ratio:.1f}x over {len(llm)} LLM calls",
                evidence=(
                    f"input tokens {inputs[0]} -> {inputs[-1]} ({ratio:.1f}x, threshold "
                    f"{config.bloat_min_growth_ratio}x), grew on {increases}/{len(inputs) - 1} steps. "
                    f"Savings assume compaction removes {config.bloat_removable_frac:.0%} of the growth"
                ),
                span_ids=[s.span_id for s in llm],
                wasted_tokens=wasted,
                est_savings=cost,
            )
        ]


class OversizedModelDetector(Detector):
    name = "oversized_model"
    label = "Oversized model"
    description = "Large-tier model used for trivial steps (tiny prompt and tiny output)."

    def detect(self, run: Run, prices: PriceTable, config: DetectorConfig) -> list[Finding]:
        trivial = [
            s
            for s in run.llm_spans
            if not s.is_error
            and prices.price_for(s.model).tier == "large"
            and s.input_tokens <= config.trivial_max_input_tokens
            and s.output_tokens <= config.trivial_max_output_tokens
        ]
        if len(trivial) < config.oversized_min_calls:
            return []
        saving = 0.0
        targets = set()
        for span in trivial:
            cheaper = prices.downgrade_price(span.model)
            if cheaper is None:
                continue
            targets.add(prices.price_for(span.model).downgrade_to)
            saving += span_cost(span, prices) - cheaper.cost(span.input_tokens, span.output_tokens)
        models = sorted({s.model or "unknown" for s in trivial})
        return [
            Finding(
                detector=self.name,
                trace_id=run.trace_id,
                agent=run.agent,
                severity="low",
                title=f"{len(trivial)} trivial steps on large model {', '.join(models)}",
                evidence=(
                    f"{len(trivial)} calls with <= {config.trivial_max_input_tokens} input and "
                    f"<= {config.trivial_max_output_tokens} output tokens ran on a large-tier model; "
                    f"routing them to {', '.join(sorted(t for t in targets if t)) or 'a small model'} "
                    "would cost less"
                ),
                span_ids=[s.span_id for s in trivial],
                wasted_tokens=0,
                est_savings=max(0.0, saving),
            )
        ]


DETECTORS: tuple[Detector, ...] = (
    ToolLoopDetector(),
    RetryStormDetector(),
    IdleRunDetector(),
    PromptBloatDetector(),
    OversizedModelDetector(),
)

DETECTOR_NAMES: tuple[str, ...] = tuple(d.name for d in DETECTORS)


def run_detectors(
    runs: list[Run],
    prices: PriceTable,
    config: DetectorConfig | None = None,
    detectors: tuple[Detector, ...] = DETECTORS,
) -> list[Finding]:
    config = config or DetectorConfig()
    findings: list[Finding] = []
    for run in runs:
        for detector in detectors:
            findings.extend(detector.detect(run, prices, config))
    return findings
