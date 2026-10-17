"""agentlens: find the waste in LLM agent traces."""

from agentlens.analyzer import AnalysisResult, analyze_file, analyze_runs
from agentlens.detectors import DetectorConfig, Finding
from agentlens.models import Run, Span
from agentlens.parser import group_runs, load_runs, load_spans
from agentlens.pricing import PriceTable

__version__ = "0.1.0"

__all__ = [
    "AnalysisResult",
    "DetectorConfig",
    "Finding",
    "PriceTable",
    "Run",
    "Span",
    "__version__",
    "analyze_file",
    "analyze_runs",
    "group_runs",
    "load_runs",
    "load_spans",
]
