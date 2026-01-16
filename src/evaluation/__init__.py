"""Evaluation module for DataVault.

Provides observability via Langfuse, metrics collection,
benchmarking, and evaluation utilities.
"""

from src.evaluation.langfuse_client import (
    LangfuseWrapper,
    TraceContext,
    SpanContext,
    TraceData,
    SpanData,
    get_langfuse,
    create_langfuse,
)
from src.evaluation.metrics import (
    MetricsCollector,
    MetricPoint,
    MetricsSummary,
    get_metrics_collector,
)
from src.evaluation.benchmarks import (
    BenchmarkSuite,
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkReport,
    create_benchmark_suite,
)
from src.evaluation.evaluator import (
    EvaluationResult,
    EvaluationScore,
    SQLEvaluator,
    RoutingEvaluator,
    ResponseEvaluator,
    evaluate_response,
)

__all__ = [
    # Langfuse
    "LangfuseWrapper",
    "TraceContext",
    "SpanContext",
    "TraceData",
    "SpanData",
    "get_langfuse",
    "create_langfuse",
    # Metrics
    "MetricsCollector",
    "MetricPoint",
    "MetricsSummary",
    "get_metrics_collector",
    # Benchmarks
    "BenchmarkSuite",
    "BenchmarkCase",
    "BenchmarkResult",
    "BenchmarkReport",
    "create_benchmark_suite",
    # Evaluator
    "EvaluationResult",
    "EvaluationScore",
    "SQLEvaluator",
    "RoutingEvaluator",
    "ResponseEvaluator",
    "evaluate_response",
]
