"""Performance metrics collection for DataVault.

Collects latency, token usage, and success rates
for monitoring agent performance.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from collections import defaultdict

from src.utils.logging import get_logger

logger = get_logger("evaluation.metrics")


@dataclass
class MetricPoint:
    """A single metric measurement."""

    name: str
    value: float
    timestamp: datetime
    tags: dict[str, str] = field(default_factory=dict)
    unit: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "unit": self.unit,
        }


@dataclass
class MetricsSummary:
    """Summary of collected metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    avg_latency_ms: float = 0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0
    total_tokens: int = 0
    total_sql_queries: int = 0
    total_rows_returned: int = 0
    by_agent: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms if self.min_latency_ms != float("inf") else 0,
            "max_latency_ms": self.max_latency_ms,
            "total_tokens": self.total_tokens,
            "total_sql_queries": self.total_sql_queries,
            "total_rows_returned": self.total_rows_returned,
            "by_agent": self.by_agent,
        }


class MetricsCollector:
    """Collects and aggregates performance metrics.

    Provides methods for tracking latency, tokens, and
    success/failure rates.
    """

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._metrics: list[MetricPoint] = []
        self._timers: dict[str, float] = {}
        self._counters: dict[str, int] = defaultdict(int)
        self._agent_metrics: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "success": 0, "latency_sum": 0}
        )

        logger.info("MetricsCollector initialized")

    def start_timer(self, name: str) -> None:
        """Start a timer.

        Args:
            name: Timer name.
        """
        self._timers[name] = time.perf_counter()

    def stop_timer(self, name: str, tags: dict[str, str] | None = None) -> float:
        """Stop a timer and record the duration.

        Args:
            name: Timer name.
            tags: Optional tags.

        Returns:
            Duration in milliseconds.
        """
        if name not in self._timers:
            return 0.0

        duration_ms = (time.perf_counter() - self._timers[name]) * 1000
        del self._timers[name]

        self.record(f"{name}_latency", duration_ms, tags, unit="ms")
        return duration_ms

    def record(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
        unit: str = "",
    ) -> None:
        """Record a metric.

        Args:
            name: Metric name.
            value: Metric value.
            tags: Optional tags.
            unit: Unit of measurement.
        """
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            unit=unit,
        )
        self._metrics.append(point)
        logger.debug(f"Recorded metric: {name}={value}{unit}")

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter.

        Args:
            name: Counter name.
            value: Increment value.
        """
        self._counters[name] += value

    def record_request(
        self,
        success: bool,
        latency_ms: float,
        agent: str | None = None,
        tokens: int = 0,
        sql_queries: int = 0,
        rows_returned: int = 0,
    ) -> None:
        """Record a complete request.

        Args:
            success: Whether request succeeded.
            latency_ms: Request latency in ms.
            agent: Agent that handled request.
            tokens: Tokens used.
            sql_queries: SQL queries executed.
            rows_returned: Rows returned.
        """
        self.increment("total_requests")
        if success:
            self.increment("successful_requests")
        else:
            self.increment("failed_requests")

        self.record("request_latency", latency_ms, unit="ms")
        self.record("tokens_used", tokens)
        self.record("sql_queries", sql_queries)
        self.record("rows_returned", rows_returned)

        if agent:
            self._agent_metrics[agent]["count"] += 1
            if success:
                self._agent_metrics[agent]["success"] += 1
            self._agent_metrics[agent]["latency_sum"] += latency_ms

    def record_llm_call(
        self,
        model: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record an LLM call.

        Args:
            model: Model name.
            latency_ms: Call latency.
            prompt_tokens: Prompt tokens.
            completion_tokens: Completion tokens.
        """
        total_tokens = prompt_tokens + completion_tokens

        self.record(
            "llm_latency",
            latency_ms,
            tags={"model": model},
            unit="ms",
        )
        self.record(
            "llm_tokens",
            total_tokens,
            tags={"model": model},
        )
        self.increment("total_tokens", total_tokens)

    def record_sql_execution(
        self,
        success: bool,
        latency_ms: float,
        rows: int = 0,
    ) -> None:
        """Record a SQL execution.

        Args:
            success: Whether query succeeded.
            latency_ms: Execution latency.
            rows: Rows returned.
        """
        self.increment("sql_queries")
        if success:
            self.increment("sql_success")
            self.increment("rows_returned", rows)
        else:
            self.increment("sql_errors")

        self.record(
            "sql_latency",
            latency_ms,
            tags={"success": str(success)},
            unit="ms",
        )

    def get_summary(self) -> MetricsSummary:
        """Get metrics summary.

        Returns:
            MetricsSummary with aggregated data.
        """
        latencies = [
            m.value for m in self._metrics
            if m.name == "request_latency"
        ]

        summary = MetricsSummary(
            total_requests=self._counters["total_requests"],
            successful_requests=self._counters["successful_requests"],
            failed_requests=self._counters["failed_requests"],
            total_latency_ms=sum(latencies),
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            total_tokens=self._counters["total_tokens"],
            total_sql_queries=self._counters["sql_queries"],
            total_rows_returned=self._counters["rows_returned"],
        )

        # Per-agent metrics
        for agent, data in self._agent_metrics.items():
            summary.by_agent[agent] = {
                "count": data["count"],
                "success": data["success"],
                "success_rate": data["success"] / max(data["count"], 1),
                "avg_latency_ms": data["latency_sum"] / max(data["count"], 1),
            }

        return summary

    def get_metrics(
        self,
        name: str | None = None,
        limit: int = 1000,
    ) -> list[MetricPoint]:
        """Get recorded metrics.

        Args:
            name: Filter by metric name.
            limit: Maximum metrics to return.

        Returns:
            List of metric points.
        """
        metrics = self._metrics
        if name:
            metrics = [m for m in metrics if m.name == name]

        return metrics[-limit:]

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._timers.clear()
        self._counters.clear()
        self._agent_metrics.clear()
        logger.info("Metrics reset")

    def export_to_json(self) -> dict[str, Any]:
        """Export all metrics as JSON-compatible dict.

        Returns:
            Dict with all metrics data.
        """
        return {
            "summary": self.get_summary().to_dict(),
            "metrics": [m.to_dict() for m in self._metrics],
            "counters": dict(self._counters),
        }


# Global collector instance
_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector.

    Returns:
        MetricsCollector instance.
    """
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
