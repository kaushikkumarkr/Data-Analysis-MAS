"""Benchmark suite for evaluating DataVault agents.

Provides predefined test cases and scoring for
measuring agent performance.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
import json

from src.utils.logging import get_logger

logger = get_logger("evaluation.benchmarks")


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""

    id: str
    name: str
    category: str
    query: str
    expected_type: str | None = None  # Expected task type (clean, analyze, visualize)
    expected_sql_pattern: str | None = None  # Regex pattern for expected SQL
    expected_columns: list[str] | None = None  # Expected columns in result
    timeout_seconds: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass 
class BenchmarkResult:
    """Result of running a benchmark case."""

    case_id: str
    passed: bool
    score: float  # 0.0 to 1.0
    latency_ms: float
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "details": self.details,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    name: str
    timestamp: datetime
    total_cases: int
    passed_cases: int
    failed_cases: int
    avg_score: float
    avg_latency_ms: float
    results: list[BenchmarkResult] = field(default_factory=list)
    by_category: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "pass_rate": self.passed_cases / max(self.total_cases, 1),
            "avg_score": self.avg_score,
            "avg_latency_ms": self.avg_latency_ms,
            "results": [r.to_dict() for r in self.results],
            "by_category": self.by_category,
        }


# Predefined benchmark cases
ROUTING_BENCHMARKS = [
    BenchmarkCase(
        id="route_001",
        name="Cleaning task routing",
        category="routing",
        query="Remove all null values from the data",
        expected_type="clean",
    ),
    BenchmarkCase(
        id="route_002",
        name="Analysis task routing",
        category="routing",
        query="What is the total revenue by region?",
        expected_type="analyze",
    ),
    BenchmarkCase(
        id="route_003",
        name="Visualization task routing",
        category="routing",
        query="Create a bar chart of sales by category",
        expected_type="visualize",
    ),
    BenchmarkCase(
        id="route_004",
        name="Deduplication routing",
        category="routing",
        query="Find and remove duplicate rows",
        expected_type="clean",
    ),
    BenchmarkCase(
        id="route_005",
        name="Aggregation routing",
        category="routing",
        query="Calculate average order value by customer",
        expected_type="analyze",
    ),
]

SQL_BENCHMARKS = [
    BenchmarkCase(
        id="sql_001",
        name="Simple count query",
        category="sql",
        query="How many rows are in the sales table?",
        expected_sql_pattern=r"SELECT\s+COUNT\s*\(",
    ),
    BenchmarkCase(
        id="sql_002",
        name="Group by aggregation",
        category="sql",
        query="Show total sales by region",
        expected_sql_pattern=r"GROUP\s+BY",
    ),
    BenchmarkCase(
        id="sql_003",
        name="Filter query",
        category="sql",
        query="Show sales greater than 1000",
        expected_sql_pattern=r"WHERE.*>.*1000",
    ),
    BenchmarkCase(
        id="sql_004",
        name="Top N query",
        category="sql",
        query="Show the top 5 customers by revenue",
        expected_sql_pattern=r"ORDER\s+BY.*LIMIT\s+5",
    ),
    BenchmarkCase(
        id="sql_005",
        name="Date filter query",
        category="sql",
        query="Show sales from last month",
        expected_sql_pattern=r"WHERE.*date",
    ),
]

VISUALIZATION_BENCHMARKS = [
    BenchmarkCase(
        id="viz_001",
        name="Bar chart request",
        category="visualization",
        query="Create a bar chart of sales by category",
        expected_type="visualize",
    ),
    BenchmarkCase(
        id="viz_002",
        name="Line chart request",
        category="visualization",
        query="Plot revenue trend over time",
        expected_type="visualize",
    ),
    BenchmarkCase(
        id="viz_003",
        name="Pie chart request",
        category="visualization",
        query="Show distribution of sales by region as a pie chart",
        expected_type="visualize",
    ),
]


class BenchmarkSuite:
    """Benchmark suite for evaluating agents.

    Runs predefined test cases and generates reports.
    """

    def __init__(
        self,
        name: str = "DataVault Benchmark Suite",
        cases: list[BenchmarkCase] | None = None,
    ) -> None:
        """Initialize benchmark suite.

        Args:
            name: Suite name.
            cases: Custom benchmark cases. Uses defaults if not provided.
        """
        self.name = name
        self.cases = cases or self._get_default_cases()
        self._results: list[BenchmarkResult] = []

        logger.info(f"BenchmarkSuite initialized with {len(self.cases)} cases")

    def _get_default_cases(self) -> list[BenchmarkCase]:
        """Get default benchmark cases."""
        return ROUTING_BENCHMARKS + SQL_BENCHMARKS + VISUALIZATION_BENCHMARKS

    def add_case(self, case: BenchmarkCase) -> None:
        """Add a benchmark case.

        Args:
            case: Benchmark case to add.
        """
        self.cases.append(case)

    async def run(
        self,
        runner: Callable[[str], Any],
        categories: list[str] | None = None,
    ) -> BenchmarkReport:
        """Run the benchmark suite.

        Args:
            runner: Async function that takes query and returns result dict.
            categories: Filter to specific categories.

        Returns:
            BenchmarkReport with results.
        """
        cases_to_run = self.cases
        if categories:
            cases_to_run = [c for c in self.cases if c.category in categories]

        logger.info(f"Running {len(cases_to_run)} benchmark cases")

        self._results = []

        for case in cases_to_run:
            result = await self._run_case(case, runner)
            self._results.append(result)

        return self._generate_report()

    async def _run_case(
        self,
        case: BenchmarkCase,
        runner: Callable[[str], Any],
    ) -> BenchmarkResult:
        """Run a single benchmark case.

        Args:
            case: Benchmark case.
            runner: Runner function.

        Returns:
            BenchmarkResult.
        """
        start_time = time.perf_counter()
        error = None
        details = {}
        score = 0.0

        try:
            result = await runner(case.query)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Evaluate result
            score, details = self._evaluate_result(case, result)

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            error = str(e)
            logger.warning(f"Benchmark {case.id} failed: {e}")

        passed = score >= 0.5 and error is None

        return BenchmarkResult(
            case_id=case.id,
            passed=passed,
            score=score,
            latency_ms=latency_ms,
            error=error,
            details=details,
        )

    def _evaluate_result(
        self,
        case: BenchmarkCase,
        result: dict[str, Any],
    ) -> tuple[float, dict]:
        """Evaluate a benchmark result.

        Args:
            case: Benchmark case.
            result: Result from runner.

        Returns:
            Tuple of (score, details).
        """
        import re

        checks = []
        details = {}

        # Check task type routing
        if case.expected_type:
            actual_type = result.get("task_type", "")
            if hasattr(actual_type, "value"):
                actual_type = actual_type.value
            type_match = case.expected_type.lower() in str(actual_type).lower()
            checks.append(1.0 if type_match else 0.0)
            details["type_match"] = type_match

        # Check SQL pattern
        if case.expected_sql_pattern:
            sql_results = result.get("sql_results", [])
            sql_matched = False
            for sql_result in sql_results:
                query = sql_result.get("query", "")
                if re.search(case.expected_sql_pattern, query, re.IGNORECASE):
                    sql_matched = True
                    break
            checks.append(1.0 if sql_matched else 0.0)
            details["sql_pattern_match"] = sql_matched

        # Check for errors
        errors = result.get("errors", [])
        has_errors = len(errors) > 0
        if not case.expected_sql_pattern:  # Only penalize errors for non-SQL tests
            checks.append(0.0 if has_errors else 1.0)
        details["has_errors"] = has_errors

        # Check for successful SQL execution
        sql_results = result.get("sql_results", [])
        successful_queries = sum(1 for r in sql_results if r.get("success", False))
        if sql_results:
            checks.append(successful_queries / len(sql_results))
            details["sql_success_rate"] = successful_queries / len(sql_results)

        # Calculate average score
        score = sum(checks) / len(checks) if checks else 0.0

        return score, details

    def _generate_report(self) -> BenchmarkReport:
        """Generate benchmark report from results.

        Returns:
            BenchmarkReport.
        """
        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        failed = total - passed

        scores = [r.score for r in self._results]
        latencies = [r.latency_ms for r in self._results]

        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Group by category
        by_category: dict[str, dict] = {}
        for case in self.cases:
            if case.category not in by_category:
                by_category[case.category] = {
                    "total": 0,
                    "passed": 0,
                    "scores": [],
                }

            result = next((r for r in self._results if r.case_id == case.id), None)
            if result:
                by_category[case.category]["total"] += 1
                if result.passed:
                    by_category[case.category]["passed"] += 1
                by_category[case.category]["scores"].append(result.score)

        # Calculate category averages
        for cat_data in by_category.values():
            scores = cat_data.pop("scores")
            cat_data["avg_score"] = sum(scores) / len(scores) if scores else 0
            cat_data["pass_rate"] = cat_data["passed"] / max(cat_data["total"], 1)

        return BenchmarkReport(
            name=self.name,
            timestamp=datetime.utcnow(),
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            avg_score=avg_score,
            avg_latency_ms=avg_latency,
            results=self._results,
            by_category=by_category,
        )

    def export_report(self, report: BenchmarkReport, path: str) -> None:
        """Export report to JSON file.

        Args:
            report: Benchmark report.
            path: Output file path.
        """
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Report exported to {path}")


def create_benchmark_suite(
    include_routing: bool = True,
    include_sql: bool = True,
    include_visualization: bool = True,
) -> BenchmarkSuite:
    """Create a benchmark suite with selected categories.

    Args:
        include_routing: Include routing benchmarks.
        include_sql: Include SQL benchmarks.
        include_visualization: Include visualization benchmarks.

    Returns:
        Configured BenchmarkSuite.
    """
    cases = []
    if include_routing:
        cases.extend(ROUTING_BENCHMARKS)
    if include_sql:
        cases.extend(SQL_BENCHMARKS)
    if include_visualization:
        cases.extend(VISUALIZATION_BENCHMARKS)

    return BenchmarkSuite(cases=cases)
