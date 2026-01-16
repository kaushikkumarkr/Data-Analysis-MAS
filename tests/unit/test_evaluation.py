"""Tests for Evaluation module.

Tests Langfuse wrapper, metrics collector, benchmarks, and evaluators.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

from src.evaluation.langfuse_client import (
    LangfuseWrapper,
    TraceContext,
    SpanContext,
    TraceData,
    SpanData,
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


# ============================================================================
# Langfuse Client Tests
# ============================================================================

class TestTraceData:
    """Tests for TraceData."""

    def test_trace_data_creation(self) -> None:
        """Test creating trace data."""
        trace = TraceData(
            id="test-123",
            name="Test Trace",
            start_time=datetime.utcnow(),
        )
        
        assert trace.id == "test-123"
        assert trace.name == "Test Trace"
        assert trace.spans == []

    def test_trace_data_to_dict(self) -> None:
        """Test converting to dict."""
        trace = TraceData(
            id="test-123",
            name="Test Trace",
            start_time=datetime.utcnow(),
            user_id="user1",
            tags=["test"],
        )
        
        d = trace.to_dict()
        
        assert d["id"] == "test-123"
        assert d["user_id"] == "user1"
        assert d["tags"] == ["test"]


class TestSpanData:
    """Tests for SpanData."""

    def test_span_data_creation(self) -> None:
        """Test creating span data."""
        span = SpanData(
            id="span-1",
            name="Test Span",
            trace_id="trace-1",
            parent_id=None,
            start_time=datetime.utcnow(),
        )
        
        assert span.id == "span-1"
        assert span.status == "running"


class TestLangfuseWrapper:
    """Tests for LangfuseWrapper."""

    def test_wrapper_creation_disabled(self) -> None:
        """Test creating wrapper with Langfuse disabled."""
        wrapper = create_langfuse(enabled=False)
        
        assert not wrapper.is_connected

    def test_trace_context_manager(self) -> None:
        """Test trace context manager."""
        wrapper = create_langfuse(enabled=False)
        
        with wrapper.trace("Test Trace", user_id="user1") as ctx:
            assert ctx.trace_id is not None
            
        # Trace should be stored
        traces = wrapper.list_traces()
        assert len(traces) == 1
        assert traces[0].name == "Test Trace"

    def test_span_context_manager(self) -> None:
        """Test span within trace."""
        wrapper = create_langfuse(enabled=False)
        
        with wrapper.trace("Test Trace") as ctx:
            with ctx.span("Test Span", input="test input") as span:
                span.set_output("test output")
        
        trace = wrapper.get_trace(ctx.trace_id)
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "Test Span"

    def test_nested_spans(self) -> None:
        """Test nested spans."""
        wrapper = create_langfuse(enabled=False)
        
        with wrapper.trace("Trace") as ctx:
            with ctx.span("Parent") as parent:
                with ctx.span("Child") as child:
                    child.set_output("child result")
                parent.set_output("parent result")
        
        trace = wrapper.get_trace(ctx.trace_id)
        assert len(trace.spans) == 2

    def test_log_generation(self) -> None:
        """Test logging LLM generation."""
        wrapper = create_langfuse(enabled=False)
        
        with wrapper.trace("LLM Trace") as ctx:
            ctx.log_generation(
                name="test_generation",
                model="llama3",
                input="Hello",
                output="Hi there!",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )
        
        # Should not raise


# ============================================================================
# Metrics Tests
# ============================================================================

class TestMetricPoint:
    """Tests for MetricPoint."""

    def test_metric_point_creation(self) -> None:
        """Test creating metric point."""
        point = MetricPoint(
            name="latency",
            value=100.5,
            timestamp=datetime.utcnow(),
            unit="ms",
        )
        
        assert point.name == "latency"
        assert point.value == 100.5
        assert point.unit == "ms"

    def test_metric_point_to_dict(self) -> None:
        """Test converting to dict."""
        point = MetricPoint(
            name="count",
            value=42,
            timestamp=datetime.utcnow(),
            tags={"agent": "analyst"},
        )
        
        d = point.to_dict()
        
        assert d["name"] == "count"
        assert d["tags"]["agent"] == "analyst"


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    @pytest.fixture
    def collector(self) -> MetricsCollector:
        """Create fresh collector."""
        return MetricsCollector()

    def test_collector_creation(self, collector) -> None:
        """Test creating collector."""
        assert collector is not None

    def test_record_metric(self, collector) -> None:
        """Test recording a metric."""
        collector.record("test_metric", 123.45, unit="ms")
        
        metrics = collector.get_metrics("test_metric")
        
        assert len(metrics) == 1
        assert metrics[0].value == 123.45

    def test_increment_counter(self, collector) -> None:
        """Test incrementing counter."""
        collector.increment("requests")
        collector.increment("requests")
        collector.increment("requests", 5)
        
        summary = collector.get_summary()
        
        # Check internal counter
        assert collector._counters["requests"] == 7

    def test_timer(self, collector) -> None:
        """Test timer functionality."""
        import time
        
        collector.start_timer("test_op")
        time.sleep(0.01)  # 10ms
        duration = collector.stop_timer("test_op")
        
        assert duration >= 10  # At least 10ms

    def test_record_request(self, collector) -> None:
        """Test recording complete request."""
        collector.record_request(
            success=True,
            latency_ms=150,
            agent="analyst",
            tokens=100,
            sql_queries=2,
            rows_returned=50,
        )
        
        summary = collector.get_summary()
        
        assert summary.total_requests == 1
        assert summary.successful_requests == 1
        assert summary.by_agent["analyst"]["count"] == 1

    def test_record_llm_call(self, collector) -> None:
        """Test recording LLM call."""
        collector.record_llm_call(
            model="llama3",
            latency_ms=500,
            prompt_tokens=50,
            completion_tokens=100,
        )
        
        metrics = collector.get_metrics("llm_latency")
        assert len(metrics) == 1

    def test_record_sql_execution(self, collector) -> None:
        """Test recording SQL execution."""
        collector.record_sql_execution(success=True, latency_ms=10, rows=25)
        collector.record_sql_execution(success=False, latency_ms=5, rows=0)
        
        summary = collector.get_summary()
        
        assert summary.total_sql_queries == 2

    def test_summary(self, collector) -> None:
        """Test getting summary."""
        collector.record_request(success=True, latency_ms=100, agent="analyst")
        collector.record_request(success=True, latency_ms=200, agent="cleaner")
        collector.record_request(success=False, latency_ms=50, agent="analyst")
        
        summary = collector.get_summary()
        
        assert summary.total_requests == 3
        assert summary.successful_requests == 2
        assert summary.failed_requests == 1
        assert summary.avg_latency_ms == pytest.approx(116.67, rel=0.1)

    def test_reset(self, collector) -> None:
        """Test resetting metrics."""
        collector.record("test", 100)
        collector.increment("counter")
        
        collector.reset()
        
        assert len(collector.get_metrics()) == 0

    def test_export_to_json(self, collector) -> None:
        """Test JSON export."""
        collector.record_request(success=True, latency_ms=100, agent="test")
        
        export = collector.export_to_json()
        
        assert "summary" in export
        assert "metrics" in export
        assert "counters" in export


# ============================================================================
# Benchmark Tests
# ============================================================================

class TestBenchmarkCase:
    """Tests for BenchmarkCase."""

    def test_case_creation(self) -> None:
        """Test creating benchmark case."""
        case = BenchmarkCase(
            id="test_001",
            name="Test Case",
            category="routing",
            query="What is the total?",
            expected_type="analyze",
        )
        
        assert case.id == "test_001"
        assert case.expected_type == "analyze"


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_result_creation(self) -> None:
        """Test creating result."""
        result = BenchmarkResult(
            case_id="test_001",
            passed=True,
            score=0.85,
            latency_ms=150,
        )
        
        assert result.passed is True
        assert result.score == 0.85

    def test_result_to_dict(self) -> None:
        """Test converting to dict."""
        result = BenchmarkResult(
            case_id="test_001",
            passed=False,
            score=0.3,
            latency_ms=100,
            error="Test error",
        )
        
        d = result.to_dict()
        
        assert d["error"] == "Test error"
        assert d["passed"] is False


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite."""

    def test_suite_creation(self) -> None:
        """Test creating suite."""
        suite = BenchmarkSuite()
        
        assert len(suite.cases) > 0

    def test_suite_with_custom_cases(self) -> None:
        """Test suite with custom cases."""
        cases = [
            BenchmarkCase(
                id="custom_001",
                name="Custom",
                category="test",
                query="Test query",
            )
        ]
        
        suite = BenchmarkSuite(cases=cases)
        
        assert len(suite.cases) == 1

    def test_add_case(self) -> None:
        """Test adding case."""
        suite = BenchmarkSuite(name="empty", cases=[])
        initial_count = len(suite.cases)
        
        suite.add_case(BenchmarkCase(
            id="new",
            name="New",
            category="test",
            query="Test",
        ))
        
        assert len(suite.cases) == initial_count + 1

    @pytest.mark.asyncio
    async def test_run_benchmark(self) -> None:
        """Test running benchmark."""
        cases = [
            BenchmarkCase(
                id="test_001",
                name="Test",
                category="routing",
                query="Count the rows",
                expected_type="analyze",
            )
        ]
        
        suite = BenchmarkSuite(cases=cases)
        
        async def mock_runner(query: str) -> dict:
            return {
                "task_type": "analyze",
                "sql_results": [{"success": True, "row_count": 10}],
                "errors": [],
            }
        
        report = await suite.run(mock_runner)
        
        assert report.total_cases == 1
        assert report.passed_cases == 1

    def test_create_benchmark_suite(self) -> None:
        """Test factory function."""
        suite = create_benchmark_suite(
            include_routing=True,
            include_sql=False,
            include_visualization=False,
        )
        
        # Should only have routing benchmarks
        assert all(c.category == "routing" for c in suite.cases)


# ============================================================================
# Evaluator Tests
# ============================================================================

class TestEvaluationResult:
    """Tests for EvaluationResult."""

    def test_from_score_excellent(self) -> None:
        """Test excellent score."""
        result = EvaluationResult.from_score(0.95, "Great!")
        
        assert result.level == EvaluationScore.EXCELLENT
        assert result.passed is True

    def test_from_score_good(self) -> None:
        """Test good score."""
        result = EvaluationResult.from_score(0.75)
        
        assert result.level == EvaluationScore.GOOD
        assert result.passed is True

    def test_from_score_failed(self) -> None:
        """Test failed score."""
        result = EvaluationResult.from_score(0.1)
        
        assert result.level == EvaluationScore.FAILED
        assert result.passed is False

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        result = EvaluationResult.from_score(0.8, "Good job")
        
        d = result.to_dict()
        
        assert d["score"] == 0.8
        assert d["feedback"] == "Good job"


class TestSQLEvaluator:
    """Tests for SQLEvaluator."""

    @pytest.fixture
    def evaluator(self) -> SQLEvaluator:
        """Create evaluator."""
        return SQLEvaluator()

    def test_evaluate_valid_syntax(self, evaluator) -> None:
        """Test evaluating valid SQL."""
        result = evaluator.evaluate_syntax(
            "SELECT * FROM sales WHERE amount > 100"
        )
        
        assert result.passed is True
        assert result.score >= 0.8

    def test_evaluate_empty_sql(self, evaluator) -> None:
        """Test evaluating empty SQL."""
        result = evaluator.evaluate_syntax("")
        
        assert result.passed is False
        assert result.score == 0.0

    def test_evaluate_successful_execution(self, evaluator) -> None:
        """Test evaluating successful execution."""
        result = evaluator.evaluate_execution(
            success=True,
            row_count=50,
        )
        
        assert result.passed is True
        assert result.score == 1.0

    def test_evaluate_failed_execution(self, evaluator) -> None:
        """Test evaluating failed execution."""
        result = evaluator.evaluate_execution(
            success=False,
            error="Syntax error",
        )
        
        assert result.passed is False
        assert "Syntax error" in result.feedback


class TestRoutingEvaluator:
    """Tests for RoutingEvaluator."""

    @pytest.fixture
    def evaluator(self) -> RoutingEvaluator:
        """Create evaluator."""
        return RoutingEvaluator()

    def test_predict_analyze(self, evaluator) -> None:
        """Test predicting analyze task."""
        result = evaluator.predict_task_type("How many rows are there?")
        assert result == "analyze"

    def test_predict_clean(self, evaluator) -> None:
        """Test predicting clean task."""
        result = evaluator.predict_task_type("Remove null values")
        assert result == "clean"

    def test_predict_visualize(self, evaluator) -> None:
        """Test predicting visualize task."""
        result = evaluator.predict_task_type("Create a bar chart")
        assert result == "visualize"

    def test_evaluate_correct_routing(self, evaluator) -> None:
        """Test correct routing evaluation."""
        result = evaluator.evaluate(
            "Show total sales",
            "analyze",
        )
        
        assert result.passed is True
        assert result.score == 1.0

    def test_evaluate_incorrect_routing(self, evaluator) -> None:
        """Test incorrect routing evaluation."""
        result = evaluator.evaluate(
            "Remove duplicates",
            "analyze",  # Should be clean
        )
        
        assert result.passed is False


class TestResponseEvaluator:
    """Tests for ResponseEvaluator."""

    @pytest.fixture
    def evaluator(self) -> ResponseEvaluator:
        """Create evaluator."""
        return ResponseEvaluator()

    def test_evaluate_good_response(self, evaluator) -> None:
        """Test evaluating a good response."""
        result = evaluator.evaluate(
            query="Count the sales",
            result={
                "task_type": "analyze",
                "sql_results": [{"success": True, "row_count": 100}],
                "errors": [],
            },
        )
        
        assert result.passed is True
        assert result.score >= 0.7

    def test_evaluate_response_with_errors(self, evaluator) -> None:
        """Test evaluating response with errors."""
        result = evaluator.evaluate(
            query="Count the sales",
            result={
                "task_type": "analyze",
                "sql_results": [],
                "errors": ["Query failed"],
            },
        )
        
        assert result.score < 0.7


class TestEvaluateResponseFunction:
    """Tests for evaluate_response convenience function."""

    def test_evaluate_response(self) -> None:
        """Test convenience function."""
        result = evaluate_response(
            query="Show me sales data",
            result={
                "task_type": "analyze",
                "sql_results": [{"success": True, "row_count": 50}],
                "errors": [],
            },
        )
        
        assert isinstance(result, EvaluationResult)
        assert result.passed is True
