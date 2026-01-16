"""Evaluation utilities for DataVault.

Provides scoring and validation for agent outputs.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.utils.logging import get_logger

logger = get_logger("evaluation.evaluator")


class EvaluationScore(str, Enum):
    """Evaluation score levels."""

    EXCELLENT = "excellent"  # 0.9-1.0
    GOOD = "good"  # 0.7-0.9
    FAIR = "fair"  # 0.5-0.7
    POOR = "poor"  # 0.3-0.5
    FAILED = "failed"  # 0.0-0.3


@dataclass
class EvaluationResult:
    """Result of evaluating an output."""

    score: float  # 0.0 to 1.0
    level: EvaluationScore
    passed: bool
    feedback: str
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_score(cls, score: float, feedback: str = "", details: dict = None) -> "EvaluationResult":
        """Create from score.

        Args:
            score: Score from 0.0 to 1.0.
            feedback: Feedback message.
            details: Additional details.

        Returns:
            EvaluationResult.
        """
        if score >= 0.9:
            level = EvaluationScore.EXCELLENT
        elif score >= 0.7:
            level = EvaluationScore.GOOD
        elif score >= 0.5:
            level = EvaluationScore.FAIR
        elif score >= 0.3:
            level = EvaluationScore.POOR
        else:
            level = EvaluationScore.FAILED

        return cls(
            score=score,
            level=level,
            passed=score >= 0.5,
            feedback=feedback,
            details=details or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "level": self.level.value,
            "passed": self.passed,
            "feedback": self.feedback,
            "details": self.details,
        }


class SQLEvaluator:
    """Evaluates SQL query quality."""

    def __init__(self) -> None:
        """Initialize SQL evaluator."""
        self._sql_keywords = {
            "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY",
            "LIMIT", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER",
            "HAVING", "DISTINCT", "COUNT", "SUM", "AVG", "MIN", "MAX",
        }

    def evaluate_syntax(self, sql: str) -> EvaluationResult:
        """Evaluate SQL syntax.

        Args:
            sql: SQL query string.

        Returns:
            EvaluationResult for syntax.
        """
        if not sql or not sql.strip():
            return EvaluationResult.from_score(
                0.0,
                "Empty SQL query",
            )

        sql_upper = sql.upper()
        score = 0.0
        issues = []

        # Check for SELECT
        if "SELECT" in sql_upper:
            score += 0.3
        else:
            issues.append("Missing SELECT clause")

        # Check for FROM
        if "FROM" in sql_upper:
            score += 0.3
        else:
            issues.append("Missing FROM clause")

        # Check balanced parentheses
        if sql.count("(") == sql.count(")"):
            score += 0.2
        else:
            issues.append("Unbalanced parentheses")

        # Check for common issues
        if sql_upper.count("SELECT") == sql_upper.count("FROM"):
            score += 0.2
        else:
            issues.append("Mismatched SELECT/FROM count")

        feedback = "Valid SQL structure" if not issues else "; ".join(issues)

        return EvaluationResult.from_score(
            min(score, 1.0),
            feedback,
            {"issues": issues},
        )

    def evaluate_execution(
        self,
        success: bool,
        error: str | None = None,
        row_count: int = 0,
    ) -> EvaluationResult:
        """Evaluate SQL execution result.

        Args:
            success: Whether query executed successfully.
            error: Error message if failed.
            row_count: Number of rows returned.

        Returns:
            EvaluationResult for execution.
        """
        if not success:
            return EvaluationResult.from_score(
                0.0,
                f"Query failed: {error}",
                {"error": error},
            )

        # Score based on returning results
        if row_count > 0:
            score = 1.0
            feedback = f"Query returned {row_count} rows"
        else:
            score = 0.7
            feedback = "Query executed but returned no rows"

        return EvaluationResult.from_score(
            score,
            feedback,
            {"row_count": row_count},
        )

    def evaluate_relevance(
        self,
        sql: str,
        query: str,
        schema_columns: list[str] | None = None,
    ) -> EvaluationResult:
        """Evaluate if SQL is relevant to the query.

        Args:
            sql: Generated SQL.
            query: Original user query.
            schema_columns: Available columns.

        Returns:
            EvaluationResult for relevance.
        """
        sql_lower = sql.lower()
        query_lower = query.lower()
        score = 0.5  # Base score

        # Check for query keywords in SQL
        query_words = set(query_lower.split())
        sql_words = set(sql_lower.split())

        common_words = query_words & sql_words
        if common_words:
            score += 0.2 * min(len(common_words) / 3, 1.0)

        # Check for aggregation keywords
        agg_keywords = ["count", "sum", "avg", "average", "total", "how many"]
        if any(k in query_lower for k in agg_keywords):
            if any(k in sql_lower for k in ["count(", "sum(", "avg("]):
                score += 0.2

        # Check for grouping
        if "by" in query_lower and "group by" in sql_lower:
            score += 0.1

        return EvaluationResult.from_score(
            min(score, 1.0),
            "SQL appears relevant to query",
        )


class RoutingEvaluator:
    """Evaluates task routing accuracy."""

    def __init__(self) -> None:
        """Initialize routing evaluator."""
        self._clean_keywords = {
            "clean", "remove", "delete", "fix", "null", "duplicate",
            "dedupe", "missing", "invalid", "correct", "sanitize",
        }
        self._analyze_keywords = {
            "analyze", "count", "sum", "average", "total", "how many",
            "what is", "show", "list", "find", "report", "query",
            "select", "get", "retrieve", "statistics",
        }
        self._visualize_keywords = {
            "chart", "graph", "plot", "visualize", "visualization",
            "bar", "line", "pie", "histogram", "scatter", "draw",
        }

    def predict_task_type(self, query: str) -> str:
        """Predict expected task type from query.

        Args:
            query: User query.

        Returns:
            Predicted task type.
        """
        query_lower = query.lower()

        # Check for visualization keywords first (most specific)
        if any(kw in query_lower for kw in self._visualize_keywords):
            return "visualize"

        # Check for cleaning keywords
        if any(kw in query_lower for kw in self._clean_keywords):
            return "clean"

        # Check for analysis keywords
        if any(kw in query_lower for kw in self._analyze_keywords):
            return "analyze"

        return "analyze"  # Default

    def evaluate(
        self,
        query: str,
        actual_type: str,
    ) -> EvaluationResult:
        """Evaluate routing decision.

        Args:
            query: User query.
            actual_type: Actual task type assigned.

        Returns:
            EvaluationResult for routing.
        """
        expected = self.predict_task_type(query)
        actual = actual_type.lower() if actual_type else ""

        # Handle enum values
        if hasattr(actual, "value"):
            actual = actual.value.lower()

        if expected in actual or actual in expected:
            return EvaluationResult.from_score(
                1.0,
                f"Correct routing to {actual}",
                {"expected": expected, "actual": actual},
            )

        return EvaluationResult.from_score(
            0.3,
            f"Expected {expected}, got {actual}",
            {"expected": expected, "actual": actual},
        )


class ResponseEvaluator:
    """Evaluates overall response quality."""

    def __init__(self) -> None:
        """Initialize response evaluator."""
        self.sql_evaluator = SQLEvaluator()
        self.routing_evaluator = RoutingEvaluator()

    def evaluate(
        self,
        query: str,
        result: dict[str, Any],
    ) -> EvaluationResult:
        """Evaluate a complete response.

        Args:
            query: Original query.
            result: Agent result dict.

        Returns:
            Overall EvaluationResult.
        """
        scores = []
        details = {}

        # Evaluate routing
        task_type = result.get("task_type", "")
        routing_result = self.routing_evaluator.evaluate(query, str(task_type))
        scores.append(routing_result.score)
        details["routing"] = routing_result.to_dict()

        # Evaluate SQL if present
        sql_results = result.get("sql_results", [])
        if sql_results:
            sql_scores = []
            for sql_result in sql_results:
                exec_result = self.sql_evaluator.evaluate_execution(
                    sql_result.get("success", False),
                    sql_result.get("error"),
                    sql_result.get("row_count", 0),
                )
                sql_scores.append(exec_result.score)
            
            if sql_scores:
                avg_sql_score = sum(sql_scores) / len(sql_scores)
                scores.append(avg_sql_score)
                details["sql_avg_score"] = avg_sql_score

        # Check for errors
        errors = result.get("errors", [])
        if errors:
            scores.append(0.0)
            details["errors"] = errors
        else:
            scores.append(1.0)

        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 0.0

        return EvaluationResult.from_score(
            overall_score,
            f"Overall evaluation with {len(scores)} checks",
            details,
        )


def evaluate_response(query: str, result: dict[str, Any]) -> EvaluationResult:
    """Convenience function to evaluate a response.

    Args:
        query: Original query.
        result: Agent result.

    Returns:
        EvaluationResult.
    """
    evaluator = ResponseEvaluator()
    return evaluator.evaluate(query, result)
