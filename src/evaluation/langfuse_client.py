"""Langfuse client wrapper for observability.

Provides tracing and logging with graceful fallback
when Langfuse is unavailable.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator
from uuid import uuid4

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger("evaluation.langfuse")


@dataclass
class SpanData:
    """Data for a trace span."""

    id: str
    name: str
    trace_id: str
    parent_id: str | None
    start_time: datetime
    end_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    input: Any = None
    output: Any = None
    level: str = "DEFAULT"


@dataclass
class TraceData:
    """Data for a complete trace."""

    id: str
    name: str
    start_time: datetime
    end_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    spans: list[SpanData] = field(default_factory=list)
    user_id: str | None = None
    session_id: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata,
            "spans": len(self.spans),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "tags": self.tags,
        }


class LangfuseWrapper:
    """Wrapper for Langfuse with graceful fallback.

    Provides tracing capabilities that work with or without
    Langfuse being available.
    """

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Initialize Langfuse wrapper.

        Args:
            public_key: Langfuse public key.
            secret_key: Langfuse secret key.
            host: Langfuse host URL.
            enabled: Whether to enable Langfuse.
        """
        config = get_config()

        self.public_key = public_key or config.langfuse.public_key
        self.secret_key = secret_key or config.langfuse.secret_key
        self.host = host or config.langfuse.host
        self.enabled = enabled if enabled is not None else config.langfuse.enabled

        self._client = None
        self._traces: dict[str, TraceData] = {}
        self._current_trace_id: str | None = None

        if self.enabled and self.public_key and self.secret_key:
            self._init_client()
        else:
            logger.info("Langfuse disabled or credentials not provided, using local fallback")

    def _init_client(self) -> None:
        """Initialize Langfuse client."""
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
            )
            logger.info(f"Langfuse client initialized: {self.host}")
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse: {e}")
            self._client = None

    @property
    def is_connected(self) -> bool:
        """Check if Langfuse is connected."""
        return self._client is not None

    @contextmanager
    def trace(
        self,
        name: str,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Generator["TraceContext", None, None]:
        """Create a trace context.

        Args:
            name: Trace name.
            user_id: User identifier.
            session_id: Session identifier.
            metadata: Additional metadata.
            tags: Tags for filtering.

        Yields:
            TraceContext for adding spans.
        """
        trace_id = str(uuid4())
        trace_data = TraceData(
            id=trace_id,
            name=name,
            start_time=datetime.utcnow(),
            metadata=metadata or {},
            user_id=user_id,
            session_id=session_id,
            tags=tags or [],
        )
        self._traces[trace_id] = trace_data
        self._current_trace_id = trace_id

        # Create Langfuse trace if available
        langfuse_trace = None
        if self._client:
            try:
                langfuse_trace = self._client.trace(
                    id=trace_id,
                    name=name,
                    user_id=user_id,
                    session_id=session_id,
                    metadata=metadata,
                    tags=tags,
                )
            except Exception as e:
                logger.warning(f"Failed to create Langfuse trace: {e}")

        ctx = TraceContext(
            trace_data=trace_data,
            langfuse_trace=langfuse_trace,
            wrapper=self,
        )

        try:
            logger.debug(f"Started trace: {name} ({trace_id})")
            yield ctx
        finally:
            trace_data.end_time = datetime.utcnow()
            self._current_trace_id = None

            if self._client:
                try:
                    self._client.flush()
                except Exception as e:
                    logger.warning(f"Failed to flush Langfuse: {e}")

            duration = (trace_data.end_time - trace_data.start_time).total_seconds()
            logger.debug(f"Completed trace: {name} in {duration:.2f}s")

    def get_trace(self, trace_id: str) -> TraceData | None:
        """Get a trace by ID.

        Args:
            trace_id: Trace ID.

        Returns:
            TraceData or None.
        """
        return self._traces.get(trace_id)

    def list_traces(self, limit: int = 100) -> list[TraceData]:
        """List recent traces.

        Args:
            limit: Maximum traces to return.

        Returns:
            List of traces.
        """
        traces = list(self._traces.values())
        traces.sort(key=lambda t: t.start_time, reverse=True)
        return traces[:limit]

    def flush(self) -> None:
        """Flush pending data to Langfuse."""
        if self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.warning(f"Failed to flush: {e}")

    def shutdown(self) -> None:
        """Shutdown the client."""
        if self._client:
            try:
                self._client.shutdown()
            except Exception as e:
                logger.warning(f"Failed to shutdown: {e}")


class TraceContext:
    """Context for a trace with span management."""

    def __init__(
        self,
        trace_data: TraceData,
        langfuse_trace: Any,
        wrapper: LangfuseWrapper,
    ) -> None:
        """Initialize trace context.

        Args:
            trace_data: Local trace data.
            langfuse_trace: Langfuse trace object.
            wrapper: Parent wrapper reference.
        """
        self.trace_data = trace_data
        self.langfuse_trace = langfuse_trace
        self.wrapper = wrapper
        self._span_stack: list[str] = []

    @property
    def trace_id(self) -> str:
        """Get trace ID."""
        return self.trace_data.id

    @contextmanager
    def span(
        self,
        name: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        level: str = "DEFAULT",
    ) -> Generator["SpanContext", None, None]:
        """Create a span within the trace.

        Args:
            name: Span name.
            input: Input data.
            metadata: Span metadata.
            level: Log level (DEBUG, DEFAULT, WARNING, ERROR).

        Yields:
            SpanContext for adding output.
        """
        span_id = str(uuid4())
        parent_id = self._span_stack[-1] if self._span_stack else None

        span_data = SpanData(
            id=span_id,
            name=name,
            trace_id=self.trace_id,
            parent_id=parent_id,
            start_time=datetime.utcnow(),
            metadata=metadata or {},
            input=input,
            level=level,
        )

        self.trace_data.spans.append(span_data)
        self._span_stack.append(span_id)

        # Create Langfuse span if available
        langfuse_span = None
        if self.langfuse_trace:
            try:
                langfuse_span = self.langfuse_trace.span(
                    id=span_id,
                    name=name,
                    input=input,
                    metadata=metadata,
                    level=level,
                )
            except Exception as e:
                logger.warning(f"Failed to create Langfuse span: {e}")

        ctx = SpanContext(span_data=span_data, langfuse_span=langfuse_span)

        try:
            yield ctx
        finally:
            span_data.end_time = datetime.utcnow()
            span_data.status = "completed"
            self._span_stack.pop()

            if langfuse_span:
                try:
                    langfuse_span.end(output=span_data.output)
                except Exception as e:
                    logger.warning(f"Failed to end Langfuse span: {e}")

    def log_generation(
        self,
        name: str,
        model: str,
        input: str | list[dict],
        output: str,
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an LLM generation.

        Args:
            name: Generation name.
            model: Model name.
            input: Input prompt or messages.
            output: Generated output.
            usage: Token usage dict (prompt_tokens, completion_tokens).
            metadata: Additional metadata.
        """
        if self.langfuse_trace:
            try:
                self.langfuse_trace.generation(
                    name=name,
                    model=model,
                    input=input,
                    output=output,
                    usage=usage,
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning(f"Failed to log generation: {e}")

        # Also log locally
        logger.debug(
            f"Generation: {name} model={model} "
            f"tokens={usage.get('total_tokens', 'N/A') if usage else 'N/A'}"
        )


class SpanContext:
    """Context for a span with output management."""

    def __init__(self, span_data: SpanData, langfuse_span: Any) -> None:
        """Initialize span context.

        Args:
            span_data: Local span data.
            langfuse_span: Langfuse span object.
        """
        self.span_data = span_data
        self.langfuse_span = langfuse_span

    def set_output(self, output: Any) -> None:
        """Set span output.

        Args:
            output: Output data.
        """
        self.span_data.output = output

    def set_error(self, error: str) -> None:
        """Mark span as error.

        Args:
            error: Error message.
        """
        self.span_data.status = "error"
        self.span_data.level = "ERROR"
        self.span_data.metadata["error"] = error


# Global wrapper instance
_wrapper: LangfuseWrapper | None = None


def get_langfuse() -> LangfuseWrapper:
    """Get or create global Langfuse wrapper.

    Returns:
        LangfuseWrapper instance.
    """
    global _wrapper
    if _wrapper is None:
        _wrapper = LangfuseWrapper()
    return _wrapper


def create_langfuse(
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
    enabled: bool = True,
) -> LangfuseWrapper:
    """Create a new Langfuse wrapper.

    Args:
        public_key: Langfuse public key.
        secret_key: Langfuse secret key.
        host: Langfuse host URL.
        enabled: Whether to enable Langfuse.

    Returns:
        New LangfuseWrapper instance.
    """
    return LangfuseWrapper(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        enabled=enabled,
    )
