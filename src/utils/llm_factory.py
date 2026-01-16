"""LLM Factory for creating the appropriate LLM backend.

Supports both MLX (Apple Silicon optimized) and Ollama backends.
MLX is preferred on Mac M1/M2/M3 systems.
"""

from typing import Any, Protocol, runtime_checkable

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger("utils.llm_factory")


@runtime_checkable
class ChatModel(Protocol):
    """Protocol for chat model interface."""

    def invoke(self, messages: Any, **kwargs) -> Any:
        """Synchronous invoke."""
        ...

    async def ainvoke(self, messages: Any, **kwargs) -> Any:
        """Asynchronous invoke."""
        ...


def create_chat_model(
    backend: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> ChatModel:
    """Create a chat model using the configured backend.

    Args:
        backend: Backend to use ('mlx' or 'ollama'). Auto-detected if None.
        model_name: Model name override.
        temperature: Temperature override.
        max_tokens: Max tokens override.
        **kwargs: Additional backend-specific parameters.

    Returns:
        ChatModel instance (MLXChatModel or ChatOllama).

    Raises:
        RuntimeError: If no suitable backend is available.
    """
    config = get_config()

    # Determine backend
    if backend is None:
        # Prefer MLX on Mac if enabled
        backend = "mlx" if config.mlx.enabled else "ollama"

    logger.info(f"Creating chat model with backend: {backend}")

    if backend == "mlx":
        return _create_mlx_model(model_name, temperature, max_tokens, **kwargs)
    elif backend == "ollama":
        return _create_ollama_model(model_name, temperature, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _create_mlx_model(
    model_name: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> ChatModel:
    """Create MLX chat model.

    Args:
        model_name: Model name override.
        temperature: Temperature override.
        max_tokens: Max tokens override.
        **kwargs: Additional parameters.

    Returns:
        MLXChatModel instance.
    """
    try:
        from src.utils.mlx_llm import MLXChatModel

        config = get_config().mlx

        return MLXChatModel(
            model_name=model_name or config.model,
            temperature=temperature if temperature is not None else config.temperature,
            max_tokens=max_tokens if max_tokens is not None else config.max_tokens,
            top_p=kwargs.get("top_p", config.top_p),
        )
    except ImportError as e:
        logger.error(f"MLX not available: {e}")
        raise RuntimeError("MLX is not available. Install with: pip install mlx-lm") from e


def _create_ollama_model(
    model_name: str | None = None,
    temperature: float | None = None,
    **kwargs,
) -> ChatModel:
    """Create Ollama chat model.

    Args:
        model_name: Model name override.
        temperature: Temperature override.
        **kwargs: Additional parameters.

    Returns:
        ChatOllama instance.
    """
    try:
        from langchain_ollama import ChatOllama

        config = get_config().ollama

        return ChatOllama(
            model=model_name or config.model,
            temperature=temperature if temperature is not None else config.temperature,
            base_url=kwargs.get("base_url", config.host),
        )
    except ImportError as e:
        logger.error(f"LangChain Ollama not available: {e}")
        raise RuntimeError(
            "LangChain Ollama is not available. Install with: pip install langchain-ollama"
        ) from e


def get_available_backends() -> list[str]:
    """Get list of available LLM backends.

    Returns:
        List of available backend names.
    """
    backends = []

    # Check MLX availability
    try:
        import mlx_lm
        backends.append("mlx")
    except ImportError:
        pass

    # Check Ollama availability
    try:
        import langchain_ollama
        backends.append("ollama")
    except ImportError:
        pass

    return backends


def is_backend_available(backend: str) -> bool:
    """Check if a specific backend is available.

    Args:
        backend: Backend name to check.

    Returns:
        True if backend is available.
    """
    return backend in get_available_backends()
