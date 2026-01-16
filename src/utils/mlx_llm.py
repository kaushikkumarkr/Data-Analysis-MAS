"""MLX LLM integration for Mac M1/M2/M3 native inference.

This module provides a LangChain-compatible LLM wrapper for MLX,
enabling efficient local inference on Apple Silicon.
"""

import asyncio
from typing import Any, Iterator, List, Optional, Mapping

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from pydantic import Field

from src.utils.logging import get_logger

logger = get_logger("utils.mlx_llm")


class MLXLLM(LLM):
    """LangChain LLM wrapper for MLX inference on Apple Silicon.

    This provides native M1/M2/M3 optimized inference using Apple's
    MLX framework. Ideal for privacy-preserving local LLM deployment.

    Attributes:
        model_name: HuggingFace model name or local path.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0-1).
        top_p: Top-p sampling parameter.
        repetition_penalty: Penalty for repeating tokens.
        trust_remote_code: Whether to trust remote code in model.
    """

    model_name: str = Field(
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        description="MLX model name from HuggingFace or local path",
    )
    max_tokens: int = Field(default=1024, description="Max tokens to generate")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    repetition_penalty: float = Field(default=1.1, description="Repetition penalty")
    trust_remote_code: bool = Field(default=True, description="Trust remote code")

    _model: Any = None
    _tokenizer: Any = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        """Return LLM type identifier."""
        return "mlx"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Return identifying parameters."""
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def _ensure_model_loaded(self) -> None:
        """Load the model if not already loaded."""
        if self._model is None:
            logger.info(f"Loading MLX model: {self.model_name}")
            try:
                from mlx_lm import load

                self._model, self._tokenizer = load(
                    self.model_name,
                    tokenizer_config={"trust_remote_code": self.trust_remote_code},
                )
                logger.info("MLX model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load MLX model: {e}")
                raise RuntimeError(f"Failed to load MLX model: {e}") from e

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt.

        Args:
            prompt: The prompt to generate from.
            stop: Optional list of stop strings.
            run_manager: Optional callback manager.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text string.
        """
        self._ensure_model_loaded()

        try:
            from mlx_lm import generate

            # Merge kwargs with defaults
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            top_p = kwargs.get("top_p", self.top_p)
            repetition_penalty = kwargs.get("repetition_penalty", self.repetition_penalty)

            response = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                verbose=False,
            )

            # Handle stop sequences
            if stop:
                for stop_seq in stop:
                    if stop_seq in response:
                        response = response[:response.index(stop_seq)]

            return response

        except Exception as e:
            logger.error(f"MLX generation failed: {e}")
            raise RuntimeError(f"MLX generation failed: {e}") from e

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async version of _call.

        Note: MLX doesn't have native async, so we run in executor.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._call(prompt, stop, run_manager, **kwargs),
        )


class MLXChatModel:
    """Chat-style interface for MLX models.

    Provides a chat completion interface similar to ChatOllama,
    suitable for use in LangGraph agents.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> None:
        """Initialize MLX chat model.

        Args:
            model_name: HuggingFace model name for MLX.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._tokenizer = None

        logger.info(f"MLXChatModel initialized with: {model_name}")

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if self._model is None:
            logger.info(f"Loading MLX model: {self.model_name}")
            from mlx_lm import load

            self._model, self._tokenizer = load(self.model_name)
            logger.info("MLX model loaded successfully")

    def _format_chat_prompt(self, messages) -> str:
        """Format messages into a chat prompt.

        Args:
            messages: List of messages or single prompt string.

        Returns:
            Formatted prompt string.
        """
        if isinstance(messages, str):
            return messages

        # Handle LangChain message objects
        parts = []
        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
                role = getattr(msg, 'type', 'user')
            elif isinstance(msg, dict):
                content = msg.get('content', '')
                role = msg.get('role', 'user')
            else:
                content = str(msg)
                role = 'user'

            if role in ('human', 'user'):
                parts.append(f"User: {content}")
            elif role in ('ai', 'assistant'):
                parts.append(f"Assistant: {content}")
            elif role == 'system':
                parts.append(f"System: {content}")
            else:
                parts.append(content)

        parts.append("Assistant:")
        return "\n\n".join(parts)

    def invoke(self, messages, **kwargs) -> Any:
        """Invoke the model with messages.

        Args:
            messages: Input messages (string or list).
            **kwargs: Additional generation parameters.

        Returns:
            Response object with content attribute.
        """
        self._ensure_loaded()

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self._format_chat_prompt(messages)

        # Get temperature from kwargs (use 'temperature' or 'temp')
        temp = kwargs.get("temperature", kwargs.get("temp", self.temperature))
        top_p = kwargs.get("top_p", self.top_p)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Create sampler with the parameters
        sampler = make_sampler(temp=temp, top_p=top_p)

        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )

        # Return object with content attribute like ChatOllama
        class Response:
            def __init__(self, content: str):
                self.content = content

        return Response(response.strip())

    async def ainvoke(self, messages, **kwargs) -> Any:
        """Async invoke.

        Args:
            messages: Input messages.
            **kwargs: Additional generation parameters.

        Returns:
            Response object with content attribute.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.invoke(messages, **kwargs),
        )


# Recommended models for Mac M1 16GB
RECOMMENDED_MODELS = {
    "small": "mlx-community/Llama-3.2-1B-Instruct-4bit",  # ~1GB, fast
    "medium": "mlx-community/Llama-3.2-3B-Instruct-4bit",  # ~2GB, balanced
    "large": "mlx-community/Qwen2.5-7B-Instruct-4bit",  # ~4GB, better quality
    "code": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",  # ~4GB, code focused
}


def get_recommended_model(size: str = "medium") -> str:
    """Get recommended model for Mac M1 16GB.

    Args:
        size: Model size preference (small, medium, large, code).

    Returns:
        Model name string.
    """
    return RECOMMENDED_MODELS.get(size, RECOMMENDED_MODELS["medium"])


def create_mlx_chat_model(
    model_size: str = "medium",
    **kwargs,
) -> MLXChatModel:
    """Factory function to create MLX chat model.

    Args:
        model_size: Size preference (small, medium, large, code).
        **kwargs: Additional model parameters.

    Returns:
        Configured MLXChatModel instance.
    """
    model_name = kwargs.pop("model_name", get_recommended_model(model_size))
    return MLXChatModel(model_name=model_name, **kwargs)
