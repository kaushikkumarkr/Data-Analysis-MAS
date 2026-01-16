"""Configuration management for DataVault."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class DuckDBConfig:
    """Configuration for DuckDB connection."""

    database_path: str = ":memory:"
    read_only: bool = False
    threads: int = 4

    @classmethod
    def from_env(cls) -> "DuckDBConfig":
        """Create configuration from environment variables.

        Returns:
            DuckDBConfig instance populated from environment.
        """
        load_dotenv()
        return cls(
            database_path=os.getenv("DUCKDB_DATABASE_PATH", ":memory:"),
            read_only=os.getenv("DUCKDB_READ_ONLY", "false").lower() == "true",
            threads=int(os.getenv("DUCKDB_THREADS", "4")),
        )


@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM."""

    host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.1
    timeout: int = 120

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        """Create configuration from environment variables.

        Returns:
            OllamaConfig instance populated from environment.
        """
        load_dotenv()
        return cls(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "120")),
        )


@dataclass
class MLXConfig:
    """Configuration for MLX LLM (Apple Silicon optimized).

    MLX is the preferred backend for Mac M1/M2/M3 systems
    as it provides native Apple Silicon optimization.
    """

    model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.9
    enabled: bool = True  # Prefer MLX on Mac

    @classmethod
    def from_env(cls) -> "MLXConfig":
        """Create configuration from environment variables.

        Returns:
            MLXConfig instance populated from environment.
        """
        load_dotenv()
        return cls(
            model=os.getenv("MLX_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit"),
            max_tokens=int(os.getenv("MLX_MAX_TOKENS", "1024")),
            temperature=float(os.getenv("MLX_TEMPERATURE", "0.1")),
            top_p=float(os.getenv("MLX_TOP_P", "0.9")),
            enabled=os.getenv("MLX_ENABLED", "true").lower() == "true",
        )


@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL (pgvector)."""

    host: str = "localhost"
    port: int = 5432
    database: str = "datavault"
    user: str = "datavault"
    password: str = "datavault"

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        """Create configuration from environment variables.

        Returns:
            PostgresConfig instance populated from environment.
        """
        load_dotenv()
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "datavault"),
            user=os.getenv("POSTGRES_USER", "datavault"),
            password=os.getenv("POSTGRES_PASSWORD", "datavault"),
        )

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string.

        Returns:
            PostgreSQL connection URL.
        """
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class LangfuseConfig:
    """Configuration for Langfuse observability."""

    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    host: str = "http://localhost:3000"
    enabled: bool = False

    @classmethod
    def from_env(cls) -> "LangfuseConfig":
        """Create configuration from environment variables.

        Returns:
            LangfuseConfig instance populated from environment.
        """
        load_dotenv()
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        return cls(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
            enabled=bool(public_key and secret_key),
        )


@dataclass
class AppConfig:
    """Main application configuration."""

    duckdb: DuckDBConfig = field(default_factory=DuckDBConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    mlx: MLXConfig = field(default_factory=MLXConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)
    log_level: str = "INFO"
    log_format: str = "json"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create full application configuration from environment.

        Returns:
            AppConfig instance populated from environment.
        """
        load_dotenv()
        return cls(
            duckdb=DuckDBConfig.from_env(),
            ollama=OllamaConfig.from_env(),
            mlx=MLXConfig.from_env(),
            postgres=PostgresConfig.from_env(),
            langfuse=LangfuseConfig.from_env(),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "json"),
        )


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create the global configuration instance.

    Returns:
        Global AppConfig instance.
    """
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
