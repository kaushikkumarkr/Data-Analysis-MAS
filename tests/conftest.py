"""Pytest configuration and fixtures."""

import os
import sys
from pathlib import Path

import pytest

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_csv_path() -> str:
    """Get path to sample sales data CSV.

    Returns:
        Absolute path to sales_data.csv.
    """
    return str(project_root / "data" / "sample" / "sales_data.csv")


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Get path for temporary database file.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Absolute path for temporary DuckDB file.
    """
    return str(tmp_path / "test.duckdb")
