#  Copyright (c) 2025, TensorCast Team.

"""
Pytest configuration and shared fixtures.
"""

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from tests.python.utils.ports import get_free_port, get_free_port_pair

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used in the test suite.

    Registering here avoids PytestUnknownMarkWarning for marks that may be
    provided by optional plugins (e.g., pytest-timeout). This keeps output
    clean even when those plugins are not installed in the environment.
    """
    config.addinivalue_line("markers", "timeout: per-test timeout in seconds")


@pytest.fixture(scope="function")
def in_memory_db() -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Fixture that provides an in-memory DuckDB database for testing.
    This is used by the GlobalStoreServicer internally.
    """
    if duckdb is None:
        # Skip only if a test actually requests this fixture in an environment
        # without DuckDB installed.
        pytest.skip("duckdb is not installed; skipping tests requiring in_memory_db")
    conn = duckdb.connect(":memory:")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="function")
def temp_db_file(tmp_path: Path) -> Iterator[str]:
    """Fixture that provides a temporary file path for a DuckDB database."""
    db_path = str(tmp_path / "test.db")
    yield db_path
    # Clean up the file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def free_port() -> int:
    """Get a single free port."""
    return get_free_port()


@pytest.fixture
def free_ports() -> tuple[int, int]:
    """Get a pair of free ports for RPC and metrics/P2P."""
    return get_free_port_pair()
