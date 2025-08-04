#  Copyright (c) 2025, StepCast Team.

"""
Pytest configuration and shared fixtures.
"""

import asyncio
import logging
import os
import sys
import types
import time
import threading
import contextlib
from collections.abc import Sequence  # Added for general Sequence typing
from typing import Any

import duckdb
import pytest

# NOTE: Removed fake CheckpointStore related imports – tests now rely on the real
# scstore._checkpoint_store implementation shipped with the project.
from tests.python.utils.ports import get_free_port, get_free_port_pair

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    yield


@pytest.fixture(scope="function")
def in_memory_db():
    """
    Fixture that provides an in-memory DuckDB database for testing.
    This is used by the GlobalModelStoreServicer internally.
    """
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture(scope="function")
def temp_db_file(tmp_path):
    """Fixture that provides a temporary file path for a DuckDB database."""
    db_path = str(tmp_path / "test.db")
    yield db_path
    # Clean up the file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def free_port():
    """Get a single free port."""
    return get_free_port()


@pytest.fixture
def free_ports():
    """Get a pair of free ports for RPC and metrics/P2P."""
    return get_free_port_pair()


def cleanup_background_threads(
    target: Sequence[threading.Thread] | Any,
    stop_events: Sequence[threading.Event] | None = None,
    timeout: float = 5.0,
) -> None:
    """Stop background threads gracefully.

    Parameters
    ----------
    target:
        Either a sequence of ``threading.Thread`` objects **or** a
        ``StoreDaemonServicer`` instance from which threads will be
        extracted automatically.
    stop_events:
        Optional iterable of ``threading.Event`` instances to ``set`` before
        stopping the threads.  Pass ``None`` when not required.
    timeout:
        Maximum time (in seconds) to wait for each thread to finish.
    """

    # ------------------------------------------------------------------
    # Normalise *target* to a ``list[threading.Thread]`` called *threads*.
    # ------------------------------------------------------------------
    threads: list[threading.Thread | None]
    if isinstance(target, Sequence):
        # Caller already provided an explicit collection of threads.
        threads = list(target)
    else:
        # Assume *target* is a StoreDaemonServicer-like object.  Extract the
        # known background worker attributes *without* using ``getattr`` or
        # ``hasattr`` helper functions.
        threads = []
        # lifecycle_worker
        try:
            lw = target.lifecycle_worker
        except AttributeError:
            lw = None
        if lw is not None:
            threads.append(lw)

        # process_watcher
        try:
            pw = target.process_watcher
        except AttributeError:
            pw = None
        if pw is not None:
            threads.append(pw)

        # connection_manager (may not be present for local-only tests)
        try:
            cm = target.connection_manager
        except AttributeError:
            cm = None
        if cm is not None:
            threads.append(cm)

    # ------------------------------------------------------------------
    # Signal optional stop events first so that threads which wait on them
    # can exit promptly.
    # ------------------------------------------------------------------
    if stop_events:
        for event in stop_events:
            if event is not None:
                event.set()

    # ------------------------------------------------------------------
    # Attempt graceful shutdown of each thread object, waiting up to
    # *timeout* seconds for it to finish.
    # ------------------------------------------------------------------
    for thread in threads:
        if thread is None:
            continue

        # Call ``stop`` method if the object defines one. Avoid ``getattr`` by
        # direct attribute access wrapped in ``try``.
        try:
            stop_method = thread.stop
        except AttributeError:
            stop_method = None  # Ignore when not available

        if stop_method is not None and callable(stop_method):
            with contextlib.suppress(Exception):  # noqa: BLE001
                stop_method()

        # Join standard ``threading.Thread`` instances so that resources are
        # released before the test ends.
        if isinstance(thread, threading.Thread) and thread.is_alive():
            thread.join(timeout=timeout)


@pytest.fixture
def cleanup_servicer_threads():
    """Fixture to help clean up StoreDaemonServicer threads."""

    def _cleanup(servicer):
        """Extract and clean up threads from a StoreDaemonServicer instance."""
        threads: list[threading.Thread] = []
        events: list[threading.Event] = []

        # Collect lifecycle worker
        if servicer.lifecycle_worker is not None:
            threads.append(servicer.lifecycle_worker)

        # Collect process watcher
        if servicer.process_watcher is not None:
            threads.append(servicer.process_watcher)

        # Collect connection manager (always defined on servicer)
        if servicer.connection_manager is not None:
            threads.append(servicer.connection_manager)

        # Clean up using the consolidated helper
        cleanup_background_threads(threads, events)

    return _cleanup
