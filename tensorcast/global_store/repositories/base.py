#  Copyright (c) 2025-2026, TensorCast Team.

"""Base repository class for data access."""

import contextvars
import itertools
import threading
import time
import traceback
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import cast

from duckdb import DuckDBPyConnection

from tensorcast.global_store.exceptions import DatabaseError
from tensorcast.logger import init_logger

logger = init_logger(__name__)
_DB_EXECUTION_LOCK = threading.RLock()
_ACTIVE_TX_LOCK = threading.RLock()
_NEXT_TX_ID = itertools.count(1)
_ACTIVE_TRANSACTIONS: dict[int, dict[str, object]] = {}
_CURRENT_TX_CONTEXT: contextvars.ContextVar[dict[str, object] | None] = (
    contextvars.ContextVar("global_store_tx_context", default=None)
)
_RECENT_WRITES_LOCK = threading.RLock()
_RECENT_WORKER_WRITES: deque[dict[str, object]] = deque(maxlen=128)
_RECENT_MUTATION_WRITES: deque[dict[str, object]] = deque(maxlen=256)
_CURSOR_STATS_LOCK = threading.RLock()
_OPEN_CURSORS_BY_CONN: dict[str, int] = {}
_OPEN_CURSORS_TOTAL = 0


@contextmanager
def db_execution_lock():
    """Serialize direct DuckDB connection execution outside repositories."""
    _DB_EXECUTION_LOCK.acquire()
    try:
        yield
    finally:
        _DB_EXECUTION_LOCK.release()


def _is_mutation_query(summary: str) -> bool:
    normalized = summary.lstrip().lower()
    return normalized.startswith(
        (
            "insert ",
            "update ",
            "delete ",
            "merge ",
            "replace ",
            "truncate ",
            "create ",
            "drop ",
            "alter ",
        )
    )


def _touches_worker_tables(summary: str) -> bool:
    lowered = summary.lower()
    return (
        " workers " in lowered
        or " worker_liveness " in lowered
        or " worker_reconcile_state " in lowered
    )


def _record_recent_worker_write(
    *,
    repo: str,
    repo_obj_id: str,
    conn_obj_id: str,
    thread: str,
    tx_id: int | None,
    query: str,
    tx_context: dict[str, object],
) -> None:
    entry = {
        "at_monotonic": time.monotonic(),
        "repo": repo,
        "repo_obj_id": repo_obj_id,
        "conn_obj_id": conn_obj_id,
        "thread": thread,
        "tx_id": tx_id,
        "query": query,
        "tx_context": dict(tx_context),
    }
    with _RECENT_WRITES_LOCK:
        _RECENT_WORKER_WRITES.append(entry)


def _recent_worker_writes_snapshot(limit: int = 12) -> list[tuple[object, ...]]:
    now = time.monotonic()
    with _RECENT_WRITES_LOCK:
        items = list(_RECENT_WORKER_WRITES)
    sliced = items[-max(1, int(limit)) :]
    return [_snapshot_recent_write(item, now) for item in sliced]


def _record_recent_mutation_write(
    *,
    repo: str,
    repo_obj_id: str,
    conn_obj_id: str,
    thread: str,
    tx_id: int | None,
    query: str,
    tx_context: dict[str, object],
) -> None:
    entry = {
        "at_monotonic": time.monotonic(),
        "repo": repo,
        "repo_obj_id": repo_obj_id,
        "conn_obj_id": conn_obj_id,
        "thread": thread,
        "tx_id": tx_id,
        "query": query,
        "tx_context": dict(tx_context),
    }
    with _RECENT_WRITES_LOCK:
        _RECENT_MUTATION_WRITES.append(entry)


def _recent_mutation_writes_snapshot(limit: int = 24) -> list[tuple[object, ...]]:
    now = time.monotonic()
    with _RECENT_WRITES_LOCK:
        items = list(_RECENT_MUTATION_WRITES)
    sliced = items[-max(1, int(limit)) :]
    return [_snapshot_recent_write(item, now) for item in sliced]


def _record_cursor_open(conn_obj_id: str) -> None:
    global _OPEN_CURSORS_TOTAL
    with _CURSOR_STATS_LOCK:
        _OPEN_CURSORS_TOTAL += 1
        _OPEN_CURSORS_BY_CONN[conn_obj_id] = (
            _OPEN_CURSORS_BY_CONN.get(conn_obj_id, 0) + 1
        )


def _record_cursor_close(conn_obj_id: str) -> None:
    global _OPEN_CURSORS_TOTAL
    with _CURSOR_STATS_LOCK:
        current = _OPEN_CURSORS_BY_CONN.get(conn_obj_id, 0)
        if current > 1:
            _OPEN_CURSORS_BY_CONN[conn_obj_id] = current - 1
        else:
            _OPEN_CURSORS_BY_CONN.pop(conn_obj_id, None)
        if _OPEN_CURSORS_TOTAL > 0:
            _OPEN_CURSORS_TOTAL -= 1


def _cursor_stats_snapshot(conn_obj_id: str) -> tuple[int, int]:
    with _CURSOR_STATS_LOCK:
        return int(_OPEN_CURSORS_BY_CONN.get(conn_obj_id, 0)), int(_OPEN_CURSORS_TOTAL)


@contextmanager
def bind_tx_context(**fields):
    """Attach lightweight context to repository transactions in this thread."""
    token = _CURRENT_TX_CONTEXT.set(dict(fields))
    try:
        yield
    finally:
        _CURRENT_TX_CONTEXT.reset(token)


def _current_tx_context() -> dict[str, object]:
    raw = _CURRENT_TX_CONTEXT.get()
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _capture_origin() -> str:
    stack = traceback.extract_stack(limit=16)
    for frame in reversed(stack[:-1]):
        if frame.filename.endswith("repositories/base.py"):
            continue
        if "contextlib.py" in frame.filename:
            continue
        return f"{frame.name}@{frame.filename}:{frame.lineno}"
    return "<unknown>"


def _coerce_float(value: object, default: float) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    return default


def _coerce_tx_context(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return cast(dict[str, object], dict(value))
    return {}


def _snapshot_recent_write(item: dict[str, object], now: float) -> tuple[object, ...]:
    return (
        round((now - _coerce_float(item.get("at_monotonic"), now)) * 1000.0, 3),
        str(item.get("repo", "")),
        str(item.get("repo_obj_id", "")),
        str(item.get("conn_obj_id", "")),
        str(item.get("thread", "")),
        item.get("tx_id"),
        str(item.get("query", "")),
        _coerce_tx_context(item.get("tx_context")),
    )


def is_transient_tx_conflict(exc: Exception) -> bool:
    """Return True when *exc* looks like a transient transaction conflict."""
    message = str(exc).lower()
    conflict_markers = (
        "write-write conflict",
        "conflict on tuple deletion",
        "conflict on update",
        "failed to commit: write-write conflict on key",
        "serialization",
        "transactioncontext error: conflict",
    )
    return any(marker in message for marker in conflict_markers)


class _LockedCursor:
    """Thread-safe cursor wrapper sharing a global DB execution lock."""

    def __init__(
        self,
        cursor: DuckDBPyConnection,
        *,
        owner_tx_id: int | None = None,
        owner_repo: str = "",
        owner_repo_obj_id: str = "",
        owner_conn_obj_id: str = "",
    ):
        self._cursor = cursor
        self._last_query_summary = "<none>"
        self._owner_tx_id = owner_tx_id
        self._owner_repo = owner_repo
        self._owner_repo_obj_id = owner_repo_obj_id
        self._owner_conn_obj_id = owner_conn_obj_id
        self._closed = False
        _record_cursor_open(self._owner_conn_obj_id)

    def _summarize_query(self, args: tuple) -> str:
        if not args:
            return "<empty>"
        query = args[0]
        if isinstance(query, str):
            compact = " ".join(query.split())
            if len(compact) > 240:
                return compact[:240] + "..."
            return compact
        rendered = repr(query)
        if len(rendered) > 240:
            return rendered[:240] + "..."
        return rendered

    def execute(self, *args, **kwargs):
        self._last_query_summary = self._summarize_query(args)
        tx_context: dict[str, object] = _current_tx_context()
        if self._owner_tx_id is not None:
            with _ACTIVE_TX_LOCK:
                tx = _ACTIVE_TRANSACTIONS.get(self._owner_tx_id)
                if tx is not None:
                    tx["last_query"] = self._last_query_summary
                    tx["last_query_at"] = time.monotonic()
                    meta_context = tx.get("tx_context")
                    if isinstance(meta_context, dict):
                        tx_context = dict(meta_context)
        if _is_mutation_query(self._last_query_summary):
            _record_recent_mutation_write(
                repo=self._owner_repo,
                repo_obj_id=self._owner_repo_obj_id,
                conn_obj_id=self._owner_conn_obj_id,
                thread=threading.current_thread().name,
                tx_id=self._owner_tx_id,
                query=self._last_query_summary,
                tx_context=tx_context,
            )
            if _touches_worker_tables(self._last_query_summary):
                _record_recent_worker_write(
                    repo=self._owner_repo,
                    repo_obj_id=self._owner_repo_obj_id,
                    conn_obj_id=self._owner_conn_obj_id,
                    thread=threading.current_thread().name,
                    tx_id=self._owner_tx_id,
                    query=self._last_query_summary,
                    tx_context=tx_context,
                )
        with _DB_EXECUTION_LOCK:
            self._cursor.execute(*args, **kwargs)
        return self

    def fetchone(self):
        with _DB_EXECUTION_LOCK:
            return self._cursor.fetchone()

    def fetchall(self):
        with _DB_EXECUTION_LOCK:
            return self._cursor.fetchall()

    def close(self) -> None:
        if self._closed:
            return
        with _DB_EXECUTION_LOCK:
            if self._closed:
                return
            self._closed = True
            self._cursor.close()
            _record_cursor_close(self._owner_conn_obj_id)

    def __del__(self):
        with suppress(Exception):
            self.close()

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    @property
    def last_query_summary(self) -> str:
        return self._last_query_summary


class BaseRepository:
    """Base class for all repositories."""

    def __init__(self, connection: DuckDBPyConnection):
        """
        Initialize repository with DuckDB connection.

        Args:
            connection: Main DuckDB connection
        """
        self.connection = connection
        self._conn = connection  # For test mocking
        self._repo_name = self.__class__.__name__

    def get_cursor(self) -> DuckDBPyConnection:
        """
        Get a thread-local cursor for database operations.

        Following DuckDB best practices for multi-threading.
        Each thread gets its own cursor from the main connection.

        Returns:
            Thread-local DuckDB cursor
        """
        try:
            return cast(
                DuckDBPyConnection,
                _LockedCursor(
                    self.connection.cursor(),
                    owner_repo=self._repo_name,
                    owner_repo_obj_id=hex(id(self)),
                    owner_conn_obj_id=hex(id(self.connection)),
                ),
            )
        except Exception as e:
            logger.error(f"Failed to get database cursor: {e}")
            raise DatabaseError(f"Database operation failed: {e}") from e

    @contextmanager
    def transaction(self) -> Iterator[DuckDBPyConnection]:
        """
        Context manager for database transactions.

        Provides atomic transaction boundaries with proper rollback on exceptions.
        Uses the main connection to ensure all operations in the transaction
        are executed atomically.

        Yields:
            DuckDBPyConnection: Connection for transaction operations
        """
        raw_cursor = None
        locked_cursor: _LockedCursor | None = None
        tx_id = next(_NEXT_TX_ID)
        tx_started_at = time.monotonic()
        thread_name = threading.current_thread().name
        origin = _capture_origin()
        tx_context = _current_tx_context()
        _DB_EXECUTION_LOCK.acquire()
        try:
            raw_cursor = self.connection.cursor()
            raw_cursor.execute("BEGIN TRANSACTION")
            with _ACTIVE_TX_LOCK:
                _ACTIVE_TRANSACTIONS[tx_id] = {
                    "repo": self._repo_name,
                    "repo_obj_id": hex(id(self)),
                    "conn_obj_id": hex(id(self.connection)),
                    "thread": thread_name,
                    "started_at": tx_started_at,
                    "last_query": "BEGIN TRANSACTION",
                    "last_query_at": tx_started_at,
                    "origin": origin,
                    "tx_context": tx_context,
                }
            locked_cursor = _LockedCursor(
                raw_cursor,
                owner_tx_id=tx_id,
                owner_repo=self._repo_name,
                owner_repo_obj_id=hex(id(self)),
                owner_conn_obj_id=hex(id(self.connection)),
            )
            yield cast(DuckDBPyConnection, locked_cursor)
            raw_cursor.execute("COMMIT")
        except Exception as e:
            rollback_error: Exception | None = None
            rollback_was_noop = False
            if raw_cursor:
                try:
                    raw_cursor.execute("ROLLBACK")
                except Exception as rollback_exc:
                    rollback_error = rollback_exc
                    rollback_message = str(rollback_exc).lower()
                    rollback_was_noop = "no transaction is active" in rollback_message
                    if rollback_was_noop:
                        logger.debug(
                            "Rollback no-op after primary transaction failure: %s",
                            rollback_exc,
                        )
                    else:
                        logger.warning(
                            "Rollback failed after primary transaction failure: %s",
                            rollback_exc,
                        )
            last_query = (
                locked_cursor.last_query_summary
                if locked_cursor is not None
                else "<unknown>"
            )
            with _ACTIVE_TX_LOCK:
                active_others = [
                    (
                        other_tx_id,
                        str(meta.get("repo", "")),
                        str(meta.get("repo_obj_id", "")),
                        str(meta.get("conn_obj_id", "")),
                        str(meta.get("thread", "")),
                        float(
                            time.monotonic()
                            - _coerce_float(meta.get("started_at"), tx_started_at)
                        ),
                        str(meta.get("last_query", "<unknown>")),
                        str(meta.get("origin", "<unknown>")),
                        _coerce_tx_context(meta.get("tx_context")),
                    )
                    for other_tx_id, meta in _ACTIVE_TRANSACTIONS.items()
                    if other_tx_id != tx_id
                ]
            active_others.sort(key=lambda item: item[0])
            recent_worker_writes = _recent_worker_writes_snapshot(limit=16)
            recent_mutation_writes = _recent_mutation_writes_snapshot(limit=24)
            conn_open_cursors, total_open_cursors = _cursor_stats_snapshot(
                hex(id(self.connection))
            )
            if rollback_error is None or rollback_was_noop:
                logger.error(
                    "Transaction failed repo=%s repo_obj_id=%s conn_obj_id=%s tx_id=%s thread=%s origin=%s tx_context=%s last_query=%s active_tx=%s recent_worker_writes=%s recent_mutation_writes=%s conn_open_cursors=%s total_open_cursors=%s error=%s",
                    self._repo_name,
                    hex(id(self)),
                    hex(id(self.connection)),
                    tx_id,
                    thread_name,
                    origin,
                    tx_context,
                    last_query,
                    active_others[:8],
                    recent_worker_writes,
                    recent_mutation_writes,
                    conn_open_cursors,
                    total_open_cursors,
                    e,
                )
            else:
                logger.error(
                    "Transaction failed repo=%s repo_obj_id=%s conn_obj_id=%s tx_id=%s thread=%s origin=%s tx_context=%s last_query=%s active_tx=%s recent_worker_writes=%s recent_mutation_writes=%s conn_open_cursors=%s total_open_cursors=%s error=%s (secondary rollback error: %s)",
                    self._repo_name,
                    hex(id(self)),
                    hex(id(self.connection)),
                    tx_id,
                    thread_name,
                    origin,
                    tx_context,
                    last_query,
                    active_others[:8],
                    recent_worker_writes,
                    recent_mutation_writes,
                    conn_open_cursors,
                    total_open_cursors,
                    e,
                    rollback_error,
                )
            raise DatabaseError(f"Transaction failed: {e}") from e
        finally:
            with _ACTIVE_TX_LOCK:
                _ACTIVE_TRANSACTIONS.pop(tx_id, None)
            if locked_cursor is not None:
                locked_cursor.close()
            elif raw_cursor:
                raw_cursor.close()
            _DB_EXECUTION_LOCK.release()
