#  Copyright (c) 2025, TensorCast Team.

"""Base repository class for data access."""

from contextlib import contextmanager

from duckdb import DuckDBPyConnection

from tensorcast.global_store.exceptions import DatabaseError
from tensorcast.logger import init_logger

logger = init_logger(__name__)


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

    def get_cursor(self) -> DuckDBPyConnection:
        """
        Get a thread-local cursor for database operations.

        Following DuckDB best practices for multi-threading.
        Each thread gets its own cursor from the main connection.

        Returns:
            Thread-local DuckDB cursor
        """
        try:
            return self.connection.cursor()
        except Exception as e:
            logger.error(f"Failed to get database cursor: {e}")
            raise DatabaseError(f"Database operation failed: {e}") from e

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.

        Provides atomic transaction boundaries with proper rollback on exceptions.
        Uses the main connection to ensure all operations in the transaction
        are executed atomically.

        Yields:
            DuckDBPyConnection: Connection for transaction operations
        """
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute("BEGIN TRANSACTION")
            yield cursor
            cursor.execute("COMMIT")
        except Exception as e:
            if cursor:
                try:
                    cursor.execute("ROLLBACK")
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback transaction: {rollback_error}")
            logger.error(f"Transaction failed: {e}")
            raise DatabaseError(f"Transaction failed: {e}") from e
        finally:
            if cursor:
                cursor.close()
