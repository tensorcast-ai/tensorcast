#  Copyright (c) 2025-2026, TensorCast Team.

import os
import re
from contextlib import suppress
from importlib import resources

import duckdb
from duckdb import DuckDBPyConnection

from tensorcast.logger import init_logger

logger = init_logger(__name__)


def parse_sql_file(file_path: str) -> list[str]:
    """
    Parse a SQL file and extract individual SQL statements.

    Args:
        file_path (str): Path to the SQL file

    Returns:
        list: List of individual SQL statements
    """
    with open(file_path, "r") as file:
        content = file.read()

    # Split by semicolon followed by newline or end of string
    # This handles the common SQL statement terminator pattern
    statements = []
    current_statement = ""

    for line in content.splitlines():
        # Remove inline comments and leading/trailing whitespace
        line_without_comment = re.sub(r"--.*$", "", line).strip()
        if not line_without_comment:
            continue

        current_statement += line_without_comment + " "

        if line_without_comment.endswith(";"):
            # Clean up the statement and add to the list
            statements.append(current_statement.strip())
            current_statement = ""

    # Handle the case where the last statement doesn't end with a semicolon
    if current_statement.strip():
        statements.append(current_statement.strip())

    return statements


def _resolve_schema_path() -> str:
    """Resolve path to canonical schema.sql without env overrides.

    Policy:
    - When running from source, use repo-root schema.sql (../../schema.sql).
    - When running from an installed wheel, use packaged tensorcast/schema.sql.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_schema = os.path.abspath(os.path.join(current_dir, "..", "..", "schema.sql"))
    if os.path.isfile(root_schema):
        logger.info("Using canonical schema at repo root: schema.sql")
        return root_schema

    # Packaged resource fallback (installed wheel)
    with suppress(Exception):
        res = resources.files("tensorcast").joinpath("schema.sql")
        if res.is_file():
            with resources.as_file(res) as p:
                logger.info("Using packaged schema: tensorcast/schema.sql")
                return str(p)

    raise FileNotFoundError(
        "schema.sql not found. Ensure repo-root schema.sql exists or install a wheel that ships tensorcast/schema.sql."
    )


def _column_exists(db: DuckDBPyConnection, table: str, column: str) -> bool:
    try:
        rows = db.execute(f"PRAGMA table_info('{table}')").fetchall()
    except Exception:
        return False
    return any(row[1] == column for row in rows)


def _apply_schema_migrations(db: DuckDBPyConnection) -> None:
    # Drop legacy key_mappings.disk_path column if present
    if _column_exists(db, "key_mappings", "disk_path"):
        try:
            db.execute("ALTER TABLE key_mappings DROP COLUMN disk_path")
            logger.info("Dropped legacy key_mappings.disk_path column")
        except Exception:
            logger.exception("Failed to drop legacy key_mappings.disk_path column")
            raise

    # Soft-delete columns for managed shared-disk GC.
    if not _column_exists(db, "artifact_disk_locations", "is_deleted"):
        try:
            # DuckDB may reject NOT NULL for ADD COLUMN on some versions; keep it nullable and
            # treat NULL as false in queries.
            db.execute(
                "ALTER TABLE artifact_disk_locations ADD COLUMN is_deleted BOOLEAN DEFAULT FALSE"
            )
            logger.info("Added artifact_disk_locations.is_deleted column")
        except Exception:
            logger.exception("Failed to add artifact_disk_locations.is_deleted column")
            raise
    if not _column_exists(db, "artifact_disk_locations", "deleted_at"):
        try:
            db.execute(
                "ALTER TABLE artifact_disk_locations ADD COLUMN deleted_at TIMESTAMP WITH TIME ZONE"
            )
            logger.info("Added artifact_disk_locations.deleted_at column")
        except Exception:
            logger.exception("Failed to add artifact_disk_locations.deleted_at column")
            raise


def init_db(db: DuckDBPyConnection) -> None:
    sql_file_path = _resolve_schema_path()
    statements = parse_sql_file(sql_file_path)
    for statement in statements:
        if statement.strip():  # 只执行非空语句
            try:
                db.execute(statement)
                logger.debug(f"Executed SQL: {statement[:50]}...")
            except Exception:
                logger.exception("Failed to execute SQL statement: %s", statement)
                raise

    _apply_schema_migrations(db)


def optimize_db(db: DuckDBPyConnection) -> None:
    """Run periodic maintenance (VACUUM/OPTIMIZE) on hot tables.

    Currently targets `replica_counters`, which receives frequent updates.
    """
    try:
        db.execute("VACUUM replica_counters;")
        # Note: DuckDB doesn't support OPTIMIZE command like PostgreSQL
        # VACUUM already performs optimization for DuckDB
        logger.debug("Database maintenance (VACUUM) finished for replica_counters")
    except Exception:
        # Log but do not propagate to avoid crashing background thread
        logger.exception("Database maintenance failed")


if __name__ == "__main__":
    import duckdb

    db = duckdb.connect()
    init_db(db)
