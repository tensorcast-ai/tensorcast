#  Copyright (c) 2025, TensorCast Team.

import os
import re

import duckdb
from duckdb import DuckDBPyConnection

from tensorcast.logger import init_logger

logger = init_logger(__name__)


def parse_sql_file(file_path):
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


def init_db(db: DuckDBPyConnection):
    # Check if tables already exist in this specific connection
    res = db.execute("SHOW TABLES").fetchall()
    if len(res) > 0:
        logger.info("Database already initialized")
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sql_file_path = os.path.join(current_dir, "init.sql")
    statements = parse_sql_file(sql_file_path)
    for statement in statements:
        if statement.strip():  # 只执行非空语句
            try:
                db.execute(statement)
                logger.debug(f"Executed SQL: {statement[:50]}...")
            except Exception as e:
                logger.error(f"Failed to execute SQL statement: {statement}")
                logger.error(f"Error: {e}")
                raise


def optimize_db(db: DuckDBPyConnection):
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
