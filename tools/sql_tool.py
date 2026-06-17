# tools/sql_tool.py
# ─────────────────────────────────────────────────────────
# SQL execution tool for AskMyData.
#
# Loads a pandas DataFrame into an in-memory SQLite database
# and exposes a callable `run_sql(query)` that the LLM-generated
# code can call alongside pandas operations.
#
# Why in-memory SQLite?
#   - Works with ANY data source (CSV, Excel, JSON, SQL files)
#   - No extra dependencies beyond stdlib sqlite3
#   - Zero setup — the table is created from the DataFrame on init
#   - Safe — isolated per execution, closed after use
#
# Usage in LLM-generated code:
#   result = run_sql("SELECT Region, AVG(Revenue) FROM sales_data GROUP BY Region")
#   result = run_sql(f"SELECT * FROM {table_name} WHERE Revenue > 1000 LIMIT 10")
# ─────────────────────────────────────────────────────────

import sqlite3
from typing import Optional

import pandas as pd


class SQLTool:
    """
    Callable SQL executor backed by an in-memory SQLite database.

    The tool is designed to be injected into the SafeExecutor's namespace
    as `run_sql`. LLM-generated code can then call it like a function:

        result = run_sql("SELECT Product, SUM(Revenue) FROM sales GROUP BY Product")

    Lifecycle: one SQLTool is created per executor.execute() call, lives
    for the duration of that call, then is garbage-collected automatically.

    Thread safety: NOT thread-safe (each call should create its own instance).
    """

    def __init__(self, df: pd.DataFrame, table_name: str) -> None:
        """
        Build the in-memory SQLite database from a DataFrame.

        Args:
            df:         The dataset (a copy — the original is never modified)
            table_name: SQL table name (e.g. "sales_data")
        """
        self.table_name = table_name
        self._conn      = sqlite3.connect(":memory:")
        df.to_sql(table_name, self._conn, index=False, if_exists="replace")

    # ── Main interface ─────────────────────────────────────────────────────────

    def __call__(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL SELECT query and return the result as a DataFrame.

        Args:
            sql: SQL query string. Must be a SELECT statement.
                 Use `self.table_name` or the variable `table_name`
                 (available in the sandbox) for the table name.

        Returns:
            pd.DataFrame — the query result

        Raises:
            sqlite3.OperationalError / pd.io.sql.DatabaseError on bad SQL.
        """
        return pd.read_sql_query(sql, self._conn)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def schema(self) -> str:
        """
        Return a compact CREATE TABLE-style schema string.
        Useful for debugging / inspecting the loaded table.
        """
        cursor = self._conn.cursor()
        cursor.execute(f"PRAGMA table_info({self.table_name})")
        cols = cursor.fetchall()
        if not cols:
            return f"Table '{self.table_name}' not found."
        col_strs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        return f"CREATE TABLE {self.table_name} ({col_strs})"

    def row_count(self) -> int:
        """Quick row count of the loaded table."""
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Explicitly close the SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
