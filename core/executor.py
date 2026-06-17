# core/executor.py
# ─────────────────────────────────────────────────────────
# Safe execution sandbox for LLM-generated Python / pandas / SQL code.
#
# Phase 4 additions vs Phase 3:
#   • run_sql(query)   — callable SQL tool injected into namespace
#   • table_name       — string variable available in the sandbox
#     LLM code can now choose EITHER pandas (df) OR SQL (run_sql)
#
# Safety measures unchanged:
#   • Restricted __builtins__  — blocks os, sys, subprocess, eval, etc.
#   • DataFrame copy           — generated code cannot mutate source data
#   • Stdout capture           — print() calls returned, not leaked
#   • Exception isolation      — any error caught and returned cleanly
#   • Result capping           — DataFrames capped at MAX_RESULT_ROWS
# ─────────────────────────────────────────────────────────

import math
import re
import statistics
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd

from tools.sql_tool import SQLTool


# ── Safe builtins whitelist ────────────────────────────────────────────────────
_SAFE_BUILTINS: dict[str, Any] = {
    "bool":       bool,       "dict":       dict,     "float":     float,
    "frozenset":  frozenset,  "int":        int,      "list":      list,
    "set":        set,        "str":        str,      "tuple":     tuple,
    "all":        all,        "any":        any,      "enumerate": enumerate,
    "filter":     filter,     "iter":       iter,     "map":       map,
    "next":       next,       "range":      range,    "reversed":  reversed,
    "sorted":     sorted,     "zip":        zip,
    "abs":        abs,        "divmod":     divmod,   "format":    format,
    "max":        max,        "min":        min,      "pow":       pow,
    "round":      round,      "sum":        sum,
    "hasattr":    hasattr,    "isinstance": isinstance, "issubclass": issubclass,
    "len":        len,        "print":      print,    "repr":      repr,
    "slice":      slice,      "type":       type,
    "True":       True,       "False":      False,    "None":      None,
}


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """
    Output of one SafeExecutor.execute() call.

    result:  Value of the `result` variable after execution (None if unset)
    output:  Captured stdout — print() calls from the generated code
    error:   Exception type + message (empty if execution succeeded)
    success: False if any exception was raised during exec
    """
    result:  Any  = None
    output:  str  = ""
    error:   str  = ""
    success: bool = True


# ── Executor ───────────────────────────────────────────────────────────────────

class SafeExecutor:
    """
    Executes LLM-generated Python/pandas/SQL code in a restricted namespace.

    The generated code can use ANY combination of:
        df          — pandas DataFrame (copy of the user's data)
        pd, np      — pandas and numpy
        run_sql(q)  — execute SQL against the same data (returns DataFrame)
        table_name  — string name of the table (for SQL queries)
        math, re, statistics — standard library modules

    All results must be stored in `result`.

    Examples of valid generated code:
        # Pandas approach
        result = df.groupby('Region')['Revenue'].mean().reset_index()

        # SQL approach
        result = run_sql(f"SELECT Region, AVG(Revenue) FROM {table_name} GROUP BY Region")

        # Mixed approach
        top = run_sql("SELECT Region FROM sales ORDER BY Revenue DESC LIMIT 1")
        result = top.iloc[0, 0]
    """

    MAX_RESULT_ROWS = 500

    def execute(
        self,
        code:       str,
        df:         pd.DataFrame,
        table_name: str = "data",
    ) -> ExecutionResult:
        """
        Run code in the sandbox with both pandas (df) and SQL (run_sql) available.

        Args:
            code:       Python code. Must store the answer in `result`.
            df:         Dataset. A copy is used — original is protected.
            table_name: SQL table name available as `table_name` in the code.

        Returns:
            ExecutionResult — result value, stdout, and error if any.
        """
        df_copy  = df.copy()
        sql_tool = SQLTool(df_copy, table_name)

        namespace: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            # ── Data access ───────────────────────────────────────────────
            "df":           df_copy,
            "run_sql":      sql_tool,       # callable: run_sql("SELECT ...")
            "table_name":   table_name,     # string: use in f-strings
            # ── Libraries ─────────────────────────────────────────────────
            "pd":           pd,
            "np":           np,
            "math":         math,
            "re":           re,
            "statistics":   statistics,
            # ── Answer slot ───────────────────────────────────────────────
            "result":       None,
        }

        stdout_buf = StringIO()
        try:
            with redirect_stdout(stdout_buf):
                exec(compile(code, "<askmydata_sandbox>", "exec"), namespace)  # noqa: S102

            raw    = namespace.get("result")
            result = self._cap(raw)

            return ExecutionResult(
                result  = result,
                output  = stdout_buf.getvalue().strip(),
                success = True,
            )

        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                result  = None,
                output  = stdout_buf.getvalue().strip(),
                error   = f"{type(exc).__name__}: {exc}",
                success = False,
            )
        finally:
            sql_tool.close()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _cap(self, raw: Any) -> Any:
        """Cap large DataFrames / Series to avoid bloating session state."""
        if isinstance(raw, pd.DataFrame) and len(raw) > self.MAX_RESULT_ROWS:
            return raw.head(self.MAX_RESULT_ROWS)
        if isinstance(raw, pd.Series) and len(raw) > self.MAX_RESULT_ROWS:
            return raw.head(self.MAX_RESULT_ROWS)
        return raw
