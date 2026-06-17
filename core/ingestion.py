# core/ingestion.py
# ─────────────────────────────────────────────────────────
# Unified data ingestion layer for AskMyData.
# Supports: CSV, Excel (.xlsx/.xls), JSON, SQLite
# Returns a DataObject that every downstream layer consumes.
# ─────────────────────────────────────────────────────────

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


# ── Data container ─────────────────────────────────────────────────────────────

@dataclass
class DataObject:
    """
    Unified container returned by every loader.
    All downstream layers (schema, RAG, agent) consume this object.
    """
    df:           pd.DataFrame
    source_type:  str          # "csv" | "excel" | "json" | "sql"
    source_name:  str          # original filename or DB path
    table_name:   str          # logical name used in prompts and schema
    row_count:    int
    col_count:    int
    excel_sheets: list = field(default_factory=list)  # available sheets (Excel only)
    sql_tables:   list = field(default_factory=list)  # available tables  (SQL only)

    def __repr__(self):
        return (
            f"DataObject(table={self.table_name!r}, "
            f"type={self.source_type}, "
            f"rows={self.row_count:,}, cols={self.col_count})"
        )


# ── CSV ────────────────────────────────────────────────────────────────────────

def load_csv(file, **kwargs) -> DataObject:
    """
    Load a CSV file.
    Falls back to latin-1 encoding if UTF-8 fails (handles many real-world CSVs).
    `file` can be a Streamlit UploadedFile or a file path string.
    """
    name = getattr(file, "name", str(file))
    try:
        df = pd.read_csv(file, **kwargs)
    except UnicodeDecodeError:
        if hasattr(file, "seek"):
            file.seek(0)
        df = pd.read_csv(file, encoding="latin-1", **kwargs)

    return DataObject(
        df=df,
        source_type="csv",
        source_name=name,
        table_name=Path(name).stem,
        row_count=len(df),
        col_count=len(df.columns),
    )


# ── Excel ──────────────────────────────────────────────────────────────────────

def get_excel_sheets(file) -> list[str]:
    """
    Return all sheet names from an Excel file without loading data.
    Call this before load_excel to populate a sheet selector in the UI.
    """
    return pd.ExcelFile(file).sheet_names


def load_excel(file, sheet_name: int | str = 0) -> DataObject:
    """
    Load one sheet from an Excel file (.xlsx / .xls).
    `sheet_name` can be a 0-based index or the sheet's string name.
    """
    xl = pd.ExcelFile(file)
    sheets = xl.sheet_names
    df = xl.parse(sheet_name)

    name = getattr(file, "name", str(file))
    chosen = sheets[sheet_name] if isinstance(sheet_name, int) else sheet_name

    return DataObject(
        df=df,
        source_type="excel",
        source_name=name,
        table_name=f"{Path(name).stem}_{chosen}",
        row_count=len(df),
        col_count=len(df.columns),
        excel_sheets=sheets,
    )


# ── JSON ───────────────────────────────────────────────────────────────────────

def load_json(file) -> DataObject:
    """
    Load a JSON file. Handles three common shapes:
      - Array of records:   [{...}, {...}]          (most common)
      - Column-oriented:    {"col1": [...], ...}
      - Nested records:     flattened with pd.json_normalize
    """
    name = getattr(file, "name", str(file))
    raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
    data = json.loads(raw)

    if isinstance(data, list):
        # Records format — use json_normalize to flatten nested keys
        df = pd.json_normalize(data)

    elif isinstance(data, dict):
        # Try column-oriented (keys = columns, values = lists of equal length)
        try:
            df = pd.DataFrame(data)
        except ValueError:
            # Single record dict — wrap in list
            df = pd.json_normalize([data])
    else:
        raise ValueError(
            "JSON must be a list of records [{...}] or a column-oriented dict. "
            f"Got: {type(data).__name__}"
        )

    return DataObject(
        df=df,
        source_type="json",
        source_name=name,
        table_name=Path(name).stem,
        row_count=len(df),
        col_count=len(df.columns),
    )


# ── SQLite ─────────────────────────────────────────────────────────────────────

def get_sql_tables(db_path: str) -> list[str]:
    """
    Return all user-created table names from a SQLite database.
    Excludes SQLite internal tables (sqlite_%).
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name;"
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def load_sql(db_path: str, table_name: str, limit: int = 10_000) -> DataObject:
    """
    Load a table from a SQLite database.
    Caps at `limit` rows (default 10 000) to prevent memory issues.
    For full-table queries the agent will generate SQL directly.
    """
    tables = get_sql_tables(db_path)
    if table_name not in tables:
        raise ValueError(
            f"Table '{table_name}' not found in {db_path}. "
            f"Available: {tables}"
        )

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            f'SELECT * FROM "{table_name}" LIMIT {limit}', conn
        )
    finally:
        conn.close()

    return DataObject(
        df=df,
        source_type="sql",
        source_name=db_path,
        table_name=table_name,
        row_count=len(df),
        col_count=len(df.columns),
        sql_tables=tables,
    )


# ── Unified router ─────────────────────────────────────────────────────────────

def load_uploaded_file(uploaded_file, sheet_name: int | str = 0) -> DataObject:
    """
    Route a Streamlit UploadedFile to the correct loader by file extension.
    Supported: .csv  .xlsx  .xls  .json

    Args:
        uploaded_file:  Streamlit UploadedFile object
        sheet_name:     Sheet index or name (only used for Excel files)

    Returns:
        DataObject ready for schema extraction and RAG ingestion.
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return load_csv(uploaded_file)

    elif name.endswith((".xlsx", ".xls")):
        return load_excel(uploaded_file, sheet_name=sheet_name)

    elif name.endswith(".json"):
        return load_json(uploaded_file)

    else:
        ext = Path(name).suffix or "(no extension)"
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            "Supported extensions: .csv, .xlsx, .xls, .json"
        )
