# core/schema.py
# ─────────────────────────────────────────────────────────
# Schema extractor for AskMyData.
# Takes a DataObject and produces a SchemaContext:
#   - schema_text       : human-readable column-by-column description
#   - sample_text       : formatted first N rows
#   - combined_context  : both merged — this is what FAISS embeds in Phase 2
# ─────────────────────────────────────────────────────────

from dataclasses import dataclass

import pandas as pd

from core.ingestion import DataObject


# ── Output container ───────────────────────────────────────────────────────────

@dataclass
class SchemaContext:
    """
    Rich schema description produced from a DataObject.
    `combined_context` is the document that gets embedded into FAISS (Phase 2)
    and injected into every agent prompt (Phase 3).
    """
    schema_text:      str         # full schema description
    sample_text:      str         # first N rows as readable text
    combined_context: str         # schema + samples merged
    column_names:     list[str]
    dtypes:           dict[str, str]
    numeric_cols:     list[str]
    categorical_cols: list[str]
    datetime_cols:    list[str]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _infer_kind(series: pd.Series) -> str:
    """
    Classify a column as 'numeric', 'datetime', or 'categorical'.
    Attempts to parse object columns as datetime before falling back to categorical.
    """
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if series.dtype == object:
        sample = series.dropna().head(20)
        try:
            pd.to_datetime(sample, infer_datetime_format=True)
            return "datetime"
        except Exception:
            pass
    return "categorical"


def _numeric_stats(series: pd.Series) -> str:
    """Compact stats string for numeric columns."""
    try:
        clean = series.dropna()
        return (
            f"min={clean.min():.4g}, max={clean.max():.4g}, "
            f"mean={clean.mean():.4g}, std={clean.std():.4g}"
        )
    except Exception:
        return "stats unavailable"


def _top_values(series: pd.Series, n: int = 5) -> str:
    """Top N most frequent values for categorical columns."""
    top = series.value_counts().head(n).index.tolist()
    return ", ".join(str(v) for v in top)


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_schema(data_obj: DataObject, sample_rows: int = 5) -> SchemaContext:
    """
    Build a rich SchemaContext from a DataObject.

    Args:
        data_obj:    Source data (any type — CSV, Excel, JSON, SQL)
        sample_rows: Number of sample rows to include in context

    Returns:
        SchemaContext with schema_text, sample_text, combined_context,
        and column classification lists (numeric / categorical / datetime).

    Example combined_context (used for FAISS embedding in Phase 2):
        Table: sales_data
        Source: CSV | Rows: 5,000 | Columns: 8
        Columns:
          OrderDate | datetime | ...
          Revenue   | numeric  | min=10, max=5000, mean=320 ...
          Region    | categorical | top values: North, South, East ...
        Sample data (5 rows):
          OrderDate   Revenue  Region
          2023-01-01  450.00   North
          ...
    """
    df = data_obj.df
    n = len(df)

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    datetime_cols: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────────
    lines = [
        f"Table: {data_obj.table_name}",
        f"Source: {data_obj.source_type.upper()}  |  "
        f"Rows: {n:,}  |  Columns: {data_obj.col_count}",
        "",
        "Columns:",
    ]

    # ── Per-column detail ───────────────────────────────────────────────────────
    for col in df.columns:
        s = df[col]
        null_n   = int(s.isnull().sum())
        null_pct = null_n / n * 100 if n > 0 else 0.0
        unique_n = int(s.nunique(dropna=True))
        kind     = _infer_kind(s)

        if kind == "numeric":
            numeric_cols.append(col)
            extra = f"stats: [{_numeric_stats(s)}]"

        elif kind == "datetime":
            datetime_cols.append(col)
            sample_vals = ", ".join(str(v) for v in s.dropna().head(3).tolist())
            extra = f"sample values: [{sample_vals}]"

        else:  # categorical
            categorical_cols.append(col)
            extra = f"top values: [{_top_values(s)}]"

        lines.append(
            f"  {col}"
            f"  |  kind={kind}"
            f"  |  dtype={s.dtype}"
            f"  |  nulls={null_n} ({null_pct:.1f}%)"
            f"  |  unique={unique_n}"
            f"  |  {extra}"
        )

    schema_text = "\n".join(lines)

    # ── Sample rows ─────────────────────────────────────────────────────────────
    sample_df = df.head(sample_rows)
    sample_text = (
        f"\nSample data ({sample_rows} rows):\n"
        + sample_df.to_string(index=False, max_colwidth=40)
    )

    combined_context = schema_text + "\n" + sample_text

    return SchemaContext(
        schema_text=schema_text,
        sample_text=sample_text,
        combined_context=combined_context,
        column_names=list(df.columns),
        dtypes={col: str(df[col].dtype) for col in df.columns},
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
    )
