# core/rag.py
# ─────────────────────────────────────────────────────────
# FAISS RAG engine for AskMyData.
#
# Chunking strategy — one LangChain Document per element:
#   1. Metadata doc   (1)  — table name, shape, column categories
#   2. Column docs    (N)  — one per column with dtype, stats, top values, code
#   3. Sample rows    (1)  — first 5 rows formatted as text
#
# Why per-column chunks?
#   FAISS matches "average revenue by region?" → Revenue + Region column chunks
#   with high precision, instead of scanning the entire schema string.
#
# The retrieved `RetrievalResult.context` is injected verbatim into the
# LangChain agent's system prompt in Phase 3.
# ─────────────────────────────────────────────────────────

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from core.ingestion import DataObject
from core.schema import SchemaContext


# ── Output container ───────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Structured output from a single RAG retrieval call."""
    query:       str
    chunks:      list[str]   # raw page_content of each retrieved Document
    chunk_types: list[str]   # "metadata" | "column" | "sample_rows"
    context:     str         # merged, labelled string → injected into agent prompt


# ── Engine ─────────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    FAISS-backed RAG engine for tabular schema retrieval.

    Typical lifecycle:
        engine = RAGEngine(get_embeddings(provider))
        n = engine.build(data_obj, schema_ctx)     # embed all chunks
        result = engine.retrieve("show me avg revenue by region", k=5)
        print(result.context)                       # send to LLM in Phase 3
    """

    def __init__(self, embeddings):
        self._embeddings = embeddings
        self._vectorstore: FAISS | None = None
        self._docs: list[Document] = []
        self._table_name: str = ""

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True once build() has been called successfully."""
        return self._vectorstore is not None

    def chunk_count(self) -> int:
        """Total number of chunks embedded in the FAISS index."""
        return len(self._docs)

    def get_all_chunks(self) -> list[tuple[str, str]]:
        """
        Return all indexed chunks as (chunk_type, page_content) tuples.
        Used by the RAG Inspector tab in the UI.
        """
        return [
            (doc.metadata.get("chunk_type", "unknown"), doc.page_content)
            for doc in self._docs
        ]

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, data_obj: DataObject, schema_ctx: SchemaContext) -> int:
        """
        Create all schema chunks and embed them into an in-memory FAISS index.

        Args:
            data_obj:   DataObject from the ingestion layer
            schema_ctx: SchemaContext from the schema extractor

        Returns:
            Number of chunks embedded (1 metadata + N columns + 1 sample rows)
        """
        self._table_name  = data_obj.table_name
        self._docs        = self._create_chunks(data_obj, schema_ctx)
        self._vectorstore = FAISS.from_documents(self._docs, self._embeddings)
        return len(self._docs)

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """
        Find the top-k schema chunks most semantically similar to `query`.

        Args:
            query: plain-English question from the user
            k:     number of chunks to retrieve (capped at total chunk count)

        Returns:
            RetrievalResult — pass `result.context` to the LangChain agent.
        """
        if not self.is_ready:
            raise RuntimeError(
                "RAGEngine not initialised. Call build() before retrieve()."
            )

        k_eff = min(k, len(self._docs))
        docs  = self._vectorstore.similarity_search(query, k=k_eff)

        chunks      = [d.page_content                       for d in docs]
        chunk_types = [d.metadata.get("chunk_type", "?")    for d in docs]
        context     = self._format_context(chunks, chunk_types)

        return RetrievalResult(
            query=query,
            chunks=chunks,
            chunk_types=chunk_types,
            context=context,
        )

    # ── Internal: chunking ─────────────────────────────────────────────────────

    def _create_chunks(
        self,
        data_obj:   DataObject,
        schema_ctx: SchemaContext,
    ) -> list[Document]:
        """
        Convert a DataObject + SchemaContext into embeddable LangChain Documents.

        Document count = 1 (metadata) + len(columns) + 1 (sample rows)
        """
        docs:  list[Document] = []
        df    = data_obj.df
        table = data_obj.table_name

        # ── 1. Table metadata ──────────────────────────────────────────────
        docs.append(Document(
            page_content=(
                f"Dataset overview: {table}\n"
                f"Source type: {data_obj.source_type.upper()}\n"
                f"Total rows: {data_obj.row_count:,}\n"
                f"Total columns: {data_obj.col_count}\n"
                f"All column names: {', '.join(schema_ctx.column_names)}\n"
                f"Numeric columns: "
                f"{', '.join(schema_ctx.numeric_cols) or 'none'}\n"
                f"Categorical columns: "
                f"{', '.join(schema_ctx.categorical_cols) or 'none'}\n"
                f"Datetime columns: "
                f"{', '.join(schema_ctx.datetime_cols) or 'none'}"
            ),
            metadata={"chunk_type": "metadata", "table": table},
        ))

        # ── 2. Per-column documents ────────────────────────────────────────
        for col in schema_ctx.column_names:
            s        = df[col]
            null_n   = int(s.isnull().sum())
            null_pct = (null_n / data_obj.row_count * 100
                        if data_obj.row_count else 0.0)
            unique_n = int(s.nunique(dropna=True))

            if col in schema_ctx.numeric_cols:
                kind  = "numeric"
                clean = s.dropna()
                extra = (
                    f"Statistics: "
                    f"min={clean.min():.4g}, max={clean.max():.4g}, "
                    f"mean={clean.mean():.4g}, std={clean.std():.4g}"
                )
            elif col in schema_ctx.datetime_cols:
                kind  = "datetime"
                svals = ", ".join(str(v) for v in s.dropna().head(3).tolist())
                extra = f"Sample values: {svals}"
            else:
                kind = "categorical"
                top  = s.value_counts().head(5).index.tolist()
                extra = (
                    f"Top values: {', '.join(str(v) for v in top)}"
                )

            docs.append(Document(
                page_content=(
                    f"Column: {col}\n"
                    f"Table: {table}\n"
                    f"Type: {kind}\n"
                    f"dtype: {s.dtype}\n"
                    f"Null values: {null_n} ({null_pct:.1f}%)\n"
                    f"Unique values: {unique_n}\n"
                    f"{extra}\n"
                    f"Pandas access: df['{col}']\n"
                    f"SQL access: SELECT \"{col}\" FROM {table}"
                ),
                metadata={
                    "chunk_type": "column",
                    "column":     col,
                    "table":      table,
                },
            ))

        # ── 3. Sample rows ─────────────────────────────────────────────────
        docs.append(Document(
            page_content=(
                f"Sample rows from {table} (first 5 rows):\n"
                + df.head(5).to_string(index=False, max_colwidth=40)
            ),
            metadata={"chunk_type": "sample_rows", "table": table},
        ))

        return docs

    # ── Internal: context formatting ───────────────────────────────────────────

    @staticmethod
    def _format_context(chunks: list[str], chunk_types: list[str]) -> str:
        """
        Merge retrieved chunks into a single labelled context string.

        Format injected into the LangChain agent prompt (Phase 3):
            [DATASET OVERVIEW]
            Dataset overview: sales_data ...

            ---

            [COLUMN DETAIL]
            Column: Revenue ...

            ---

            [SAMPLE DATA]
            Sample rows from sales_data ...
        """
        _labels = {
            "metadata":    "DATASET OVERVIEW",
            "column":      "COLUMN DETAIL",
            "sample_rows": "SAMPLE DATA",
        }
        parts = [
            f"[{_labels.get(ctype, 'CONTEXT')}]\n{chunk}"
            for chunk, ctype in zip(chunks, chunk_types)
        ]
        return "\n\n---\n\n".join(parts)
