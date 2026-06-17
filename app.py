# app.py — AskMyData
# Phase 1: Data ingestion + schema viewer       ✓
# Phase 2: FAISS RAG pipeline + retrieval       ✓
# Phase 3: LangChain agent + full Q&A           ✓
# Phase 4: SQL tool + ConversationMemory        ✓
# Phase 5: UI polish + Streamlit Cloud deploy   ← CURRENT

from datetime import datetime

import plotly.io as pio
import streamlit as st

from core.agent import AgentResponse, DataAnalysisAgent
from core.ingestion import (
    get_excel_sheets,
    get_sql_tables,
    load_sql,
    load_uploaded_file,
)
from core.llm import get_embeddings, get_llm
from core.memory import ChatMemory
from core.rag import RAGEngine
from core.schema import SchemaContext, extract_schema

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AskMyData",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Hide default Streamlit chrome ─── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Tighten top padding ──────────── */
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

/* ── Suggestion chips ─────────────── */
div[data-testid="column"] > div > div > div > button[kind="secondary"] {
    border-radius: 20px;
    font-size: 13px;
    padding: 4px 12px;
    border: 1px solid #E2E8F0;
    background: #F8FAFC;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Memory separator ─────────────── */
.mem-sep {
    text-align: center;
    font-size: 11px;
    color: #94A3B8;
    padding: 2px 0 6px 0;
    border-top: 1px dashed #CBD5E1;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "data_obj":       None,
    "schema_ctx":     None,
    "rag_engine":     None,
    "rag_chunk_count": 0,
    "rag_error":      None,
    "agent":          None,
    "agent_error":    None,
    "chat_memory":    None,
    "chat_history":   [],
    "pending_query":  None,     # set by suggestion buttons
    "query_result":   None,
    "llm_provider":   "Groq (free)",
    "memory_k":       5,
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_rag(data_obj, schema_ctx, provider: str) -> None:
    try:
        eng = RAGEngine(get_embeddings(provider))
        n   = eng.build(data_obj, schema_ctx)
        st.session_state.rag_engine      = eng
        st.session_state.rag_chunk_count = n
        st.session_state.rag_error       = None
    except Exception as exc:
        st.session_state.rag_engine      = None
        st.session_state.rag_chunk_count = 0
        st.session_state.rag_error       = str(exc)


def _build_agent(provider: str) -> None:
    try:
        st.session_state.agent       = DataAnalysisAgent(get_llm(provider))
        st.session_state.agent_error = None
    except Exception as exc:
        st.session_state.agent       = None
        st.session_state.agent_error = str(exc)


def _store(data_obj) -> None:
    schema_ctx = extract_schema(data_obj)
    st.session_state.data_obj     = data_obj
    st.session_state.schema_ctx   = schema_ctx
    st.session_state.chat_history = []
    st.session_state.pending_query = None
    st.session_state.query_result = None
    st.session_state.chat_memory  = ChatMemory(k=st.session_state.memory_k)
    _build_rag(data_obj, schema_ctx, st.session_state.llm_provider)
    _build_agent(st.session_state.llm_provider)


def _clear() -> None:
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v


# ── Suggestion generator ───────────────────────────────────────────────────────

def _suggestions(ctx: SchemaContext) -> list[str]:
    """
    Generate 4 relevant example questions from the schema.
    Prioritises the column types actually present in the dataset.
    """
    tips: list[str] = []

    num = ctx.numeric_cols
    cat = ctx.categorical_cols
    dt  = ctx.datetime_cols

    if num and cat:
        tips.append(f"What is the average {num[0]} by {cat[0]}?")
        tips.append(f"Which {cat[0]} has the highest total {num[0]}?")
    elif num:
        tips.append(f"What are the summary statistics for {num[0]}?")
        tips.append(f"Show the distribution of {num[0]}")

    if dt and num:
        tips.append(f"How does {num[0]} change over time ({dt[0]})?")
    elif cat:
        tips.append(f"What are the top 5 values in {cat[0]}?")

    generic = [
        "How many rows have missing values?",
        "Are there any outliers in the data?",
        "Show me the top 10 rows by highest value",
        "What is the overall count by category?",
    ]
    while len(tips) < 4:
        tips.append(generic[len(tips) % len(generic)])

    return tips[:4]


# ── Chat export ────────────────────────────────────────────────────────────────

def _export_chat(history: list[dict], dataset_name: str) -> bytes:
    """Render the chat history as a clean Markdown document."""
    lines = [
        "# AskMyData — Chat Export",
        f"**Dataset:** `{dataset_name}`",
        f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
    ]
    for msg in history:
        if msg["role"] == "user":
            lines += [f"**You:** {msg['content']}", ""]
        else:
            lines += [f"**Assistant:** {msg['content']}", ""]
            if msg.get("code"):
                lines += ["```python", msg["code"], "```", ""]
            lines += ["---", ""]
    return "\n".join(lines).encode("utf-8")


# ── Message renderer ───────────────────────────────────────────────────────────

def _render_assistant(msg: dict) -> None:
    st.markdown(msg["content"])
    if msg.get("error"):
        st.warning(f"⚠ Execution note: {msg['error']}")
    if msg.get("code"):
        with st.expander("🐍  Generated code", expanded=False):
            st.code(msg["code"], language="python")
    if msg.get("chart_json"):
        st.plotly_chart(
            pio.from_json(msg["chart_json"]),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔍 AskMyData")
    st.caption("Query your data in plain English")
    st.divider()

    # ── LLM selector ──────────────────────────────────────────────────────────
    st.markdown("**Model**")
    prev = st.session_state.llm_provider
    st.session_state.llm_provider = st.selectbox(
        "LLM",
        ["Groq (free)", "OpenAI GPT-4o"],
        label_visibility="collapsed",
        help=(
            "Groq  → Llama-3.3-70b-versatile | embeddings: all-MiniLM-L6-v2 (local)\n"
            "OpenAI → GPT-4o                  | embeddings: text-embedding-3-small"
        ),
    )
    if st.session_state.llm_provider != prev and st.session_state.data_obj:
        with st.spinner("Switching provider…"):
            _build_rag(
                st.session_state.data_obj,
                st.session_state.schema_ctx,
                st.session_state.llm_provider,
            )
            _build_agent(st.session_state.llm_provider)

    st.divider()

    # ── Data source ───────────────────────────────────────────────────────────
    st.markdown("**Data source**")
    source_mode = st.radio(
        "Mode", ["Upload file", "SQLite DB"],
        horizontal=True, label_visibility="collapsed",
    )

    if source_mode == "Upload file":
        uploaded  = st.file_uploader(
            "Upload", type=["csv", "xlsx", "xls", "json"],
            label_visibility="collapsed",
        )
        sheet_idx = 0
        if uploaded:
            if uploaded.name.lower().endswith((".xlsx", ".xls")):
                try:
                    sheets = get_excel_sheets(uploaded)
                    uploaded.seek(0)
                    if len(sheets) > 1:
                        chosen    = st.selectbox("Sheet", sheets)
                        sheet_idx = sheets.index(chosen)
                except Exception:
                    sheet_idx = 0
            with st.spinner("Loading and indexing data…"):
                try:
                    _store(load_uploaded_file(uploaded, sheet_name=sheet_idx))
                except Exception as exc:
                    st.error(f"Load error: {exc}")

        if st.session_state.data_obj:
            obj = st.session_state.data_obj
            st.success(f"✓ {obj.source_name}")
            st.caption(f"{obj.row_count:,} rows · {obj.col_count} columns")

    else:
        db_path = st.text_input(
            "DB path", placeholder="/path/to/database.db",
            label_visibility="collapsed",
        )
        if db_path:
            try:
                tables = get_sql_tables(db_path)
                if tables:
                    chosen_table = st.selectbox("Table", tables)
                    if st.button("Load table", use_container_width=True):
                        with st.spinner("Loading and indexing table…"):
                            try:
                                _store(load_sql(db_path, chosen_table))
                            except Exception as exc:
                                st.error(f"Load error: {exc}")
                else:
                    st.warning("No tables found.")
            except Exception as exc:
                st.error(f"Cannot connect: {exc}")

    # ── Memory controls ────────────────────────────────────────────────────────
    if st.session_state.chat_memory is not None:
        st.divider()
        st.markdown("**Memory**")
        mem   = st.session_state.chat_memory
        new_k = st.slider(
            "Window (exchanges)",
            min_value=2, max_value=10, value=mem.window_size,
            help="Past Q&A exchanges the agent can reference in follow-ups.",
        )
        if new_k != mem.window_size:
            mem.resize(new_k)
            st.session_state.memory_k = new_k

        in_mem = mem.exchange_count
        if in_mem == 0:
            st.caption("No history yet.")
        else:
            st.progress(
                in_mem / mem.window_size,
                text=f"{in_mem} / {mem.window_size} exchanges in memory",
            )
        if not mem.is_empty:
            if st.button("Clear memory", use_container_width=True):
                mem.clear()
                st.rerun()

    # ── Status + clear ─────────────────────────────────────────────────────────
    if st.session_state.data_obj:
        st.divider()
        if st.session_state.rag_engine:
            st.success(f"✓ RAG · {st.session_state.rag_chunk_count} chunks")
        else:
            st.warning(f"⚠ RAG: {st.session_state.rag_error}")
        if st.session_state.agent:
            st.success(f"✓ Agent · {st.session_state.llm_provider}")
        else:
            st.warning(f"⚠ Agent: {st.session_state.agent_error}")
        if st.button("Clear dataset", use_container_width=True, type="secondary"):
            _clear()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE  (no data loaded)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.data_obj is None:
    st.markdown(
        "<h1 style='margin-bottom:0'>🔍 AskMyData</h1>"
        "<p style='color:#64748B; font-size:18px; margin-top:4px'>"
        "Upload any dataset and query it in plain English.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.info("📂 **Upload**\nCSV, Excel, JSON or SQLite")
    c2.info("🔍 **RAG**\nSchema embedded into FAISS")
    c3.info("🤖 **Agent**\nAuto-generates pandas or SQL")
    c4.info("📊 **Output**\nAnswer · Code · Chart")

    st.divider()
    st.markdown("#### Supported formats")
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.success("`.csv`")
    fc2.success("`.xlsx` / `.xls`")
    fc3.success("`.json`")
    fc4.success("`.db` / `.sqlite`")

    st.divider()
    st.markdown(
        "← **Upload a file from the sidebar** to start asking questions.",
        unsafe_allow_html=False,
    )

    # Footer
    st.markdown(
        "<div style='position:fixed; bottom:12px; right:20px; "
        "font-size:11px; color:#94A3B8;'>"
        "Built with LangChain · FAISS · Streamlit · Plotly"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA  (data loaded)
# ══════════════════════════════════════════════════════════════════════════════

obj = st.session_state.data_obj
ctx = st.session_state.schema_ctx
mem = st.session_state.chat_memory

# ── Header ─────────────────────────────────────────────────────────────────────
h_col, m1, m2, m3 = st.columns([5, 1, 1, 1])
with h_col:
    st.markdown(f"### {obj.table_name}")
    st.caption(
        f"{obj.source_type.upper()} · {obj.row_count:,} rows · "
        f"{obj.col_count} cols · {st.session_state.llm_provider}"
    )
m1.metric("Numeric",     len(ctx.numeric_cols))
m2.metric("Categorical", len(ctx.categorical_cols))
m3.metric("Datetime",    len(ctx.datetime_cols))
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_chat, tab_preview, tab_schema, tab_rag = st.tabs([
    "💬  Ask a question",
    "📋  Data preview",
    "🗂  Schema",
    "🔍  RAG index",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════

with tab_chat:

    # Guard
    if not st.session_state.rag_engine or not st.session_state.agent:
        errors = []
        if not st.session_state.rag_engine:
            errors.append(f"RAG: {st.session_state.rag_error}")
        if not st.session_state.agent:
            errors.append(f"Agent: {st.session_state.agent_error}")
        for err in errors:
            st.error(err)
        st.info("Add your API key to `.streamlit/secrets.toml` then reload the dataset.")
        st.stop()

    # ── Export button (top-right when history exists) ──────────────────────────
    chat_history = st.session_state.chat_history
    if chat_history:
        _, export_col = st.columns([8, 2])
        with export_col:
            st.download_button(
                "📥 Export chat",
                data=_export_chat(chat_history, obj.table_name),
                file_name=f"askmydata_{obj.table_name}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # ── Memory window marker ───────────────────────────────────────────────────
    memory_start = max(0, len(chat_history) - mem.window_size * 2)

    # ── Render existing chat history ───────────────────────────────────────────
    for i, msg in enumerate(chat_history):
        if i == memory_start and memory_start > 0:
            st.markdown(
                "<div class='mem-sep'>↑ older messages — outside memory window</div>",
                unsafe_allow_html=True,
            )
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                _render_assistant(msg)

    # ── Suggestion chips (shown only before first message) ─────────────────────
    if not chat_history:
        st.markdown(
            "<p style='color:#64748B; font-size:13px; margin:12px 0 6px'>Try asking:</p>",
            unsafe_allow_html=True,
        )
        sugs = _suggestions(ctx)
        cols = st.columns(len(sugs))
        for col, sug in zip(cols, sugs):
            if col.button(sug, use_container_width=True, type="secondary"):
                st.session_state.pending_query = sug
                st.rerun()

    # ── Resolve active prompt: pending (suggestion) OR typed input ─────────────
    typed   = st.chat_input("Ask something about your data…")
    pending = st.session_state.pop("pending_query", None)
    prompt  = pending or typed

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.status("Analysing…", expanded=True) as status:

                status.write("🔍  Retrieving schema context from FAISS…")
                try:
                    rag_result = st.session_state.rag_engine.retrieve(prompt, k=5)
                except Exception as exc:
                    st.error(f"RAG error: {exc}")
                    st.stop()

                status.write("🧠  Generating pandas / SQL code…")
                status.write("⚙️  Executing in secure sandbox…")

                response: AgentResponse = st.session_state.agent.run(
                    query        = prompt,
                    rag_context  = rag_result.context,
                    df           = obj.df,
                    table_name   = obj.table_name,
                    chat_history = mem.get_history_str(),
                )

                status.update(
                    label    = "Done!" if response.success else "Completed with issues",
                    state    = "complete" if response.success else "error",
                    expanded = False,
                )

            st.markdown(response.answer)
            if response.error and not response.success:
                st.warning(f"⚠ Execution note: {response.error}")
            if response.code:
                with st.expander("🐍  Generated code", expanded=False):
                    st.code(response.code, language="python")
            chart_json = None
            if response.chart:
                chart_json = pio.to_json(response.chart)
                st.plotly_chart(response.chart, use_container_width=True)

        mem.add_exchange(prompt, response.answer)
        chat_history.append({
            "role":       "assistant",
            "content":    response.answer,
            "code":       response.code,
            "chart_json": chart_json,
            "error":      response.error if not response.success else "",
        })

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center; color:#94A3B8; font-size:11px; "
        "padding-top:20px;'>"
        "LangChain · FAISS · Streamlit · Plotly</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA PREVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_preview:
    st.caption(f"Showing first 100 of {obj.row_count:,} rows.")
    st.dataframe(obj.df.head(100), use_container_width=True, height=460)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

with tab_schema:
    st.caption("Auto-extracted from your dataset — embedded into FAISS per column.")
    st.code(ctx.schema_text, language="text")
    with st.expander("Sample rows (also embedded)"):
        st.code(ctx.sample_text, language="text")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RAG INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════

with tab_rag:
    if not st.session_state.rag_engine:
        st.error(f"RAG not built: {st.session_state.rag_error}")
        st.stop()

    engine     = st.session_state.rag_engine
    all_chunks = engine.get_all_chunks()
    st.caption(
        f"{len(all_chunks)} FAISS chunks — "
        f"1 metadata + {len(ctx.column_names)} columns + 1 sample rows"
    )

    _ICONS  = {"metadata": "📊", "column": "📋", "sample_rows": "🔢"}
    _LABELS = {"metadata": "Dataset overview", "column": "Column detail",
               "sample_rows": "Sample rows"}

    for i, (ctype, content) in enumerate(all_chunks):
        col_label = (
            " — " + content.split("\n")[0].replace("Column: ", "")
            if ctype == "column" else ""
        )
        with st.expander(
            f"{_ICONS.get(ctype,'📄')}  Chunk {i+1}: "
            f"{_LABELS.get(ctype, ctype)}{col_label}"
        ):
            st.code(content, language="text")
