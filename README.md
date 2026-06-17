# 🔍 AskMyData

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?logo=chainlink&logoColor=white)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-RAG-0064B5)](https://faiss.ai)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Upload any dataset — CSV, Excel, JSON, or SQLite — and query it in plain English.**
> The AI agent auto-generates pandas or SQL code, executes it in a secure sandbox,
> and returns a written answer with an auto-generated chart. No coding required.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 **Multi-format ingestion** | CSV, Excel (multi-sheet), JSON (nested), SQLite DB |
| 🔍 **FAISS RAG** | Schema + sample rows embedded per-column — precise retrieval for any query |
| 🤖 **Dual-tool agent** | LangChain agent chooses between pandas and SQL depending on the question |
| 🛡 **Secure sandbox** | Whitelist-only `__builtins__`, DataFrame copy, stdout capture, auto-retry on failure |
| 🔄 **Multi-turn memory** | LangChain `ConversationBufferWindowMemory` — follow-up questions just work |
| 📊 **Auto charts** | Heuristic Plotly chart (bar, line, scatter, pie, histogram) auto-selected per result |
| 🔀 **Switchable LLM** | Toggle between Groq Llama-3.3-70b (free) and OpenAI GPT-4o mid-conversation |
| 📥 **Export chat** | Download full conversation as Markdown |

---

## 🏗 Architecture

```
User ──► Streamlit UI
              │
              ▼
    Data Ingestion Layer          CSV · Excel · JSON · SQLite
              │
              ▼
    Schema Extractor              per-column: dtype, stats, top values
              │
              ▼
    FAISS RAG Engine              one Document per column + metadata + sample rows
              │     ▲
              │     │  top-k retrieved context
              ▼     │
    LangChain Agent ◄─── ConversationBufferWindowMemory (k=5)
        │        │
        ▼        ▼
  PandasTool   SQLTool          both run in SafeExecutor sandbox
        │
        ▼
    LLM Selector                 Groq Llama-3.3-70b  |  OpenAI GPT-4o
        │
        ▼
    Output Display               Answer · Generated Code · Plotly Chart
```

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit 1.35+ | UI, file upload, chat interface |
| **Agent** | LangChain 0.3+ (LCEL) | Chain orchestration, memory |
| **LLM (free)** | Groq `llama-3.3-70b-versatile` | Code gen + answer gen |
| **LLM (paid)** | OpenAI `gpt-4o` | Higher-quality analysis |
| **Embeddings (free)** | HuggingFace `all-MiniLM-L6-v2` | Schema embedding (no API key) |
| **Embeddings (paid)** | OpenAI `text-embedding-3-small` | Higher-quality retrieval |
| **Vector store** | FAISS (in-memory) | Schema similarity search |
| **SQL engine** | SQLite (in-memory) | SQL queries on any DataFrame |
| **Charts** | Plotly Express | Auto-generated visualisations |
| **Data** | pandas + numpy | Data manipulation |

---

## 🚀 Quick Start (Local)

### 1. Clone and set up

```bash
git clone https://github.com/venkatraman0400-blip/askmydata.git
cd askmydata
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add API keys

Create `.streamlit/secrets.toml` (already in `.gitignore`):

```toml
# Get a free key at console.groq.com — no credit card needed
GROQ_API_KEY   = "gsk_..."

# Optional — only needed if you switch to OpenAI GPT-4o
OPENAI_API_KEY = "sk-..."
```

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), upload any CSV, and start asking questions.

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit — AskMyData v1.0"
   git remote add origin https://github.com/YOUR_USERNAME/askmydata.git
   git push -u origin main
   ```

2. **Create app on [share.streamlit.io](https://share.streamlit.io)**
   - Repository: `YOUR_USERNAME/askmydata`
   - Branch: `main`
   - Main file path: `app.py`

3. **Add secrets**
   In App Settings → Secrets, paste:
   ```toml
   GROQ_API_KEY = "gsk_..."
   OPENAI_API_KEY = "sk-..."   # optional
   ```

4. **Deploy** — first build takes ~3–5 min (downloads sentence-transformers model)

> **Note:** The HuggingFace embedding model (~80 MB) is downloaded on first app start
> and cached by Streamlit Cloud. Subsequent starts are instant.

---

## 💬 Example Queries

| Query | What the agent does |
|---|---|
| `What is the average revenue by region?` | `df.groupby('Region')['Revenue'].mean()` → bar chart |
| `Show me the top 10 customers by order value` | `run_sql("SELECT customer, SUM(order_value) ... LIMIT 10")` → table + bar |
| `How does sales trend over time?` | `df.groupby('Date')['Sales'].sum()` → line chart |
| `Which region had the highest growth last quarter?` | Follow-up — uses memory from previous answer |
| `Are there any outliers in the Price column?` | IQR calculation → filtered DataFrame |
| `How many rows have missing values?` | `df.isnull().sum()` → per-column counts |

---

## 📁 Project Structure

```
askmydata/
├── app.py                     ← Streamlit entry point (UI + state)
├── requirements.txt
├── .gitignore
│
├── .streamlit/
│   ├── config.toml            ← Theme + server settings
│   └── secrets.toml           ← API keys (never committed)
│
├── core/
│   ├── ingestion.py           ← CSV / Excel / JSON / SQLite loaders
│   ├── schema.py              ← Schema extractor (dtype, stats, top values)
│   ├── rag.py                 ← FAISS RAG engine (build + retrieve)
│   ├── agent.py               ← DataAnalysisAgent (code gen → exec → answer)
│   ├── executor.py            ← Safe sandbox (pandas + SQL)
│   ├── memory.py              ← ConversationBufferWindowMemory wrapper
│   └── llm.py                 ← LLM + embeddings factory (Groq / OpenAI)
│
└── tools/
    └── sql_tool.py            ← In-memory SQLite tool for SQL queries
```

---

## 🔑 Environment Variables

| Variable | Required | Where to get |
|---|---|---|
| `GROQ_API_KEY` | Yes (for free tier) | [console.groq.com](https://console.groq.com) — free, no credit card |
| `OPENAI_API_KEY` | Only for GPT-4o | [platform.openai.com](https://platform.openai.com) |

---

## 🤝 Built With

- [LangChain](https://python.langchain.com) — agent framework and memory
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [Streamlit](https://streamlit.io) — web UI
- [Plotly](https://plotly.com/python) — interactive charts
- [sentence-transformers](https://sbert.net) — free embeddings
- [Groq](https://groq.com) — free, fast LLM inference

---

*Built as part of a Data Science & AI/ML portfolio — B.E. Computer Science + Professional Certification in Data Science with AI (Boston Institute of Analytics, 2026)*
