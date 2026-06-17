# core/agent.py
# ─────────────────────────────────────────────────────────
# DataAnalysisAgent — Phase 4 update.
#
# New in Phase 4 vs Phase 3:
#   • Agent prompt now offers BOTH pandas and SQL (run_sql)
#   • table_name passed to executor so SQL code resolves the table
#   • Chat history sourced from ChatMemory (windowed, LangChain-backed)
#   • Retry prompt also includes SQL instructions
# ─────────────────────────────────────────────────────────

import re as _re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.executor import ExecutionResult, SafeExecutor


# ── Prompts ────────────────────────────────────────────────────────────────────

_CODE_GEN_TEMPLATE = """\
You are an expert data analyst with full Python and SQL skills.

DATASET CONTEXT — schema, column statistics, sample rows:
{rag_context}

CONVERSATION HISTORY (remember this for follow-up questions):
{chat_history}

AVAILABLE TOOLS IN THE SANDBOX:
  df            — the dataset as a pandas DataFrame
  pd, np        — pandas and numpy
  math, re, statistics — standard library
  run_sql(q)    — execute SQL and return a DataFrame
                  e.g. run_sql("SELECT col, AVG(val) FROM {table_name} GROUP BY col")
  table_name    — string "{table_name}" (use in f-strings for SQL)

TASK: Write Python code to answer the user's question.

RULES:
1. Choose the clearest approach — pandas for complex transformations, SQL for aggregations
2. Store the FINAL answer in `result` (scalar, string, Series, or DataFrame)
3. Do NOT import files, read CSVs, or use matplotlib/seaborn/plotly
4. Output ONLY the executable Python code — no markdown fences, no comments

USER QUESTION: {question}

Python code:"""


_RETRY_TEMPLATE = """\
You are an expert data analyst.

DATASET CONTEXT:
{rag_context}

CONVERSATION HISTORY:
{chat_history}

AVAILABLE TOOLS: df, pd, np, run_sql(q), table_name="{table_name}", math, re, statistics

TASK: Fix the code that failed to answer: "{question}"

YOUR PREVIOUS CODE:
{failed_code}

ERROR:
{error_msg}

Write corrected code. Store the answer in `result`. Consider switching between
pandas and SQL if one approach keeps failing.

Corrected Python code:"""


_ANSWER_TEMPLATE = """\
A user asked: "{question}"

The data analysis returned this result:
{result_str}

Write a clear, direct answer in 1-3 sentences.
Rules:
- Include specific numbers or named values from the result
- Never say "Based on the data" or "According to the analysis"
- Never mention Python, pandas, DataFrames, or SQL
- Be concise and informative

Answer:"""


# ── Response container ─────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    answer:  str                 = ""
    code:    str                 = ""
    chart:   Optional[go.Figure] = None
    result:  Any                 = None
    error:   str                 = ""
    success: bool                = True


# ── Agent ─────────────────────────────────────────────────────────────────────

class DataAnalysisAgent:
    """
    Two-step LangChain chain: code generation → execution → answer + chart.

    Phase 4 additions:
      • SQL support via run_sql() in the executor namespace
      • table_name propagated through the full pipeline
      • chat_history injected from ChatMemory (windowed LangChain memory)
    """

    def __init__(self, llm) -> None:
        self._llm      = llm
        self._executor = SafeExecutor()

        self._code_chain = (
            ChatPromptTemplate.from_template(_CODE_GEN_TEMPLATE)
            | self._llm
            | StrOutputParser()
        )
        self._retry_chain = (
            ChatPromptTemplate.from_template(_RETRY_TEMPLATE)
            | self._llm
            | StrOutputParser()
        )
        self._answer_chain = (
            ChatPromptTemplate.from_template(_ANSWER_TEMPLATE)
            | self._llm
            | StrOutputParser()
        )

    # ── Public ─────────────────────────────────────────────────────────────────

    def run(
        self,
        query:        str,
        rag_context:  str,
        df:           pd.DataFrame,
        table_name:   str = "data",
        chat_history: str = "",
    ) -> AgentResponse:
        """
        Answer a natural-language question about a DataFrame.

        Args:
            query:        User's plain-English question
            rag_context:  Schema context from RAGEngine.retrieve()
            df:           The dataset
            table_name:   SQL table name (matches ingestion DataObject.table_name)
            chat_history: Windowed history string from ChatMemory.get_history_str()

        Returns:
            AgentResponse — answer, code, chart, raw result, and error if any.
        """
        history = chat_history or "None"

        # ── 1. Generate code ───────────────────────────────────────────────
        try:
            raw_code = self._code_chain.invoke({
                "question":     query,
                "rag_context":  rag_context,
                "chat_history": history,
                "table_name":   table_name,
            })
        except Exception as exc:
            return AgentResponse(
                answer  = f"I couldn't reach the AI model: {exc}",
                error   = str(exc),
                success = False,
            )

        code = self._clean_code(raw_code)

        # ── 2. Execute (with table_name for SQL support) ───────────────────
        exec_result: ExecutionResult = self._executor.execute(
            code, df, table_name=table_name
        )

        # ── 3. One retry if execution failed ──────────────────────────────
        if not exec_result.success:
            code, exec_result = self._retry(
                query, rag_context, history, df, table_name,
                code, exec_result.error,
            )

        # ── 4. Resolve result value ────────────────────────────────────────
        if exec_result.success:
            result_val = exec_result.result
            if result_val is None and exec_result.output:
                result_val = exec_result.output
            result_str = self._format_result(result_val)
        else:
            result_val = None
            result_str = f"Execution failed: {exec_result.error}"

        # ── 5. Natural-language answer ─────────────────────────────────────
        try:
            answer = self._answer_chain.invoke({
                "question":   query,
                "result_str": result_str,
            }).strip()
        except Exception:
            answer = f"Result: {result_str}"

        # ── 6. Auto chart ──────────────────────────────────────────────────
        chart = (
            self._auto_chart(result_val, query)
            if exec_result.success and result_val is not None
            else None
        )

        return AgentResponse(
            answer  = answer,
            code    = code,
            chart   = chart,
            result  = result_val,
            error   = exec_result.error,
            success = exec_result.success,
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _retry(
        self,
        query:        str,
        rag_context:  str,
        history:      str,
        df:           pd.DataFrame,
        table_name:   str,
        failed_code:  str,
        error_msg:    str,
    ) -> tuple[str, ExecutionResult]:
        try:
            raw = self._retry_chain.invoke({
                "question":     query,
                "rag_context":  rag_context,
                "chat_history": history,
                "table_name":   table_name,
                "failed_code":  failed_code,
                "error_msg":    error_msg,
            })
            fixed      = self._clean_code(raw)
            fix_result = self._executor.execute(fixed, df, table_name=table_name)
            return fixed, fix_result
        except Exception as exc:
            return failed_code, ExecutionResult(
                error=f"Retry failed: {exc}", success=False
            )

    @staticmethod
    def _clean_code(raw: str) -> str:
        """Strip markdown fences LLMs sometimes emit despite instructions."""
        code  = raw.strip()
        match = _re.match(r"^```[a-zA-Z]*\n?(.*?)```$", code, _re.DOTALL)
        if match:
            return match.group(1).strip()
        return code.removeprefix("```").removesuffix("```").strip()

    @staticmethod
    def _format_result(result: Any) -> str:
        if result is None:
            return "No result produced."
        if isinstance(result, pd.DataFrame):
            n       = len(result)
            preview = result.head(20).to_string(index=False, max_colwidth=50)
            if n > 20:
                preview += f"\n... {n} rows total, showing first 20"
            return preview
        if isinstance(result, pd.Series):
            return result.head(20).to_string()
        if isinstance(result, float):
            return f"{result:,.4f}".rstrip("0").rstrip(".")
        return str(result)

    @staticmethod
    def _auto_chart(result: Any, query: str) -> Optional[go.Figure]:
        """Heuristic Plotly chart generation from execution results."""
        _TIME_WORDS = {"date", "time", "month", "year", "week", "day", "quarter"}
        q = query.lower()

        if isinstance(result, pd.Series):
            idx_name = result.index.name or "Category"
            val_name = result.name or "Value"
            result   = result.reset_index()
            result.columns = [idx_name, val_name]

        if not isinstance(result, pd.DataFrame) or result.empty:
            return None

        num_cols = result.select_dtypes(include="number").columns.tolist()
        cat_cols = result.select_dtypes(exclude="number").columns.tolist()

        if not num_cols:
            return None

        is_time = any(w in q for w in _TIME_WORDS)
        if cat_cols:
            is_time = is_time or any(w in cat_cols[0].lower() for w in _TIME_WORDS)

        try:
            if cat_cols and num_cols:
                x, y = cat_cols[0], num_cols[0]
                if is_time:
                    fig = px.line(result, x=x, y=y, markers=True)
                elif len(result) == 2:
                    fig = px.pie(result, names=x, values=y)
                else:
                    fig = px.bar(result, x=x, y=y)
            elif len(num_cols) >= 2:
                fig = px.scatter(result, x=num_cols[0], y=num_cols[1])
            else:
                fig = px.histogram(result, x=num_cols[0])
        except Exception:
            return None

        # Use the user's question as the chart title (trimmed)
        title = query.rstrip("?").strip()
        title = title[0].upper() + title[1:] if title else ""

        fig.update_layout(
            title         = dict(text=title[:72], font=dict(size=13), x=0.02),
            margin        = {"l": 20, "r": 20, "t": 48, "b": 20},
            plot_bgcolor  = "rgba(0,0,0,0)",
            paper_bgcolor = "rgba(0,0,0,0)",
        )
        return fig
