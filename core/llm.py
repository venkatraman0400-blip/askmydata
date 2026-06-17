# core/llm.py
# ─────────────────────────────────────────────────────────
# LLM + Embeddings factory for AskMyData.
#
# Provider: "Groq (free)"   → Llama-3.3-70b + HuggingFace all-MiniLM-L6-v2
# Provider: "OpenAI GPT-4o" → GPT-4o + text-embedding-3-small
#
# @st.cache_resource on the HF loader means the ~80 MB model is downloaded
# once per server lifetime and reused across all user sessions.
# ─────────────────────────────────────────────────────────

import streamlit as st

# ── Model identifiers ──────────────────────────────────────────────────────────
HF_EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
OAI_EMBED_MODEL = "text-embedding-3-small"
GROQ_LLM_MODEL  = "llama-3.3-70b-versatile"
OAI_LLM_MODEL   = "gpt-4o"


# ── HuggingFace model — cached globally ───────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_hf_embeddings():
    """
    Download and cache the HuggingFace embedding model.
    Runs once on first call; subsequent calls return the cached object instantly.
    Model: all-MiniLM-L6-v2 (~80 MB, CPU-only, no API key needed)
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=HF_EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Public: embeddings factory ─────────────────────────────────────────────────

def get_embeddings(provider: str):
    """
    Return a LangChain Embeddings object for the selected provider.

    Args:
        provider: "Groq (free)" or "OpenAI GPT-4o"

    Returns:
        HuggingFaceEmbeddings (free) or OpenAIEmbeddings (paid)

    Raises:
        ValueError if the required API key is missing or placeholder.
    """
    if provider == "OpenAI GPT-4o":
        from langchain_openai import OpenAIEmbeddings

        key = _get_key("OPENAI_API_KEY", prefix="sk-")
        return OpenAIEmbeddings(model=OAI_EMBED_MODEL, api_key=key)
    else:
        return _load_hf_embeddings()


# ── Public: LLM factory (used from Phase 3 onward) ────────────────────────────

def get_llm(provider: str):
    """
    Return a LangChain LLM object for the selected provider.
    Both are configured with temperature=0 for deterministic code generation.

    Args:
        provider: "Groq (free)" or "OpenAI GPT-4o"

    Raises:
        ValueError if the required API key is missing or placeholder.
    """
    if provider == "OpenAI GPT-4o":
        from langchain_openai import ChatOpenAI

        key = _get_key("OPENAI_API_KEY", prefix="sk-")
        return ChatOpenAI(model=OAI_LLM_MODEL, api_key=key, temperature=0)
    else:
        from langchain_groq import ChatGroq

        key = _get_key("GROQ_API_KEY", prefix="gsk_")
        return ChatGroq(model=GROQ_LLM_MODEL, groq_api_key=key, temperature=0)


# ── Internal helper ────────────────────────────────────────────────────────────

def _get_key(secret_name: str, prefix: str) -> str:
    """
    Retrieve an API key from Streamlit secrets with validation.
    Raises a clear ValueError if the key is absent or still a placeholder.
    """
    key = st.secrets.get(secret_name, "")
    if not key or "your" in key.lower() or not key.startswith(prefix):
        raise ValueError(
            f"'{secret_name}' is missing or still a placeholder.\n"
            f"Add it to .streamlit/secrets.toml:\n"
            f"  {secret_name} = \"{prefix}...\""
        )
    return key
