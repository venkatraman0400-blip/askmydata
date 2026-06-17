# core/memory.py
# ─────────────────────────────────────────────────────────
# Multi-turn conversation memory for AskMyData.
#
# Wraps LangChain ConversationBufferWindowMemory with:
#   • Streamlit-friendly interface (storable in session_state)
#   • Windowed history — remembers last k exchanges only
#   • Automatic context trimming — older messages are dropped
#     so the LLM prompt never overflows
#   • Live resize — window size can change without losing
#     recent exchanges
#
# One ChatMemory lives in st.session_state per loaded dataset.
# It is reset when the user loads a new dataset.
# ─────────────────────────────────────────────────────────

from langchain.memory import ConversationBufferWindowMemory


class ChatMemory:
    """
    Windowed conversation memory for the AskMyData agent.

    Remembers the last `k` question-answer exchanges. When a new
    exchange is added and the window is full, the oldest exchange
    is silently dropped — keeping the prompt size bounded.

    Usage:
        mem = ChatMemory(k=5)
        mem.add_exchange("What is avg revenue?", "The average is $4,253.")
        history_str = mem.get_history_str()   # injected into agent prompt
    """

    def __init__(self, k: int = 5) -> None:
        self.k    = k
        self._mem = self._make_memory(k)

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_exchange(self, human: str, ai: str) -> None:
        """
        Record one question + answer exchange.
        If the window is full, the oldest exchange is automatically dropped.
        """
        self._mem.save_context(
            inputs={"input":  human},
            outputs={"output": ai},
        )

    def get_history_str(self) -> str:
        """
        Return the windowed conversation history as a formatted string.

        Format (injected verbatim into the agent prompt):
            User: What is the total revenue?
            Assistant: The total revenue across all orders is $2.45 million.

            User: Which region leads?
            Assistant: The North region leads with $890,000.

        Returns "None" when there is no history yet.
        """
        raw = self._mem.load_memory_variables({}).get("history", "").strip()
        return raw if raw else "None"

    def clear(self) -> None:
        """Wipe all stored exchanges."""
        self._mem.clear()

    def resize(self, new_k: int) -> None:
        """
        Change the memory window size.
        Recent exchanges (up to new_k) are preserved; older ones are dropped.
        No-op if new_k equals the current window size.
        """
        if new_k == self.k:
            return

        old_messages = list(self._mem.chat_memory.messages)
        self.k       = new_k
        self._mem    = self._make_memory(new_k)

        # Replay the last new_k exchanges from the old history
        pairs_to_keep = old_messages[-(new_k * 2):]
        for i in range(0, len(pairs_to_keep) - 1, 2):
            try:
                self._mem.save_context(
                    inputs  = {"input":  pairs_to_keep[i].content},
                    outputs = {"output": pairs_to_keep[i + 1].content},
                )
            except (IndexError, AttributeError):
                break

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def exchange_count(self) -> int:
        """Number of complete exchanges currently stored in the window."""
        return len(self._mem.chat_memory.messages) // 2

    @property
    def window_size(self) -> int:
        """Maximum exchanges the window can hold (= k)."""
        return self.k

    @property
    def is_empty(self) -> bool:
        return self.exchange_count == 0

    # ── Internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_memory(k: int) -> ConversationBufferWindowMemory:
        return ConversationBufferWindowMemory(
            k                = k,
            human_prefix     = "User",
            ai_prefix        = "Assistant",
            return_messages  = False,   # return plain string, not message objects
        )
