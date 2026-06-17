# core/memory.py
# ─────────────────────────────────────────────────────────
# Multi-turn conversation memory for AskMyData.
# Pure Python — no langchain.memory dependency.
# Stores the last k question-answer exchanges.
# ─────────────────────────────────────────────────────────


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
        self.k          = k
        self._exchanges: list[tuple[str, str]] = []  # (human, ai) pairs

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_exchange(self, human: str, ai: str) -> None:
        """Record one Q&A exchange. Drops oldest if window is full."""
        self._exchanges.append((human, ai))
        if len(self._exchanges) > self.k:
            self._exchanges = self._exchanges[-self.k:]

    def get_history_str(self) -> str:
        """
        Return the windowed history as a formatted string for prompt injection.
        Returns "None" when there is no history yet.
        """
        if not self._exchanges:
            return "None"
        lines = []
        for human, ai in self._exchanges:
            lines.append(f"User: {human}")
            lines.append(f"Assistant: {ai}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Wipe all stored exchanges."""
        self._exchanges = []

    def resize(self, new_k: int) -> None:
        """Change window size, keeping the most recent exchanges."""
        self.k = new_k
        if len(self._exchanges) > new_k:
            self._exchanges = self._exchanges[-new_k:]

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def exchange_count(self) -> int:
        return len(self._exchanges)

    @property
    def window_size(self) -> int:
        return self.k

    @property
    def is_empty(self) -> bool:
        return len(self._exchanges) == 0
