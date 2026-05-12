"""Sliding-window conversation memory for multi-turn chat."""

from typing import List
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str   # "user" or "assistant"
    content: str


class ConversationMemory:
    def __init__(self, window: int = 6):
        # Each "turn" = one user message + one assistant message (2 entries)
        self._turns_window = window
        self._history: List[Message] = []

    def add_user(self, content: str) -> None:
        self._history.append(Message(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self._history.append(Message(role="assistant", content=content))

    def get_history(self) -> List[Message]:
        return self._history[-(self._turns_window * 2):]

    def to_langchain_messages(self) -> List[dict]:
        return [
            {"role": m.role, "content": m.content}
            for m in self.get_history()
        ]

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)
