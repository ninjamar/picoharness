from __future__ import annotations


class InputHistory:
    def __init__(self) -> None:
        self._entries: list[str] = []
        self._index: int = -1
        self._saved: str = ""

    def push(self, text: str) -> None:
        if text:
            self._entries.append(text)

    def up(self, current: str) -> str | None:
        if not self._entries:
            return None
        if self._index == -1:
            self._saved = current
            self._index = len(self._entries) - 1
        elif self._index > 0:
            self._index -= 1
        return self._entries[self._index]

    def down(self) -> str | None:
        if self._index == -1:
            return None
        if self._index < len(self._entries) - 1:
            self._index += 1
            return self._entries[self._index]
        else:
            self._index = -1
            return self._saved
