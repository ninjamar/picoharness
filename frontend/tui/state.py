from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from pyratatui import ListState, TextArea


class CommandMode(Enum):
    CHOICE = "choice"
    MULTISELECT = "multiselect"
    TEXT = "text"


@dataclass
class CompletionState:
    items: list[tuple[str, str]] = field(default_factory=list)
    index: int = 0
    visible: bool = False
    list_state: ListState = field(default_factory=ListState)

    def update(self, text: str, commands: list[tuple[str, str]]) -> None:
        if text.startswith("/"):
            after_slash = text[1:]
            if " " in after_slash:
                self.visible = False
                return
            query = after_slash.lower() if after_slash.strip() else ""
            matches = [(f"/{cmd}", desc) for cmd, desc in commands if cmd.startswith(query)]
            if matches:
                self.items = matches
                self.index = 0
                self.list_state.select(0)
                self.visible = True
            else:
                self.visible = False
        else:
            self.visible = False

    def handle_key(self, key_code: str | None) -> str | None:
        if not key_code:
            return None
        if key_code == "Up":
            if self.index > 0:
                self.index -= 1
                self.list_state.select(self.index)
        elif key_code == "Down":
            if self.index < len(self.items) - 1:
                self.index += 1
                self.list_state.select(self.index)
        elif key_code == "Tab" or key_code == "Enter":
            if self.items:
                return self.items[self.index][0]
        elif key_code == "Esc":
            self.visible = False
        return None


@dataclass
class CommandPanelState:
    visible: bool = False
    mode: CommandMode = CommandMode.CHOICE
    choices: list[str] = field(default_factory=list)
    selected: set[str] = field(default_factory=set)
    list_state: ListState = field(default_factory=ListState)
    textarea: TextArea = field(default_factory=TextArea)
    callback: Callable[[Any], None] | None = None

    def handle_key(self, key_code: str | None) -> None:
        if not key_code:
            return

        if key_code == "Up":
            self.list_state.select_previous()
        elif key_code == "Down":
            self.list_state.select_next()
        elif key_code == "Enter":
            selected_idx = self.list_state.selected
            if self.mode == CommandMode.CHOICE:
                if selected_idx is not None and selected_idx < len(self.choices):
                    if self.callback:
                        self.callback(self.choices[selected_idx])
                self.visible = False
            elif self.mode == CommandMode.MULTISELECT:
                selected_choices = [self.choices[i] for i in range(len(self.choices)) if i in self.selected]
                if self.callback:
                    self.callback(selected_choices)
                self.visible = False
            elif self.mode == CommandMode.TEXT:
                if self.callback:
                    self.callback("\n".join(self.textarea.lines()))
                self.visible = False
        elif key_code == "Esc":
            self.visible = False
        elif key_code == " " and self.mode == CommandMode.MULTISELECT:
            selected_idx = self.list_state.selected
            if selected_idx is not None:
                choice = self.choices[selected_idx]
                if choice in self.selected:
                    self.selected.discard(choice)
                else:
                    self.selected.add(choice)
        elif self.mode == CommandMode.TEXT:
            self.textarea.input_key(key_code, False, False, False)
