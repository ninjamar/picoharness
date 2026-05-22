from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from pyratatui import (
    Block,
    Clear,
    Color,
    List,
    ListItem,
    ListState,
    Rect,
    Style,
    TextArea,
)


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


@dataclass
class CommandPanelState:
    visible: bool = False
    mode: CommandMode = CommandMode.CHOICE
    choices: list[str] = field(default_factory=list)
    selected: set[str] = field(default_factory=set)
    list_state: ListState = field(default_factory=ListState)
    textarea: TextArea = field(default_factory=TextArea)
    callback: Callable[[Any], None] | None = None


def centered_rect(area: Rect, width: int, height: int) -> Rect:
    x = max(0, (area.width - width) // 2)
    y = max(0, (area.height - height) // 2)
    return Rect(area.x + x, area.y + y, width, height)


def draw_completion_menu(frame, area: Rect, state: CompletionState) -> None:
    if not state.visible or not state.items:
        return
    items = [ListItem(f"{cmd}  {desc}") for cmd, desc in state.items]
    list_widget = (
        List(items).block(Block().bordered()).highlight_style(Style().fg(Color.yellow()).bold()).highlight_symbol("▶ ")
    )
    frame.render_stateful_list(list_widget, area, state.list_state)


def draw_command_panel(frame, area: Rect, state: CommandPanelState) -> None:
    if not state.visible:
        return

    if state.mode == CommandMode.TEXT:
        frame.render_widget(Clear(), area)
        dialog_rect = centered_rect(area, min(60, area.width - 2), 5)
        frame.render_widget(Block().bordered().title("Enter value").style(Style().fg(Color.yellow())), dialog_rect)
        content_rect = Rect(dialog_rect.x + 1, dialog_rect.y + 1, dialog_rect.width - 2, 1)
        frame.render_textarea(state.textarea, content_rect)
    else:
        frame.render_widget(Clear(), area)
        dialog_rect = centered_rect(area, min(40, area.width - 2), min(10, area.height - 2))
        if state.mode == CommandMode.MULTISELECT:
            items = [ListItem(("✓ " if c in state.selected else "  ") + c) for c in state.choices]
        else:
            items = [ListItem(c) for c in state.choices]
        list_widget = (
            List(items)
            .block(
                Block()
                .bordered()
                .title("Select" if state.mode == CommandMode.CHOICE else "Select multiple")
                .style(Style().fg(Color.yellow()))
            )
            .highlight_style(Style().bg(Color.yellow()).fg(Color.black()))
            .highlight_symbol("▶ ")
        )
        frame.render_stateful_list(list_widget, dialog_rect, state.list_state)
