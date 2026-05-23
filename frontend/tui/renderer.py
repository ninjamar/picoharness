from __future__ import annotations

from typing import TYPE_CHECKING

from pyratatui import (
    Block,
    Constraint,
    Direction,
    Layout,
    Paragraph,
)

from frontend.tui.dialog import draw_command_panel, draw_completion_menu
from frontend.tui.state import CommandPanelState, CompletionState

if TYPE_CHECKING:
    from frontend.chat_app import ChatApp


def overlay_height(completion: CompletionState, command: CommandPanelState) -> int:
    if completion.visible and completion.items:
        return min(10, len(completion.items) + 2)
    elif command.visible:
        return min(12, len(command.choices) + 4) if command.choices else 7
    return 0


def render(frame, app: ChatApp) -> None:
    h = overlay_height(app._completion, app._command)
    chunks = (
        Layout()
        .direction(Direction.Vertical)
        .constraints([Constraint.fill(1), Constraint.length(h), Constraint.length(3)])
        .split(frame.area)
    )
    chat_area, overlay_area, input_area = chunks

    if not app._chat_log.lines:
        frame.render_widget(
            Paragraph.from_string("LocalAI  Ctrl+C to quit · Ctrl+J for newline · /help for commands").block(
                Block().bordered()
            ),
            chat_area,
        )
    else:
        visible_height = int(chat_area.height)
        total_lines = len(app._chat_log.lines)
        if total_lines <= visible_height:
            visible_lines = app._chat_log.lines
        else:
            start = max(0, total_lines - visible_height)
            visible_lines = app._chat_log.lines[start : start + visible_height]
        text = "\n".join(visible_lines)
        frame.render_widget(Paragraph.from_string(text).block(Block().bordered()), chat_area)

    if app._completion.visible:
        draw_completion_menu(frame, overlay_area, app._completion)
    elif app._command.visible:
        draw_command_panel(frame, overlay_area, app._command)

    frame.render_textarea(app._textarea, input_area)
