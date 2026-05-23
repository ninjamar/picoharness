from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pyratatui import TextArea

if TYPE_CHECKING:
    from frontend.chat_app import ChatApp


class Command(ABC):
    @abstractmethod
    async def execute(self, app: ChatApp) -> None:
        pass


class ExitCommand(Command):
    async def execute(self, app: ChatApp) -> None:
        app._should_exit = True


class CancelCurrentCommand(Command):
    async def execute(self, app: ChatApp) -> None:
        app.api.cancel_current()


class CompletionSelectCommand(Command):
    def __init__(self, text: str) -> None:
        self.text = text

    async def execute(self, app: ChatApp) -> None:
        app._textarea = TextArea()
        app._textarea.insert_str(self.text + " ")
        app._completion.visible = False


class CommandPanelKeyCommand(Command):
    def __init__(self, key_code: str) -> None:
        self.key_code = key_code

    async def execute(self, app: ChatApp) -> None:
        app._command.handle_key(self.key_code)


class SubmitCommand(Command):
    def __init__(self, text: str) -> None:
        self.text = text

    async def execute(self, app: ChatApp) -> None:
        app._history.push(self.text)
        app.api.feed(str(uuid.uuid4()), self.text)
        app._turn_in_progress = True
        app._input_locked = True
        app._textarea = TextArea()


class HistoryUpCommand(Command):
    def __init__(self, current_text: str) -> None:
        self.current_text = current_text

    async def execute(self, app: ChatApp) -> None:
        text = app._history.up(self.current_text)
        if text is not None:
            app._textarea = TextArea.from_lines(text.split("\n"))


class HistoryDownCommand(Command):
    async def execute(self, app: ChatApp) -> None:
        text = app._history.down()
        if text is not None:
            if text:
                app._textarea = TextArea.from_lines(text.split("\n"))
            else:
                app._textarea = TextArea()


class NewlineCommand(Command):
    async def execute(self, app: ChatApp) -> None:
        app._textarea.insert_str("\n")


class InsertKeyCommand(Command):
    def __init__(self, key_code: str, ctrl: bool = False, alt: bool = False, shift: bool = False) -> None:
        self.key_code = key_code
        self.ctrl = ctrl
        self.alt = alt
        self.shift = shift

    async def execute(self, app: ChatApp) -> None:
        app._textarea.input_key(self.key_code, self.ctrl, self.alt, self.shift)


def handle_key(ev, app: ChatApp) -> Command | None:
    if ev is None:
        return None

    key_code = getattr(ev, "code", None)
    ctrl = getattr(ev, "ctrl", False)
    alt = getattr(ev, "alt", False)
    shift = getattr(ev, "shift", False)

    if key_code == "c" and ctrl:
        if app._turn_in_progress:
            return CancelCurrentCommand()
        else:
            return ExitCommand()

    if app._input_locked:
        return None

    if app._completion.visible:
        selected = app._completion.handle_key(key_code)
        if selected:
            return CompletionSelectCommand(selected)
        return None

    if app._command.visible:
        if key_code is not None:
            return CommandPanelKeyCommand(key_code)
        return None

    if key_code == "Enter":
        text = "\n".join(app._textarea.lines()).strip()
        if text:
            return SubmitCommand(text)
    elif key_code == "Up":
        return HistoryUpCommand("\n".join(app._textarea.lines()))
    elif key_code == "Down":
        return HistoryDownCommand()
    elif key_code == "j" and ctrl:
        return NewlineCommand()
    elif isinstance(key_code, str):
        return InsertKeyCommand(key_code, ctrl, alt, shift)

    return None
