from __future__ import annotations

import asyncio
from pathlib import Path

from pyratatui import AsyncTerminal, TextArea

from backend.api import BackendAPI
from frontend.chat_log import ChatLog
from frontend.commands import dispatch_command
from frontend.input_history import InputHistory
from frontend.schema import FIELDS
from frontend.tui import CompletionState, CommandPanelState, handle_key, render

SYSTEM_PROMPT_PATH = Path(__file__).parent / "files" / "system_prompt.md"

_ALL_COMMANDS: list[tuple[str, str]] = [(f.name, f.description) for f in FIELDS] + [
    ("help", "Show available commands"),
    ("quit", "Exit the application"),
]


class ChatApp:
    def __init__(self, api: BackendAPI, show_think: bool = True) -> None:
        self.api = api
        self._show_thinking = show_think
        self._turn_in_progress = False

        self._chat_log = ChatLog()
        self._textarea = TextArea()
        self._input_locked = False
        self._history = InputHistory()

        self._completion = CompletionState()
        self._command = CommandPanelState()
        self._should_exit = False

    @property
    def show_think(self) -> bool:
        return self._show_thinking

    def set_show_think(self, value: bool) -> None:
        self._show_thinking = value

    async def _consume_events(self) -> None:
        async for event in self.api.stream_events():
            if self._chat_log.process(event, self._show_thinking):
                self._turn_in_progress = False
                self._input_locked = False

    async def _handle_key(self, ev) -> None:
        cmd = handle_key(ev, self)
        if cmd:
            await cmd.execute(self)

        text = "\n".join(self._textarea.lines())
        self._completion.update(text, _ALL_COMMANDS)

    async def _run(self) -> None:
        async with self.api:
            async with AsyncTerminal() as term:
                term.hide_cursor()
                stream_task = asyncio.create_task(self._consume_events())

                try:
                    async for ev in term.events(fps=30):
                        if ev is not None and getattr(ev, "code", None) == "Enter":
                            text = "\n".join(self._textarea.lines()).strip()
                            if text.startswith("/") and not self._command.visible and not self._completion.visible:
                                await dispatch_command(text, self)
                                self._textarea = TextArea()
                                self._completion.update("", _ALL_COMMANDS)
                                term.draw(lambda frame: render(frame, self))
                                continue

                        await self._handle_key(ev)
                        term.draw(lambda frame: render(frame, self))

                        if self._should_exit:
                            break

                finally:
                    term.show_cursor()
                    stream_task.cancel()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass
