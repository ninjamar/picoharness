from __future__ import annotations

import asyncio
import contextlib
import json
import re
from dataclasses import dataclass, field

import blessed
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI

from ..core.backend import ChatBackend
from ..events import (Event, ResponseEvent, ThinkingEvent, ToolEndEvent,
                      ToolStartEvent, UserInputEvent)

_ISO2022_RE = re.compile(r"\x1b[()][0-9A-Za-z]")


@dataclass
class ConversationTurn:
    user_input: str
    events: list[Event] = field(default_factory=list)


def _print_event(term: blessed.Terminal, event: Event, show_thinking: bool, last_event: Event | None) -> None:
    """Print one streaming event fragment with blessed formatting."""
    match event:
        case UserInputEvent():
            pass
        case ThinkingEvent(text=fragment):
            if show_thinking:
                print(term.italic + term.dim + fragment + term.normal, end="", flush=True)
        case ResponseEvent(text=fragment):
            if isinstance(last_event, ThinkingEvent):
                # Add newline when transitioning from thinking to response

                print()

            print(fragment, end="", flush=True)

        case ToolStartEvent(id=tool_id, name=name, input=inp):
            inp_str = json.dumps(inp) if isinstance(inp, dict) else str(inp)
            print(f"\n{term.bold_blue(f'[tool] {name}({inp_str})')} ...", flush=True)
        case ToolEndEvent(id=tool_id, output=output):
            print(term.bold_blue(f"  → {output}"), flush=True)


class ChatApp:
    def __init__(self, backend: ChatBackend) -> None:
        self._backend = backend
        self._term = blessed.Terminal()
        self._prompt = PromptSession()

        self._show_thinking = True

    def run(self) -> None:
        """Run the chat loop using character-at-a-time input with blessed."""
        print(self._term.bold("Chat started. Press Ctrl+C to quit.\n"))
        loop = asyncio.new_event_loop()
        try:
            while True:
                try:
                    user_text = loop.run_until_complete(self._read_line())
                except KeyboardInterrupt, EOFError:
                    print("\nBye.")
                    break

                if not user_text:
                    continue

                turn = ConversationTurn(user_input=user_text)
                task = loop.create_task(self._stream_turn(turn))
                try:
                    loop.run_until_complete(task)
                except KeyboardInterrupt:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        loop.run_until_complete(task)
                    print()
                print()
        finally:
            loop.close()

    async def _read_line(self) -> str:
        """Read a line from the terminal"""

        return await self._prompt.prompt_async(
            # The prompt needs to be included in this function so it can be synced with the input.
            message=ANSI(
                # However, the ANSI class (which handles the escape codes from blessed) doesn't
                # support some of blessed's generated output --a zero width character or something.
                # So, a regex is used to remove it.
                _ISO2022_RE.sub("", self._term.bold_green(">>> "))
            )
        )

    async def _stream_turn(self, turn: ConversationTurn) -> None:
        """Stream the backend response for a turn."""
        last_event: Event | None = None
        async for event in self._backend.stream(turn.user_input):
            turn.events.append(event)
            _print_event(self._term, event, self._show_thinking, last_event)
            last_event = event
