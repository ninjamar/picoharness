from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import blessed
from prompt_toolkit import PromptSession

from ..core.backend import ChatBackend
from ..events import (Event, ResponseEvent, ThinkingEvent, ToolEndEvent,
                      ToolStartEvent, UserInputEvent)


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
        print(self._term.bold("Chat started. Press Ctrl+V to toggle thinking, Ctrl+C to quit.\n"))
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        """Async chat loop."""
        while True:
            try:
                user_text = await self._read_line()
            except KeyboardInterrupt, EOFError:
                print("\nBye.")
                break

            if not user_text:
                continue

            turn = ConversationTurn(user_input=user_text)
            # print(self._term.bold_green(f">>> {user_text}"))
            await self._stream_turn(turn)
            print()

    async def _read_line(self) -> str:
        """Read a line from the terminal, detecting Ctrl+V for thinking toggle."""
        with self._term.cbreak():
            prompt = self._term.bold_green(">>> ")
            print(prompt, end="", flush=True)

            return await self._prompt.prompt_async()

    async def _stream_turn(self, turn: ConversationTurn) -> None:
        """Stream the backend response for a turn."""
        last_event: Event | None = None
        async for event in self._backend.stream(turn.user_input):
            turn.events.append(event)
            _print_event(self._term, event, self._show_thinking, last_event)
            last_event = event
