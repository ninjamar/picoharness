import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal

import ollama
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

MODEL = "lfm2.5-thinking:latest"

STYLE = Style.from_dict(
    {
        "prompt": "ansibrightgreen bold",
        "thinking": "ansidarkgray italic",
        "response": "ansiwhite",
    }
)


@dataclass
class ChatEvent:
    pass


@dataclass
class ThinkingChunk(ChatEvent):
    """A fragment of the model's internal reasoning."""

    text: str


@dataclass
class ResponseChunk(ChatEvent):
    """A fragment of the model's visible reply."""

    text: str


@dataclass
class ToolCallRequest(ChatEvent):
    """Model asked to invoke a tool. Stub for future tool support."""

    pass


class ChatBackend:
    """Manages ollama interactions and conversation history."""

    def __init__(self, model: str = MODEL) -> None:
        self.model = model
        self.client = ollama.AsyncClient()
        self.messages: list[dict[str, str]] = []

    async def stream(self, user_input: str) -> AsyncGenerator[ChatEvent, None]:
        """Stream response events from the model.

        Yields ChatEvent objects (ThinkingChunk, ResponseChunk, ToolCallRequest).
        Appends user message to history before streaming, appends assistant
        response after streaming completes.
        """
        self.messages.append({"role": "user", "content": user_input})

        think = ""
        response = ""

        async for part in await self.client.chat(model=self.model, messages=self.messages, stream=True, think=True):
            if data := part.message.thinking:
                think += data
                yield ThinkingChunk(data)

            if data := part.message.content:
                response += data
                yield ResponseChunk(data)

        self.messages.append({"role": "assistant", "content": response})


class TerminalUI:
    """Handles terminal I/O: prompts, styled output, and formatted text rendering."""

    def __init__(self, style: Style = STYLE) -> None:
        self.session = PromptSession()
        self.style = style

    async def get_input(self) -> str:
        """Get user input from the prompt, looping on empty input."""
        while True:
            with patch_stdout():
                user_input = await self.session.prompt_async(
                    FormattedText([("class:prompt", "> ")]),
                    style=self.style,
                )

            if s := user_input.strip():
                return s

    async def render_stream(self, events: AsyncGenerator[ChatEvent, None]) -> None:
        """Consume and render a stream of chat events.

        Handles mode transitions between thinking and response output,
        emitting newlines at boundaries to separate different output types.
        """
        prev_mode: str | None = None

        async for event in events:
            match event:
                case ThinkingChunk():
                    if prev_mode is not None and prev_mode != "think":
                        print()
                    self.print_chunk(event)
                    prev_mode = "think"

                case ResponseChunk():
                    if prev_mode is not None and prev_mode != "text":
                        print()
                    self.print_chunk(event)
                    prev_mode = "text"

        print()  # Trailing newline after stream completes

    def print_info(self, text: str) -> None:
        """Print informational text."""
        print(text)

    def print_chunk(self, chunk: ChatEvent):
        fmt = None
        if isinstance(chunk, ThinkingChunk):
            fmt = "thinking"
        elif isinstance(chunk, ResponseChunk):
            fmt = "response"

        print_formatted_text(
            FormattedText([(f"class:{fmt}", chunk.text)]),
            end="",
            flush=True,
            style=self.style,
        )


async def _app():
    backend = ChatBackend(MODEL)
    ui = TerminalUI()

    ui.print_info(
            f"Running model {MODEL}. Ensure the context window has been turned up for optimal usage"
    )
    
    while True:
        try:
            user_input = await ui.get_input()
        except KeyboardInterrupt:
            break
        try:
            events = backend.stream(user_input)
            await ui.render_stream(events)
        except Exception as e:
            print(e)
            continue

def app():
    asyncio.run(_app())
    
if __name__ == "__main__":
    app()
