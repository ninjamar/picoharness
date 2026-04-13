import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

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
class ThinkingEvent(ChatEvent):
    """A fragment of the model's internal reasoning."""

    text: str


@dataclass
class ResponseEvent(ChatEvent):
    """A fragment of the model's visible reply."""

    text: str


@dataclass
class ToolCallRequest(ChatEvent):
    """Model asked to invoke a tool."""

    tool_name: str
    arguments: dict


class BaseTool:
    """Base class for tools that can be called by the model."""

    name: str = ""
    description: str = ""
    parameters: dict = {}

    _registry: dict[str, type["BaseTool"]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name:
            BaseTool._registry[cls.name] = cls

    @classmethod
    def to_ollama(cls) -> dict:
        """Return the Ollama-compatible tool definition."""
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": cls.parameters,
            },
        }

    async def execute(self, **kwargs) -> str:
        """Execute the tool and return a result string."""
        raise NotImplementedError


class ReadFileTool(BaseTool):
    """Tool to read the contents of a file."""

    name = "read_file"
    description = "Read the contents of a file on disk and return them as a string."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file to read.",
            }
        },
        "required": ["path"],
    }

    async def execute(self, path: str, **kwargs) -> str:
        try:
            return await asyncio.to_thread(Path(path).read_text)
        except OSError as e:
            return f"Error reading file: {e}"


class ChatBackend:
    """Manages ollama interactions and conversation history."""

    def __init__(self, model: str = MODEL, system_prompt=None, tools: list = None) -> None:
        self.model = model
        self.client = ollama.AsyncClient()
        self.messages: list[dict[str, str]] = [] if system_prompt is None else [system_prompt]

        self.tools = [] if tools is None else tools

    async def stream(self, user_input: str | None = None) -> AsyncGenerator[ChatEvent, None]:
        """Stream response events from the model.

        Yields ChatEvent objects (ThinkingEvent, ResponseEvent, ToolCallRequest).
        Appends user message to history before streaming (if user_input is not None),
        appends assistant response after streaming completes.
        """
        if user_input is not None:
            self.messages.append({"role": "user", "content": user_input})

        think = ""
        response = ""
        pending_tool_calls: list[dict] = []

        async for part in await self.client.chat(
            model=self.model,
            messages=self.messages,
            stream=True,
            think=True,
            tools=[tool.to_ollama() for tool in self.tools] or None,
        ):
            if data := part.message.thinking:
                think += data
                yield ThinkingEvent(data)

            if data := part.message.content:
                response += data
                yield ResponseEvent(data)

            if part.message.tool_calls:
                for tc in part.message.tool_calls:
                    pending_tool_calls.append(
                        {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    )
                    yield ToolCallRequest(
                        tool_name=tc.function.name,
                        arguments=tc.function.arguments,
                    )

        if pending_tool_calls:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "tool_calls": pending_tool_calls,
                }
            )
        else:
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
                case ThinkingEvent():
                    if prev_mode is not None and prev_mode != "thinking":
                        print()
                    self.print_event(event)
                    prev_mode = "thinking"

                case ResponseEvent():
                    if prev_mode is not None and prev_mode != "response":
                        print()
                    self.print_event(event)
                    prev_mode = "response"

        print()  # Trailing newline after stream completes

    def print_info(self, text: str) -> None:
        """Print informational text."""
        print(text)

    def print_event(self, event: ChatEvent):
        fmt = None
        if isinstance(event, ThinkingEvent):
            fmt = "thinking"
        elif isinstance(event, ResponseEvent):
            fmt = "response"

        print_formatted_text(
            FormattedText([(f"class:{fmt}", event.text)]),
            end="",
            flush=True,
            style=self.style,
        )


async def _app(tools: list[type[BaseTool]] | None = None):
    backend = ChatBackend(MODEL, tools=tools or [])
    ui = TerminalUI()

    ui.print_info(f"Running model {MODEL}. Ensure the context window has been turned up for optimal usage")

    while True:
        try:
            user_input = await ui.get_input()
        except KeyboardInterrupt:
            break

        try:
            next_input: str | None = user_input
            while True:
                tool_requests: list[ToolCallRequest] = []

                async def tee(gen: AsyncGenerator[ChatEvent, None]) -> AsyncGenerator[ChatEvent, None]:
                    async for event in gen:
                        if isinstance(event, ToolCallRequest):
                            tool_requests.append(event)
                        else:
                            yield event

                await ui.render_stream(tee(backend.stream(next_input)))
                next_input = None  # subsequent passes: no new user message

                if not tool_requests:
                    break

                for tool_request in tool_requests:
                    tool_cls = BaseTool._registry.get(tool_request.tool_name)
                    if tool_cls:
                        result = await tool_cls().execute(**tool_request.arguments)
                    else:
                        result = f"Unknown tool: {tool_request.tool_name}"
                    backend.messages.append({"role": "tool", "content": result})

        except asyncio.CancelledError:
            print()
            continue


def app(tools: list[type[BaseTool]] | None = None):
    asyncio.run(_app(tools=tools))


if __name__ == "__main__":
    app()
