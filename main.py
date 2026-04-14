import asyncio
import inspect
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, get_args, get_origin, get_type_hints

import ollama
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

MODEL = "lfm2:24b"

STYLE = Style.from_dict(
    {
        "prompt": "ansibrightgreen bold",
        "thinking": "ansidarkgray italic",
        "response": "ansiwhite",
        "tool": "ansiblue bold",
    }
)


@dataclass
class ThinkingEvent:
    """A fragment of the model's internal reasoning."""

    text: str


@dataclass
class ResponseEvent:
    """A fragment of the model's visible reply."""

    text: str


@dataclass
class ToolEvent:
    name: str
    input: dict
    output: str


def _py_to_json_type(hint: type) -> str:
    """Map a Python type to its JSON Schema type string.

    Handles both scalar types (str, int, float, bool) and generic aliases
    (list[...], dict[...], tuple[...]). Falls back to "string" for unknown types.
    """
    origin = get_origin(hint)
    if origin is not None:
        return {list: "array", dict: "object", tuple: "array"}.get(origin, "string")
    return {str: "string", int: "integer", float: "number", bool: "boolean"}.get(hint, "string")


class BaseTool:
    """Base class for tools that can be called by the model."""

    name: str = ""

    @classmethod
    def to_ollama(cls) -> dict:
        """Return the Ollama-compatible tool definition."""
        doc = inspect.getdoc(cls.execute) or ""
        description = doc.split("\n\n")[0].replace("\n", " ").strip()

        hints = get_type_hints(cls.execute, include_extras=True)
        sig = inspect.signature(cls.execute)

        properties = {}
        required = []
        for pname, param in sig.parameters.items():
            if pname in ("cls", "kwargs"):
                continue

            hint = hints.get(pname)
            if hint is not None and get_origin(hint) is Annotated:
                base, desc = get_args(hint)[0], get_args(hint)[1]
            else:
                base, desc = hint, ""

            properties[pname] = {"type": _py_to_json_type(base) if base else "string", "description": desc}
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    @classmethod
    async def execute(cls, **kwargs) -> str:
        """Execute the tool and return a result string."""
        raise NotImplementedError


class ReadFileTool(BaseTool):
    """Tool to read the contents of a file."""

    name = "read_file"

    @classmethod
    async def execute(cls, path: Annotated[str, "Absolute or relative path to the file to read."], **kwargs) -> str:
        """Read the contents of a file on disk and return them as a string."""
        try:
            return await asyncio.to_thread(Path(path).read_text)
        except OSError as e:
            return f"Error reading file: {e}"
        
class WeatherApiTool(BaseTool):
    name = "stub"

    @classmethod
    async def execute(cls) -> str:
        """Get the weather"""
        return "The weather is sunny today"


class ChatBackend:
    """Manages ollama interactions and conversation history."""

    def __init__(self, model: str = MODEL, system_prompt=None, tools: list = None) -> None:
        self.model = model
        self.client = ollama.AsyncClient()
        self.messages: list[dict[str, str]] = [] if system_prompt is None else [system_prompt]
        self.tools = [] if tools is None else tools

    async def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return its result."""
        for tool_cls in self.tools:
            if tool_cls.name == tool_name:
                return await tool_cls.execute(**arguments)
        return f"Unknown tool: {tool_name}"

    async def stream(self, user_input: str | None = None) -> AsyncGenerator[ThinkingEvent | ResponseEvent, None]:
        """Stream response events from the model.

        Yields ThinkingEvent and ResponseEvent objects. Handles tool execution
        internally—when the model calls a tool, it executes it and continues
        generating the response based on the tool output.
        """
        if user_input is not None:
            self.messages.append({"role": "user", "content": user_input})

        while True:
            response = ""
            tool_calls: list[dict] = []

            async for part in await self.client.chat(
                model=self.model,
                messages=self.messages,
                stream=True,
                think=False,
                tools=[tool.to_ollama() for tool in self.tools] or None,
            ):
                if data := part.message.thinking:
                    yield ThinkingEvent(data)

                if data := part.message.content:
                    response += data
                    yield ResponseEvent(data)

                if part.message.tool_calls:
                    tool_calls.extend(
                        {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in part.message.tool_calls
                    )

            msg = {"role": "assistant", "content": response}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            self.messages.append(msg)

            if not tool_calls:  # exit when no tools left
                break

            # Execute tools and append results
            for tc in tool_calls:
                result = await self._execute_tool(tc["function"]["name"], tc["function"]["arguments"])

                yield ToolEvent(
                    name=tc["function"]["name"],
                    input=tc["function"]["arguments"],
                    output=result,
                )
                self.messages.append({"role": "tool", "content": result})

            # Loop continues: client.chat() is called again with updated messages


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

    async def render_stream(self, events: AsyncGenerator[ThinkingEvent | ResponseEvent | ToolEvent, None]) -> None:
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
                    fmt = "thinking"
                    print_formatted_text(
                        FormattedText([(f"class:{fmt}", event.text)]),
                        end="",
                        flush=True,
                        style=self.style,
                    )
                    prev_mode = "thinking"

                case ResponseEvent():
                    if prev_mode is not None and prev_mode != "response":
                        print()
                    fmt = "response"
                    print_formatted_text(
                        FormattedText([(f"class:{fmt}", event.text)]),
                        end="",
                        flush=True,
                        style=self.style,
                    )
                    prev_mode = "response"

                case ToolEvent():
                    if prev_mode is not None:
                        print()
                    print_formatted_text(
                        FormattedText(
                            [
                                ("class:tool", f"[tool: {event.name}]\n"),
                                ("class:tool", f"input: {event.input}\n"),
                                ("class:tool", f"output: {event.output}"),
                            ]
                        ),
                        end="",
                        flush=True,
                        style=self.style,
                    )
                    prev_mode = "tool"

        print()  # Trailing newline after stream completes


async def main(tools: list[type[BaseTool]] | None = None):
    """Run the interactive chat application."""
    backend = ChatBackend(MODEL, tools=tools or [])
    ui = TerminalUI()

    print(f"Running model {MODEL}. Ensure the context window has been turned up for optimal usage")

    while True:
        try:
            user_input = await ui.get_input()
        except KeyboardInterrupt:
            break

        try:
            await ui.render_stream(backend.stream(user_input))
        except asyncio.CancelledError:
            print()
            continue


if __name__ == "__main__":
    asyncio.run(main([WeatherApiTool]))
