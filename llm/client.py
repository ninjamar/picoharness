from collections.abc import AsyncGenerator

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

from llm.events import ResponseEvent, ThinkingEvent, ToolEvent

STYLE = Style.from_dict(
    {
        "prompt": "ansibrightgreen bold",
        "thinking": "ansidarkgray italic",
        "response": "ansiwhite",
        "tool": "ansiblue bold",
    }
)


class Client:
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
