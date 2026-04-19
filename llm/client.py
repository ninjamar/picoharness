from collections.abc import AsyncGenerator

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

from .events import ResponseEvent, ThinkingEvent, ToolEndEvent, ToolStartEvent

STYLE = Style.from_dict(
    {
        "prompt": "ansibrightgreen bold",
        "thinking": "ansidarkgray italic",
        "response": "ansiwhite",
        "tool_start": "ansiblue bold",
        "tool_end": "ansipurple bold",
    }
)


class TerminalUI:
    """Handles terminal I/O: prompts, styled output, and formatted text rendering."""

    def __init__(self, config, style: Style = STYLE) -> None:
        self.config = config
        self.session = PromptSession()
        self.style = style

    async def get_input(self) -> str:
        """Get user input from the prompt, looping on empty input."""
        while True:
            with patch_stdout():
                user_input = await self.session.prompt_async(
                    FormattedText([("class:prompt", ">>> ")]),
                    style=self.style,
                )

            if s := user_input.strip():
                return s

    async def render_stream(
        self, events: AsyncGenerator[ThinkingEvent | ResponseEvent | ToolStartEvent | ToolEndEvent, None]
    ) -> None:
        """Consume and render a stream of chat events.

        Handles mode transitions between thinking, response, and tool events,
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

                case ToolStartEvent():
                    if prev_mode is not None:
                        print()
                    print_formatted_text(
                        FormattedText(
                            [
                                ("class:tool_start", f"[tool: {event.name}]\n"),
                                ("class:tool_start", f"input: {event.input}"),
                            ]
                        ),
                        end="\n",
                        flush=True,
                        style=self.style,
                    )
                    prev_mode = "tool"

                case ToolEndEvent():
                    print_formatted_text(
                        FormattedText(
                            [
                                ("class:tool_end", f"output ({event.id[:8]}): {event.output}"),
                            ]
                        ),
                        end="\n",
                        flush=True,
                        style=self.style,
                    )

        print()  # Trailing newline after stream completes
