import argparse
import asyncio
import signal
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from backend import (
    Backend,
    DoneEvent,
    Event,
    ResponseEvent,
    ThinkingEvent,
    ToolErrorEvent,
    ToolOutputEvent,
    ToolStartEvent,
    UserInputEvent,
)
from backend.backend import ALLOWED_TOOLS
from backend.provider.provider import OllamaProvider, OpenAICompatibleProvider
from frontend import kitty

MAX_TOOL_OUTPUT = 500

SYSTEM_PROMPT_PATH = Path(__file__).parent / "files" / "system_prompt.md"


@dataclass
class _ThinkingSegment:
    text: str


@dataclass
class _ResponseSegment:
    text: str


_Segment = _ThinkingSegment | _ResponseSegment | ToolStartEvent | ToolOutputEvent | ToolErrorEvent


def _fmt_tool_input(inp: dict | str) -> str:
    if not isinstance(inp, dict):
        return repr(inp)
    items = list(inp.items())
    if len(items) == 1:
        return repr(items[0][1])
    return ", ".join(f"{k}={v!r}" for k, v in items)


def _build_live(segments):
    """Build live display from typed segments."""
    parts = []
    for seg in segments:
        match seg:
            case _ThinkingSegment(text=text):
                parts.append(Text(text, style="dim italic"))
            case _ResponseSegment(text=text):
                parts.append(Markdown(text))
            case ToolStartEvent(tool_name=name, tool_input=inp):
                parts.append(Text(f"\n⏺ {name}({_fmt_tool_input(inp)})", style="bold blue"))
            case ToolOutputEvent(result=result, output_format=fmt):
                match fmt:
                    case "all":
                        parts.append(Text(result, style="dim cyan"))
                    case "truncate":
                        if len(result) > MAX_TOOL_OUTPUT:
                            result = result[:MAX_TOOL_OUTPUT] + "… [truncated]"
                        lines = [f"  {line}" for line in result.splitlines()]
                        parts.append(Text("\n".join(lines), style="dim cyan"))
                    case "none":
                        pass
            case ToolErrorEvent(error=err):
                parts.append(Text(f"  Error: {err}", style="red"))
            case DoneEvent(error=error):
                if error:
                    parts.append(Text(f"Error: {error}", style="red"))

    return Group(*parts) if parts else Group()


class ChatFrontend:
    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self._console = Console(highlight=False, markup=True)
        self._prompt = PromptSession()

        self._live: Live | None = None

        self._use_kitty = kitty.detect_kitty()
        self._bindings = self._setup_bindings()

        if self._use_kitty:
            kitty.init_kitty()  # TODO: should the use guard be included in init_kitty?

    def _setup_bindings(self) -> KeyBindings:
        bindings = kitty.make_input_bindings()

        @bindings.add("c-c")
        def _(event):
            # If there's text in the buffer, clear it; otherwise exit
            if event.app.current_buffer.text:
                event.app.current_buffer.text = ""
            else:
                event.app.exit(exception=KeyboardInterrupt())

        return bindings

    def run(self) -> None:
        """Sync entry point — creates event loop and runs the chat."""
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        try:
            async with self._backend:
                self._print_header()
                events_gen = self._backend.stream_events()

                while True:
                    try:
                        user_text = await self._read_input()
                    except KeyboardInterrupt, EOFError:  # This is valid Python 3.14 synax: see PEP PEP 758
                        self._console.print("\n[dim]Bye.[/dim]")
                        break
                    if not user_text.strip():
                        continue

                    input_id = str(uuid.uuid4())

                    self._backend.feed(input_id, user_text)

                    if self._use_kitty:
                        kitty.end_kitty()
                    loop = asyncio.get_event_loop()
                    task = asyncio.ensure_future(self._collect_turn(input_id, events_gen))
                    loop.add_signal_handler(signal.SIGINT, task.cancel)
                    try:
                        await task
                    except asyncio.CancelledError:
                        # Pressing ctrl-c causes the cancellation which then gets sent to the backend.
                        # For consistency, the backend still sends DoneEvent with interrupt = true
                        self._backend.cancel_current()
                        self._console.print("\n[dim]Interrupted.[/dim]")
                        events_gen = self._backend.stream_events()
                    finally:
                        loop.remove_signal_handler(signal.SIGINT)
                        if self._use_kitty:
                            kitty.init_kitty()

                    self._console.print()
        finally:
            if self._use_kitty:
                kitty.end_kitty()

    async def _read_input(self) -> str:
        return await self._prompt.prompt_async(
            "> ",
            multiline=True,
            key_bindings=self._bindings,
        )

    async def _collect_turn(self, input_id: str, events_gen: AsyncGenerator) -> None:
        segments: list[_Segment] = []
        with Live(console=self._console, auto_refresh=False) as live:
            async for event in events_gen:
                if event.id != input_id:
                    # TODO: This doesn't handle a non-linear flow: other events are just dropped
                    # What should happen: events are always caught, then they are put in a queue.
                    # Then this function would operate on that queue
                    continue
                self._render_event(live, segments, event)
                if isinstance(event, DoneEvent):
                    # This should NOT be in _render_event as _collecting turn stops here.
                    # Only rendered in case of error
                    # if event.error:
                    #    self._console.print(f"\n[red]Error: {event.error}[/red]")
                    break

    def _render_event(self, live: Live, segments, event: Event) -> None:
        match event:
            case UserInputEvent():
                return
            case DoneEvent():
                # There could be an error, so display it
                segments.append(event)
            case ThinkingEvent(fragment=f):
                # Merge segments
                if segments and isinstance(segments[-1], _ThinkingSegment):
                    segments[-1].text += f
                else:
                    segments.append(_ThinkingSegment(f))
            case ResponseEvent(fragment=f):
                if segments and isinstance(segments[-1], _ResponseSegment):
                    segments[-1].text += f
                else:
                    segments.append(_ResponseSegment(f))
            case ToolStartEvent() | ToolOutputEvent() | ToolErrorEvent():
                segments.append(event)

        live.update(_build_live(segments))
        live.refresh()

    def _print_header(self) -> None:
        self._console.rule("[bold]LocalAI Chat[/bold]")
        newline_hint = "Shift+Enter" if self._use_kitty else "Ctrl+J"
        self._console.print(f"[dim]Ctrl+C or Ctrl+D to quit • {newline_hint} for newline[/dim]\n")


def cli() -> None:
    parser = argparse.ArgumentParser(description="LocalAI TUI")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", required=True, help="'ollama' or 'host:port' for OpenAI-compatible")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--system-prompt-path",
        default=None,
        help="Path to a markdown file used as the system prompt",
    )
    args = parser.parse_args()

    tools = ALLOWED_TOOLS

    if args.provider == "ollama":
        provider = OllamaProvider()
    else:
        provider = OpenAICompatibleProvider(base_url=f"http://{args.provider}/v1")

    system_prompt = None
    prompt_path = Path(args.system_prompt_path) if args.system_prompt_path else SYSTEM_PROMPT_PATH
    if prompt_path.exists():
        system_prompt = prompt_path.read_text()
    backend = Backend(provider=provider, model=args.model, think=args.think, tools=tools, system_prompt=system_prompt)
    ChatFrontend(backend=backend).run()
