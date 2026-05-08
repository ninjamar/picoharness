import argparse
import asyncio
import signal
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import choice, print_formatted_text
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text
from tabulate import tabulate

from backend import (
    Backend,
    DoneEvent,
    Event,
    ModelInfo,
    ResponseEvent,
    ThinkingEvent,
    ToolErrorEvent,
    ToolOutputEvent,
    ToolStartEvent,
    UserInputEvent,
)
from backend.backend import ALLOWED_TOOLS
from backend.provider.provider import OllamaProvider, OpenAICompatibleProvider
from frontend.kitty import Kitty, get_input_bindings

MAX_TOOL_OUTPUT = 500

SYSTEM_PROMPT_PATH = Path(__file__).parent / "files" / "system_prompt.md"

# _ISO2022_RE = re.compile(r"\x1b[()][0-9A-Za-z]")


@dataclass
class _ThinkingSegment:
    text: str


@dataclass
class _ResponseSegment:
    text: str


@dataclass
class Command:
    name: str
    description: str
    handler: Callable[["ChatFrontend", list[str]], Awaitable[None]]
    completions: dict[str, Any] | None = None


_Segment = _ThinkingSegment | _ResponseSegment | ToolStartEvent | ToolOutputEvent | ToolErrorEvent

_CMD_STYLE = Style.from_dict(
    {
        "cmd": "bold cyan",
        "desc": "italic",
        "head": "bold underline",
        "args": "dim",
        "selected-option": "fg:ansigreen bold",
        "number": "fg:ansicyan",
        "current-marker": "bold",
    }
)


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

        self._commands: dict[str, Command] = {}
        self._prompt = PromptSession(
            completer=self._register_commands(
                [
                    Command("help", "Show available commands", ChatFrontend._cmd_help),
                    Command("quit", "Exit the application", ChatFrontend._cmd_quit),
                    Command("model", "View and select the active model", ChatFrontend._cmd_model),
                ]
            )
        )

        self._bindings = self._setup_bindings()

        self.kitty = Kitty()

    def _register_commands(self, commands: list[Command]) -> NestedCompleter:
        self._commands = {cmd.name: cmd for cmd in commands}
        nested: dict[str, Any] = {f"/{name}": (cmd.completions or None) for name, cmd in self._commands.items()}
        return NestedCompleter.from_nested_dict(nested)

    async def _dispatch_command(self, raw: str) -> None:
        parts = raw.lstrip("/").split()
        if not parts:
            return
        name, args = parts[0], parts[1:]
        if name not in self._commands:
            print_formatted_text(
                FormattedText([("class:cmd", f"Unknown command: /{name}. Type /help for available commands.\n")]),
                style=_CMD_STYLE,
            )
            return
        await self._commands[name].handler(self, args)

    async def _cmd_help(self, args: list[str]) -> None:
        lines: list[tuple[str, str]] = [("class:head", "Available commands:\n\n")]
        for name in sorted(self._commands.keys()):
            cmd = self._commands[name]
            lines += [
                ("class:cmd", f"  /{cmd.name}"),
                ("", "  "),
                ("class:desc", cmd.description),
                ("", "\n"),
            ]
            if cmd.completions:
                keys = ", ".join(sorted(cmd.completions.keys()))
                lines.append(("class:args", f"    args: {keys}\n"))
        lines.append(("", "\n"))
        print_formatted_text(FormattedText(lines), style=_CMD_STYLE)

    async def _cmd_quit(self, args: list[str]) -> None:
        raise EOFError()

    async def _cmd_model(self, args: list[str]) -> None:
        models = await self._backend.config.get_available_models()
        current = self._backend.config.model

        # Sort models by name
        sorted_models = sorted(models, key=lambda m: m.name)

        # Prepare rows for table formatting
        rows = []
        for m in sorted_models:
            rows.append(
                [
                    m.name,
                    m.parameter_size or "",
                    m.quantization_level or "",
                ]
            )

        # Format as table using tabulate
        table_str = tabulate(rows, headers=["NAME", "SIZE", "QUANT"], tablefmt="plain")

        # Extract formatted rows (skip header line)
        table_lines = table_str.split("\n")
        formatted_rows = [line for line in table_lines[1:] if line.strip()]

        # Build values tuples from sorted models and formatted rows
        # Make the model name bold if it's the currently used model
        values = []
        for m, row in zip(sorted_models, formatted_rows):
            if m.name == current:
                # Make only the name part bold
                name_len = len(m.name)
                label = FormattedText([("class:current-marker", row[:name_len]), ("", row[name_len:])])
            else:
                label = row
            values.append((m.name, label))

        selected = await asyncio.to_thread(
            choice,
            message=f"Select Model (current: {current})",
            options=values,
            default=current,
            style=_CMD_STYLE,
        )

        if selected and selected != current:
            await self._backend.config.set_model(selected)
            print_formatted_text(
                FormattedText([("class:cmd", f"Model set to: {selected}\n")]),
                style=_CMD_STYLE,
            )

    def _setup_bindings(self) -> KeyBindings:
        bindings = get_input_bindings()

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
        async with self._backend:
            self._print_header()
            events_gen = self._backend.stream_events()

            while True:
                with self.kitty:
                    try:
                        user_text = await self._read_input()

                        if not user_text.strip():
                            continue

                        if user_text.startswith("/"):
                            await self._dispatch_command(user_text)
                            continue
                    except KeyboardInterrupt, EOFError:  # This is valid Python 3.14 synax: see PEP PEP 758
                        self._console.print("\n[dim]Bye.[/dim]")
                        break

                input_id = str(uuid.uuid4())

                self._backend.feed(input_id, user_text)

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

                self._console.print()

    async def _read_input(self) -> str:
        return await self._prompt.prompt_async(
            "> ",
            multiline=True,
            key_bindings=self._bindings,
            # message=ANSI(
            #     # However, the ANSI class (which handles the escape codes from blessed) doesn't
            #     # support some of blessed's generated output --a zero width character or something.
            #     # So, a regex is used to remove it.
            #     _ISO2022_RE.sub("", self._term.bold_green(">>> "))
            # )
        )

    async def _collect_turn(self, input_id: str, events_gen: AsyncGenerator) -> None:
        segments: list[_Segment] = []
        with Live(console=self._console, auto_refresh=False, vertical_overflow="visible") as live:
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
        newline_hint = "Shift+Enter" if self.kitty.use_kitty else "Ctrl+J"
        self._console.print(f"[dim]Ctrl+C or Ctrl+D to quit • {newline_hint} for newline • /help for help[/dim]\n")


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
