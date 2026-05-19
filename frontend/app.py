import argparse
import asyncio
import signal
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.shortcuts import print_formatted_text
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from backend.api import (
    ALL_TOOLS,
    BackendAPI,
    DoneEvent,
    ResponseEvent,
    ThinkingEvent,
    ToolErrorEvent,
    ToolOutputEvent,
    ToolStartEvent,
    UserInputEvent,
)
from backend.provider import OllamaProvider, OpenAICompatibleProvider
from frontend.commands import build_all_commands
from frontend.config_io import generate_config, load_config
from frontend.input import Kitty, get_input_bindings
from frontend.style import _CMD_STYLE

if TYPE_CHECKING:
    from frontend.commands._internal import Command

MAX_TOOL_OUTPUT = 500

SYSTEM_PROMPT_PATH = Path(__file__).parent / "files" / "system_prompt.md"


def _fmt_tool_input(inp: dict | str) -> str:
    if not isinstance(inp, dict):
        return repr(inp)
    items = list(inp.items())
    if len(items) == 1:
        return repr(items[0][1])
    return ", ".join(f"{k}={v!r}" for k, v in items)


class ChatFrontend:
    def __init__(self, api: BackendAPI, show_think: bool = True) -> None:
        self.api = api  # Public property for schema getters/setters
        self._console = Console(highlight=False, markup=True)
        self._show_thinking: bool = show_think

        # Build commands from schema + hand-written help/quit
        all_commands = list(build_all_commands(self).values())

        self._commands: dict[str, type[Command]] = {}
        self._prompt = PromptSession(completer=self._register_commands(all_commands))

        self._bindings = get_input_bindings()

        self.kitty = Kitty()

    def _register_commands(self, commands: list[type[Command]]) -> NestedCompleter:
        self._commands = {cmd.name: cmd for cmd in commands}
        nested: dict[str, Any] = {f"/{cmd.name}": (cmd.completions or None) for cmd in commands}
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
        await self._commands[name].execute(self, args)

    @property
    def show_think(self) -> bool:
        return self._show_thinking

    def set_show_think(self, value: bool) -> None:
        """Set whether to display thinking output (used by schema setter)."""
        self._show_thinking = value

    def run(self) -> None:
        """Sync entry point — creates event loop and runs the chat."""
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        async with self.api:
            self._print_header()
            events_gen = self.api.stream_events()

            while True:
                try:
                    with self.kitty:  # Kitty protocol for input and menus
                        user_text = await self._read_input()

                        if not user_text.strip():
                            continue

                        if user_text.startswith("/"):
                            await self._dispatch_command(user_text)
                            continue
                except KeyboardInterrupt, EOFError:
                    self._console.print("\n[dim]Bye.[/dim]")
                    break

                input_id = str(uuid.uuid4())

                self.api.feed(input_id, user_text)

                loop = asyncio.get_event_loop()
                task = asyncio.ensure_future(self._collect_turn(input_id, events_gen))
                loop.add_signal_handler(signal.SIGINT, task.cancel)
                try:
                    await task
                except asyncio.CancelledError:
                    # Pressing ctrl-c causes the cancellation which then gets sent to the backend.
                    # For consistency, the backend still sends DoneEvent with interrupt = true
                    self.api.cancel_current()
                    if self._active_live is not None:
                        self._active_live.stop()
                        self._active_live = None
                        self._console.print()
                    self._console.print("\n[dim]Interrupted.[/dim]")
                    events_gen = self.api.stream_events()
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
        response_buffer = ""
        self._active_live: Live | None = None

        async for event in events_gen:
            if event.id != input_id:
                continue

            # Commit any active response Live before handling non-response events
            if not isinstance(event, ResponseEvent) and self._active_live is not None:
                self._active_live.stop()
                self._active_live = None
                response_buffer = ""
                self._console.print()

            match event:
                case UserInputEvent():
                    pass
                case ThinkingEvent(fragment=f):
                    if self._show_thinking:
                        self._console.print(Text(f, style="dim italic"), end="")
                case ResponseEvent(fragment=f):
                    response_buffer += f
                    if self._active_live is None:
                        self._active_live = Live(
                            Markdown(response_buffer),
                            console=self._console,
                            auto_refresh=False,
                            vertical_overflow="ellipsis",
                        )
                        self._active_live.start()
                    self._active_live.update(Markdown(response_buffer), refresh=True)
                case ToolStartEvent(tool_name=name, tool_input=inp):
                    label = f"⏺ {name}({_fmt_tool_input(inp)})"
                    self._console.print(label, style="bold blue")
                case ToolOutputEvent(result=result, output_format=fmt):
                    match fmt:
                        case "all":
                            self._console.print(result, style="cyan dim")
                        case "truncate":
                            if len(result) > MAX_TOOL_OUTPUT:
                                result = result[:MAX_TOOL_OUTPUT] + "… [truncated]"
                            indented = "\n".join(f"  {line}" for line in result.splitlines())
                            self._console.print(indented, style="cyan dim")
                        case "none":
                            pass
                case ToolErrorEvent(error=err):
                    self._console.print(f"  Error: {err}", style="red")
                case DoneEvent(error=error):
                    if error:
                        self._console.print(f"Error: {error}", style="red")

            if isinstance(event, DoneEvent):
                break

        if self._active_live is not None:
            self._active_live.stop()
            self._active_live = None
            self._console.print()

    def _print_header(self) -> None:
        self._console.rule("[bold]LocalAI Chat[/bold]")
        newline_hint = "Shift+Enter" if self.kitty.use_kitty else "Ctrl+J"
        self._console.print(f"[dim]Ctrl+C or Ctrl+D to quit • {newline_hint} for newline • /help for help[/dim]\n")


def cli() -> None:
    parser = argparse.ArgumentParser(description="LocalAI TUI")
    parser.add_argument("--config", default=None, help="Path to TOML config file")
    parser.add_argument("--preset", default=None, help="Preset name (default: first section)")
    parser.add_argument(
        "--generate-config",
        metavar="PATH",
        default=None,
        help="Generate a sample config file at PATH and exit",
    )
    args = parser.parse_args()

    if args.generate_config:
        generate_config(Path(args.generate_config))
        return

    if not args.config:
        parser.error("--config is required (or use --generate-config PATH to create one)")

    cfg = load_config(Path(args.config), args.preset)

    tool_name_map: dict[str, type] = {tool.name: tool for tool in ALL_TOOLS}
    tools = []
    tool_names = cfg.enabled_tools if cfg.enabled_tools else list(tool_name_map.keys())
    for name in tool_names:
        if name not in tool_name_map:
            raise SystemExit(f"Unknown tool '{name}'. Valid: {list(tool_name_map.keys())}")
        tools.append(tool_name_map[name])

    provider = (
        OllamaProvider() if cfg.provider == "ollama" else OpenAICompatibleProvider(base_url=f"http://{cfg.provider}/v1")
    )

    system_prompt = None
    prompt_path = Path(cfg.system_prompt_path) if cfg.system_prompt_path else SYSTEM_PROMPT_PATH
    if prompt_path.exists():
        system_prompt = prompt_path.read_text()

    api = BackendAPI(
        provider=provider,
        model=cfg.model,
        think=cfg.think,
        tool_classes=tools,
        system_prompt=system_prompt,
        system_prompt_path=cfg.system_prompt_path,
    )
    ChatFrontend(api=api, show_think=cfg.show_think).run()
