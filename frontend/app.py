import argparse
import asyncio
import signal
import tomllib
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.shortcuts.choice_input import ChoiceInput
from prompt_toolkit.styles import Style
from pydantic import BaseModel, Field
from rich.console import Console
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
from frontend.input import Kitty, get_input_bindings

MAX_TOOL_OUTPUT = 500

SYSTEM_PROMPT_PATH = Path(__file__).parent / "files" / "system_prompt.md"

TOOL_NAME_MAP: dict[str, type] = {tool.name: tool for tool in ALLOWED_TOOLS}


class AppConfig(BaseModel):
    model: str
    provider: str
    think: bool = False
    show_think: bool = True
    system_prompt_path: str | None = None
    enabled_tools: list[str] = Field(default_factory=lambda: [t.name for t in ALLOWED_TOOLS])


# _ISO2022_RE = re.compile(r"\x1b[()][0-9A-Za-z]")

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


class _ShowUsageMessage(Exception):
    pass


class Command:
    name: str
    description: str
    usage_message: str | None = None
    completions: dict[str, Any] | None = None

    @classmethod
    async def execute(cls, frontend: "ChatFrontend", args: list[str]) -> None:
        try:
            await cls._execute(frontend, args)
        except _ShowUsageMessage:
            if cls.usage_message:
                print_formatted_text(
                    FormattedText([("class:cmd", cls.usage_message + "\n")]),
                    style=_CMD_STYLE,
                )

    @staticmethod
    async def _execute(frontend: "ChatFrontend", args: list[str]) -> None:
        raise NotImplementedError


class HelpCommand(Command):
    name = "help"
    description = "Show available commands"
    completions = None

    @staticmethod
    async def _execute(frontend: "ChatFrontend", args: list[str]) -> None:
        lines: list[tuple[str, str]] = [("class:head", "Available commands:\n\n")]
        for cmd_name in sorted(frontend._commands.keys()):
            cmd = frontend._commands[cmd_name]
            lines += [
                ("class:cmd", f"  /{cmd.name}"),
                ("", "  "),
                ("class:desc", cmd.description),
                ("", "\n"),
            ]
            if cmd.completions:
                keys = ", ".join(sorted(cmd.completions.keys()))
                lines.append(("class:args", f"    args: {keys}\n"))
            if cmd.usage_message:
                lines.append(("class:args", f"    {cmd.usage_message}\n"))
        lines.append(("", "\n"))
        print_formatted_text(FormattedText(lines), style=_CMD_STYLE)


class QuitCommand(Command):
    name = "quit"
    description = "Exit the application"
    completions = None

    @staticmethod
    async def _execute(frontend: "ChatFrontend", args: list[str]) -> None:
        raise EOFError()


class ModelCommand(Command):
    name = "model"
    description = "List available models and select interactively, or set by name"
    completions = None

    @staticmethod
    async def _execute(frontend: "ChatFrontend", args: list[str]) -> None:
        if args:
            await frontend._backend.config.set_model(args[0])
            print_formatted_text(
                FormattedText([("class:cmd", f"Model set to: {args[0]}\n")]),
                style=_CMD_STYLE,
            )
            return

        models = await frontend._backend.config.get_available_models()
        current = frontend._backend.config.model
        sorted_models = sorted(models, key=lambda m: m.name)

        names = [m.name for m in sorted_models]
        sizes = [m.parameter_size or "" for m in sorted_models]
        quants = [m.quantization_level or "" for m in sorted_models]

        name_w = max((len(n) for n in names), default=0)
        size_w = max((len(s) for s in sizes), default=0)

        values = []
        for m, name, size, quant in zip(sorted_models, names, sizes, quants):
            rest = f"{name:<{name_w}}  {size:<{size_w}}  {quant}"[len(name) :]
            label = (
                FormattedText([("class:current-marker", name), ("", rest)])
                if m.name == current
                else f"{name:<{name_w}}  {size:<{size_w}}  {quant}"
            )
            values.append((m.name, label))

        escape_kb = KeyBindings()

        @escape_kb.add("escape")
        def _escape(event: Any) -> None:
            event.app.exit(result=None)

        @escape_kb.add("c-c")
        def _ctrl_c(event: Any) -> None:
            event.app.exit(result=None)

        choice_input = ChoiceInput(
            message=f"Select Model (current: {current})",
            options=values,
            default=current,
            style=_CMD_STYLE,
            key_bindings=escape_kb,
        )
        app = choice_input._create_application()
        app.erase_when_done = True

        try:
            selected = await asyncio.to_thread(app.run)
        except KeyboardInterrupt:
            selected = None

        if selected and selected != current:
            await frontend._backend.config.set_model(selected)
            print_formatted_text(
                FormattedText([("class:cmd", f"Model set to: {selected}\n")]),
                style=_CMD_STYLE,
            )


class ThinkCommand(Command):
    name = "think"
    description = "Toggle extended thinking mode on/off, or control thinking output visibility with 'view'"
    usage_message = "Usage: /think on|off|view [show|hide]"
    completions = {
        "on": None,
        "off": None,
        "view": {"show": None, "hide": None},  # Separate so it is clear that it is changing the view
    }

    @staticmethod
    async def _execute(frontend: "ChatFrontend", args: list[str]) -> None:
        if not args or args[0].lower() not in ("on", "off", "view"):
            raise _ShowUsageMessage()
        match val := args[0].lower():
            case "on" | "off":
                action = val == "on"
                frontend._backend.config.set_think(action)
                value = args[0].lower() == "on"
                state = "enabled" if value else "disabled"
                print_formatted_text(
                    FormattedText([("class:cmd", f"Thinking {state}\n")]),
                    style=_CMD_STYLE,
                )
            case "view":
                sub = args[1].lower() if len(args) > 1 else None
                match sub:
                    case "show":
                        frontend._show_thinking = True
                    case "hide":
                        frontend._show_thinking = False
                    case None:
                        frontend._show_thinking = not frontend._show_thinking
                    case _:
                        raise _ShowUsageMessage()
                state = "visible" if frontend._show_thinking else "hidden"
                print_formatted_text(
                    FormattedText([("class:cmd", f"Thinking output {state}\n")]),
                    style=_CMD_STYLE,
                )


def _fmt_tool_input(inp: dict | str) -> str:
    if not isinstance(inp, dict):
        return repr(inp)
    items = list(inp.items())
    if len(items) == 1:
        return repr(items[0][1])
    return ", ".join(f"{k}={v!r}" for k, v in items)


class ChatFrontend:
    def __init__(self, backend: Backend, show_think: bool = True) -> None:
        self._backend = backend
        self._console = Console(highlight=False, markup=True)
        self._show_thinking: bool = show_think

        self._commands: dict[str, type[Command]] = {}
        self._prompt = PromptSession(
            completer=self._register_commands(
                [
                    HelpCommand,
                    QuitCommand,
                    ModelCommand,
                    ThinkCommand,
                ]
            )
        )

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

    def run(self) -> None:
        """Sync entry point — creates event loop and runs the chat."""
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        async with self._backend:
            self._print_header()
            events_gen = self._backend.stream_events()

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
                    if self._active_live is not None:
                        self._active_live.stop()
                        self._active_live = None
                        self._console.print()
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


def load_config(path: Path, preset: str | None = None) -> AppConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    if not data:
        raise SystemExit("Config file is empty")

    if preset is None:
        preset = next(iter(data))

    if preset not in data:
        raise SystemExit(f"Preset '{preset}' not found. Available: {list(data.keys())}")

    return AppConfig.model_validate(data[preset])


def generate_config(path: Path) -> None:
    all_tools = [t.name for t in ALLOWED_TOOLS]
    wikipedia_tools = ["search_wikipedia", "get_wikipedia_page"]

    lines = [
        "[base]",
        'model = "qwen2.5:3b"',
        'provider = "ollama"',
        "think = false",
        "show_think = true",
        '# system_prompt_path = "/path/to/system_prompt.md"',
        f"enabled_tools = {all_tools!r}".replace("'", '"'),
        "",
        "[search_wikipedia]",
        'model = "qwen2.5:3b"',
        'provider = "ollama"',
        f"enabled_tools = {wikipedia_tools!r}".replace("'", '"'),
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"Config written to {path}")


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

    tools = []
    for name in cfg.enabled_tools:
        if name not in TOOL_NAME_MAP:
            raise SystemExit(f"Unknown tool '{name}'. Valid: {list(TOOL_NAME_MAP.keys())}")
        tools.append(TOOL_NAME_MAP[name])

    provider = (
        OllamaProvider() if cfg.provider == "ollama" else OpenAICompatibleProvider(base_url=f"http://{cfg.provider}/v1")
    )

    system_prompt = None
    prompt_path = Path(cfg.system_prompt_path) if cfg.system_prompt_path else SYSTEM_PROMPT_PATH
    if prompt_path.exists():
        system_prompt = prompt_path.read_text()

    backend = Backend(provider=provider, model=cfg.model, think=cfg.think, tools=tools, system_prompt=system_prompt)
    ChatFrontend(backend=backend, show_think=cfg.show_think).run()
