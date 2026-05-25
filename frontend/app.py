from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.shortcuts import checkboxlist_dialog, input_dialog, radiolist_dialog
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from backend.api import (
    ALL_TOOLS,
    BackendAPI,
    DoneEvent,
    Event,
    ResponseEvent,
    ThinkingEvent,
    ToolErrorEvent,
    ToolOutputEvent,
    ToolStartEvent,
    UserInputEvent,
)
from backend.provider import OllamaProvider, OpenAICompatibleProvider
from frontend.config_io import generate_config, load_config
from frontend.schema import (
    CUSTOM_PROVIDER_LABEL,
    FIELDS,
    DialogueMenu,
    MultiSelectMenu,
    TextInputMenu,
    ToggleMenu,
    resolve_choices,
)

MAX_TOOL_OUTPUT = 500
SYSTEM_PROMPT_PATH = Path(__file__).parent / "files" / "system_prompt.md"
UPDATE_INTERVAL = 0.05  # seconds between Live refresh calls


def _fmt_tool_input(inp: dict | str) -> str:
    if not isinstance(inp, dict):
        return repr(inp)
    items = list(inp.items())
    if len(items) == 1:
        return repr(items[0][1])
    return ", ".join(f"{k}={v!r}" for k, v in items)


def _build_completer() -> NestedCompleter:
    nested: dict = {f"/{f.name}": None for f in FIELDS}
    nested["/help"] = None
    nested["/quit"] = None
    return NestedCompleter.from_nested_dict(nested)


def _build_key_bindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("c-c")
    def _(event) -> None:
        buf = event.app.current_buffer
        if buf.text:
            buf.reset()
        else:
            event.app.exit(exception=KeyboardInterrupt())

    return kb


async def _run_dialog(app):
    kb = KeyBindings()

    @kb.add("escape")
    def _(event) -> None:
        event.app.exit(result=None)

    app.key_bindings = merge_key_bindings([app.key_bindings, kb])
    return await app.run_async()


class ChatFrontend:
    def __init__(self, api: BackendAPI, show_think: bool = True) -> None:
        self.api = api
        self._show_thinking = show_think
        self._console = Console(highlight=False, markup=True)
        self._prompt: PromptSession = PromptSession(
            completer=_build_completer(),
            complete_while_typing=True,
            key_bindings=_build_key_bindings(),
        )

    @property
    def show_think(self) -> bool:
        return self._show_thinking

    def set_show_think(self, value: bool) -> None:
        self._show_thinking = value

    def run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        async with self.api:
            self._print_header()
            while True:
                try:
                    text = await self._prompt.prompt_async("> ")
                except KeyboardInterrupt, EOFError:
                    self._console.print("\n[red]Bye.[/red]")
                    break
                text = text.strip()
                if not text:
                    continue
                if text.startswith("/"):
                    await self._dispatch_command(text)
                else:
                    input_id = str(uuid.uuid4())
                    self.api.feed(input_id, text)
                    try:
                        await self._collect_turn(input_id)
                    except KeyboardInterrupt, asyncio.CancelledError:
                        self.api.cancel_current()
                        self._console.print("[red]Interrupted.[/red]")

    async def _dispatch_command(self, raw: str) -> None:
        parts = raw[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "quit":
            raise SystemExit(0)
        if cmd == "help":
            self._show_help()
            return

        field_map = {f.name: f for f in FIELDS}
        field = field_map.get(cmd)
        if not field:
            self._console.print(f"[yellow]Unknown command /{cmd}. Type /help.[/yellow]")
            return

        match field.menu:
            case ToggleMenu() as m:
                if arg in ("on", "off"):
                    m.set_current(self, arg == "on")
                    self._console.print(f"[dim]/{field.name} → {arg}[/dim]")
                else:
                    result = await _run_dialog(
                        radiolist_dialog(
                            title=field.name,
                            text=field.description,
                            values=[("on", "on"), ("off", "off")],
                        )
                    )
                    if result is not None:
                        m.set_current(self, result == "on")
                        self._console.print(f"[dim]/{field.name} → {result}[/dim]")

            case TextInputMenu() as m:
                if arg:
                    val = None if (m.nullable and arg.lower() == "none") else arg
                    m.set_current(self, val)
                    self._console.print(f"[dim]/{field.name} → {val!r}[/dim]")
                else:
                    current = m.get_current(self)
                    result = await _run_dialog(
                        input_dialog(
                            title=field.name,
                            text=field.description,
                            default=current or "",
                        )
                    )
                    if result is not None:
                        val = None if (m.nullable and result.lower() == "none") else result
                        m.set_current(self, val)
                        self._console.print(f"[dim]/{field.name} → {val!r}[/dim]")

            case DialogueMenu() as m:
                if arg:
                    m.set_current(self, arg)
                    self._console.print(f"[dim]/{field.name} → {arg}[/dim]")
                else:
                    choices = await resolve_choices(m.choices, self)
                    result = await _run_dialog(
                        radiolist_dialog(
                            title=field.name,
                            text=field.description,
                            values=[(c, c) for c in choices],
                        )
                    )
                    if result == CUSTOM_PROVIDER_LABEL:
                        url = await _run_dialog(
                            input_dialog(
                                title="Custom provider URL",
                                text="Enter OpenAI-compatible base URL (host:port):",
                            )
                        )
                        if url:
                            m.set_current(self, url)
                            self._console.print(f"[dim]/{field.name} → {url!r}[/dim]")
                    elif result is not None:
                        m.set_current(self, result)
                        self._console.print(f"[dim]/{field.name} → {result}[/dim]")

            case MultiSelectMenu() as m:
                if arg:
                    selected = [s.strip() for s in arg.replace(",", " ").split() if s.strip()]
                    m.set_current(self, selected)
                    self._console.print(f"[dim]/{field.name} → {selected}[/dim]")
                else:
                    choices = await resolve_choices(m.choices, self)
                    current = list(m.get_current(self) or [])
                    result = await _run_dialog(
                        checkboxlist_dialog(
                            title=field.name,
                            text=field.description,
                            values=[(c, c) for c in choices],
                            default_values=current,
                        )
                    )
                    if result is not None:
                        m.set_current(self, result)
                        self._console.print(f"[dim]/{field.name} → {result}[/dim]")

    async def _collect_turn(self, _input_id: str) -> None:
        response_buf = ""
        live: Live | None = None
        last_update = 0.0
        prev_event: Event | None = None

        def start_new_block(event):
            if (
                prev_event is not None
                and not isinstance(prev_event, UserInputEvent)
                and not isinstance(event, type(prev_event))
            ):
                self._console.print("\n")

        async for event in self.api.stream_events():
            match event:
                case UserInputEvent(text=t):
                    pass  # Don't re print user text back to screen
                    # self._console.print(f"[bold]>[/bold] {t}")

                case ThinkingEvent(fragment=f):
                    start_new_block(event)

                    if self._show_thinking:
                        self._console.print(f, style="dim italic", end="")

                case ResponseEvent(fragment=f):
                    start_new_block(event)

                    response_buf += f
                    now = time.monotonic()
                    if live is None:
                        live = Live(
                            Markdown(response_buf),
                            console=self._console,
                            auto_refresh=False,
                        )
                        live.start()
                        last_update = now
                    elif now - last_update >= UPDATE_INTERVAL:
                        live.update(Markdown(response_buf))
                        last_update = now

                case ToolStartEvent(tool_name=name, tool_input=inp):
                    start_new_block(event)

                    if live is not None:
                        live.update(Markdown(response_buf))
                        live.stop()
                        live = None
                        response_buf = ""
                    self._console.print(f"[blue bold]⏺ {name}[/blue bold]({_fmt_tool_input(inp)})")

                case ToolOutputEvent(result=result, output_format=fmt):
                    match fmt:
                        case "all":
                            self._console.print(result, style="cyan dim")
                        case "truncate":
                            if len(result) > MAX_TOOL_OUTPUT:
                                result = result[:MAX_TOOL_OUTPUT] + "… [truncated]"
                            indented = "\n".join(f"  {line}" for line in result.splitlines())
                            self._console.print(indented, style="cyan dim")

                case ToolErrorEvent(tool_name=name, error=err):
                    self._console.print(f"  {name}: {err}", style="red")

                case DoneEvent(error=error):
                    if live is not None:
                        live.update(Markdown(response_buf))
                        live.stop()
                        live = None
                    if error:
                        self._console.print(f"[red]Error: {error}[/red]")
                    break

            prev_event = event

        self._console.print()

    def _show_help(self) -> None:
        lines = ["[bold underline]Commands[/bold underline]"]
        for field in FIELDS:
            menu = field.menu
            if isinstance(menu, ToggleMenu):
                usage = f"/{field.name} on|off"
            elif isinstance(menu, TextInputMenu):
                nullable_hint = "|none" if menu.nullable else ""
                usage = f"/{field.name} <value{nullable_hint}>"
            elif isinstance(menu, MultiSelectMenu):
                usage = f"/{field.name} [tool ...]"
            else:
                usage = f"/{field.name} [value]"
            lines.append(f"  [cyan]/{field.name}[/cyan]  {field.description}  [dim]{usage}[/dim]")
        lines.append("  [cyan]/help[/cyan]  Show this message")
        lines.append("  [cyan]/quit[/cyan]  Exit")
        self._console.print("\n".join(lines))

    def _print_header(self) -> None:
        self._console.rule("[bold]LocalAI[/bold]")
        self._console.print("[dim]Ctrl+C or Ctrl+D to quit · /help for commands[/dim]\n")


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
