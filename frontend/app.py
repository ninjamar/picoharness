from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from rich.markdown import Markdown
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widget import Widget
from textual.widgets import Input, RichLog, Select, SelectionList

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
from frontend.config_io import generate_config, load_config
from frontend.schema import (
    CUSTOM_PROVIDER_LABEL,
    FIELDS,
    DialogueMenu,
    FieldDef,
    MultiSelectMenu,
    TextInputMenu,
    ToggleMenu,
    resolve_choices,
)
from frontend.widgets import InputArea

MAX_TOOL_OUTPUT = 500
SYSTEM_PROMPT_PATH = Path(__file__).parent / "files" / "system_prompt.md"


def _fmt_tool_input(inp: dict | str) -> str:
    if not isinstance(inp, dict):
        return repr(inp)
    items = list(inp.items())
    if len(items) == 1:
        return repr(items[0][1])
    return ", ".join(f"{k}={v!r}" for k, v in items)


class ChatApp(App):
    CSS_PATH = "style.tcss"
    BINDINGS = [
        Binding("ctrl+c", "maybe_quit", show=False, priority=True),
    ]

    def __init__(self, api: BackendAPI, show_think: bool = True) -> None:
        super().__init__()
        self.api = api
        self._show_thinking = show_think
        self._turn_in_progress = False
        # Pending state for command panel interactions
        self._pending_field: FieldDef | None = None
        self._pending_dialogue_menu: DialogueMenu | None = None
        self._pending_dialogue_field: FieldDef | None = None

    @property
    def show_think(self) -> bool:
        return self._show_thinking

    def set_show_think(self, value: bool) -> None:
        self._show_thinking = value

    def compose(self) -> ComposeResult:
        yield RichLog(id="chat-log", markup=True, wrap=True)
        yield RichLog(id="streaming-panel", markup=True, wrap=True, classes="hidden")
        yield Container(id="command-panel", classes="hidden")
        yield InputArea(id="input-area")

    async def on_mount(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("[bold]LocalAI Chat[/bold]")
        log.write("[dim]Ctrl+C to quit · Ctrl+J for newline · /help for commands[/dim]\n")
        self.query_one("#input-area").focus()
        self.run_worker(self._event_loop(), exclusive=True)

    async def _event_loop(self) -> None:
        response_buf = ""
        thinking_buf = ""
        log = self.query_one("#chat-log", RichLog)
        streaming = self.query_one("#streaming-panel", RichLog)

        async for event in self.api.stream_events():
            match event:
                case UserInputEvent(text=t):
                    log.write(f"[bold green]You:[/bold green] {t}\n")
                case ThinkingEvent(fragment=f):
                    if self._show_thinking:
                        thinking_buf += f
                        streaming.remove_class("hidden")
                        streaming.clear()
                        streaming.write(Text(thinking_buf, style="dim italic"))
                case ResponseEvent(fragment=f):
                    if thinking_buf:
                        thinking_buf = ""
                        streaming.clear()
                    response_buf += f
                    streaming.remove_class("hidden")
                    streaming.clear()
                    streaming.write(Markdown(response_buf))
                case ToolStartEvent(tool_name=name, tool_input=inp):
                    if response_buf:
                        log.write(Markdown(response_buf))
                        response_buf = ""
                    streaming.add_class("hidden")
                    streaming.clear()
                    log.write(f"[bold blue]⏺ {name}({_fmt_tool_input(inp)})[/bold blue]")
                case ToolOutputEvent(result=result, output_format=fmt):
                    match fmt:
                        case "all":
                            log.write(Text(result, style="cyan dim"))
                        case "truncate":
                            if len(result) > MAX_TOOL_OUTPUT:
                                result = result[:MAX_TOOL_OUTPUT] + "… [truncated]"
                            indented = "\n".join(f"  {line}" for line in result.splitlines())
                            log.write(Text(indented, style="cyan dim"))
                        case "none":
                            pass
                case ToolErrorEvent(tool_name=name, error=err):
                    log.write(f"[red]  {name} error: {err}[/red]")
                case DoneEvent(error=error):
                    if response_buf:
                        log.write(Markdown(response_buf))
                        response_buf = ""
                    thinking_buf = ""
                    streaming.add_class("hidden")
                    streaming.clear()
                    if error:
                        log.write(f"[red]Error: {error}[/red]")
                    log.write("")
                    self._turn_in_progress = False
                    self._unlock_input()

    # ── Input handling ────────────────────────────────────────────────────────

    def on_input_area_submitted(self, msg: InputArea.Submitted) -> None:
        text = msg.text
        if text.startswith("/"):
            self.run_worker(self._dispatch_command(text), exclusive=False)
        else:
            self.api.feed(str(uuid.uuid4()), text)
            self._turn_in_progress = True
            self._lock_input()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter in a command panel Input widget."""
        field = self._pending_field
        if field is None:
            return
        event.stop()
        value = event.value.strip()
        match field.menu:
            case TextInputMenu() as m:
                val: str | None = None if (m.nullable and value.lower() == "none") else (value or None)
                m.set_current(self, val)
                self._log(f"[dim]{field.name} → {val}[/dim]")
            case DialogueMenu() as m:
                # Provider custom URL path
                if value:
                    m.set_current(self, value)
                    self._log(f"[dim]{field.name} → {value}[/dim]")
        self._pending_field = None
        self._hide_panel()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle selection in the command panel's Select widget."""
        m = self._pending_dialogue_menu
        field = self._pending_dialogue_field
        if m is None or field is None:
            return
        value = event.value
        if value is Select.BLANK:
            return
        event.stop()
        if value == CUSTOM_PROVIDER_LABEL:
            # Sequential swap: replace Select with an Input for the URL
            inp = Input(placeholder="Enter base URL (e.g. localhost:11434)")
            self._pending_field = field
            self._pending_dialogue_menu = None
            self._pending_dialogue_field = None
            self._swap_panel(inp)
            return
        m.set_current(self, value)
        self._log(f"[dim]{field.name} → {value}[/dim]")
        self._pending_dialogue_menu = None
        self._pending_dialogue_field = None
        self._hide_panel()

    def on_key(self, event: events.Key) -> None:
        panel = self.query_one("#command-panel", Container)
        if panel.has_class("hidden"):
            return
        if event.key == "escape":
            self._pending_field = None
            self._pending_dialogue_menu = None
            self._pending_dialogue_field = None
            self._hide_panel()
            event.stop()
        elif event.key == "enter":
            # Confirm SelectionList on Enter
            for sel in panel.query(SelectionList):
                field = self._pending_field
                if field and isinstance(field.menu, MultiSelectMenu):
                    selected = [str(v) for v in sel.selected]
                    field.menu.set_current(self, selected)
                    self._log(f"[dim]{field.name} = {selected}[/dim]")
                    self._pending_field = None
                    self._hide_panel()
                    event.stop()
                    return

    def action_maybe_quit(self) -> None:
        if self._turn_in_progress:
            self.api.cancel_current()
        else:
            self.exit()

    # ── Schema-driven command dispatch ────────────────────────────────────────

    async def _dispatch_command(self, text: str) -> None:
        parts = text[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "help":
            self._show_help()
            return
        if cmd == "quit":
            self.exit()
            return

        field_map: dict[str, FieldDef] = {f.name: f for f in FIELDS}
        field = field_map.get(cmd)
        if not field:
            self._log(f"[yellow]Unknown command /{cmd}. Type /help.[/yellow]")
            return

        await self._execute_field(field, arg)

    async def _execute_field(self, field: FieldDef, arg: str) -> None:
        match field.menu:
            case ToggleMenu() as m:
                if arg not in ("on", "off"):
                    self._log(f"[yellow]Usage: /{field.name} on|off[/yellow]")
                    return
                m.set_current(self, arg == "on")
                self._log(f"[dim]{field.name} = {arg}[/dim]")

            case DialogueMenu() as m:
                if arg:
                    m.set_current(self, arg)
                    self._log(f"[dim]{field.name} → {arg}[/dim]")
                    return
                choices = await resolve_choices(m.choices, self)
                current = m.get_current(self)
                select = Select(
                    [(c, c) for c in choices],
                    value=current if current in choices else Select.BLANK,
                    allow_blank=True,
                )
                select.border_title = f"/{field.name}"
                self._pending_dialogue_menu = m
                self._pending_dialogue_field = field
                self._show_panel(select)

            case TextInputMenu() as m:
                if arg:
                    val: str | None = None if (m.nullable and arg.lower() == "none") else arg
                    m.set_current(self, val)
                    self._log(f"[dim]{field.name} → {val}[/dim]")
                    return
                current_val = m.get_current(self)
                inp = Input(
                    value=str(current_val) if current_val is not None else "",
                    placeholder=field.description,
                )
                inp.border_title = f"/{field.name} — Enter to confirm, Escape to cancel"
                self._pending_field = field
                self._show_panel(inp)

            case MultiSelectMenu() as m:
                choices = await resolve_choices(m.choices, self)
                current = list(m.get_current(self) or [])
                sel = SelectionList(*[(c, c, c in current) for c in choices])
                sel.border_title = f"/{field.name} — Space to toggle, Enter to confirm"
                self._pending_field = field
                self._show_panel(sel)

    # ── Panel helpers ─────────────────────────────────────────────────────────

    def _show_panel(self, widget: Widget) -> None:
        panel = self.query_one("#command-panel", Container)
        panel.remove_children()
        panel.mount(widget)
        panel.remove_class("hidden")
        self.call_after_refresh(widget.focus)

    def _hide_panel(self) -> None:
        panel = self.query_one("#command-panel", Container)
        panel.add_class("hidden")
        panel.remove_children()
        self.query_one("#input-area").focus()

    def _swap_panel(self, widget: Widget) -> None:
        panel = self.query_one("#command-panel", Container)
        panel.remove_children()
        panel.mount(widget)
        self.call_after_refresh(widget.focus)

    def _lock_input(self) -> None:
        self.query_one("#input-area", InputArea).disabled = True

    def _unlock_input(self) -> None:
        inp = self.query_one("#input-area", InputArea)
        inp.disabled = False
        inp.focus()

    def _log(self, msg: str) -> None:
        self.query_one("#chat-log", RichLog).write(msg)

    def _show_help(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("[bold underline]Available commands:[/bold underline]\n")
        for field in FIELDS:
            menu = field.menu
            if isinstance(menu, ToggleMenu):
                usage = f"/{field.name} on|off"
            elif isinstance(menu, TextInputMenu):
                nullable_hint = "|none" if menu.nullable else ""
                usage = f"/{field.name} <value{nullable_hint}>"
            else:
                usage = f"/{field.name}"
            log.write(f"  [cyan]/{field.name}[/cyan]  [italic]{field.description}[/italic]  [dim]{usage}[/dim]")
        log.write("  [cyan]/help[/cyan]  [italic]Show this message[/italic]")
        log.write("  [cyan]/quit[/cyan]  [italic]Exit the application[/italic]\n")

    def run_app(self) -> None:
        """Sync entry point — wraps the API lifecycle around the Textual app."""
        import asyncio

        async def _run() -> None:
            async with self.api:
                await self.run_async()

        asyncio.run(_run())


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
    ChatApp(api=api, show_think=cfg.show_think).run_app()
