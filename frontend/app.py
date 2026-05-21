from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static

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
    MultiSelectMenu,
    TextInputMenu,
    ToggleMenu,
    resolve_choices,
)
from frontend.widgets import InputArea, InputOverlay

MAX_TOOL_OUTPUT = 500
SYSTEM_PROMPT_PATH = Path(__file__).parent / "files" / "system_prompt.md"

_ALL_COMMANDS: list[tuple[str, str]] = [(f.name, f.description) for f in FIELDS] + [
    ("help", "Show available commands"),
    ("quit", "Exit the application"),
]


def _fmt_tool_input(inp: dict | str) -> str:
    if not isinstance(inp, dict):
        return repr(inp)
    items = list(inp.items())
    if len(items) == 1:
        return repr(items[0][1])
    return ", ".join(f"{k}={v!r}" for k, v in items)


class ChatApp(App):
    ansi_color = True
    theme = "nord"
    CSS_PATH = "style.tcss"
    BINDINGS = [
        Binding("ctrl+c", "maybe_quit", show=False, priority=True),
    ]

    def __init__(self, api: BackendAPI, show_think: bool = True) -> None:
        super().__init__(ansi_color=True)
        self.api = api
        self._show_thinking = show_think
        self._turn_in_progress = False
        self._current_md: Markdown | None = None
        self._response_buf: str = ""
        self._thinking_md: Static | None = None
        self._thinking_buf: str = ""
        self._last_event_type: str | None = None
        self._history: list[str] = []

    @property
    def show_think(self) -> bool:
        return self._show_thinking

    def set_show_think(self, value: bool) -> None:
        self._show_thinking = value

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chat-area"):
            pass
        yield InputOverlay(id="input-overlay")
        yield InputArea(id="input-area", compact=True, placeholder="Type here")

    async def on_mount(self) -> None:
        chat = self.query_one("#chat-area", VerticalScroll)
        await chat.mount(
            Static("[bold]LocalAI[/bold]  [dim]Ctrl+C to quit · Ctrl+J for newline · /help for commands[/dim]\n")
        )
        self.query_one("#input-area").focus()
        self.run_worker(self._event_loop(), exclusive=True)

    # ── Streaming event loop ──────────────────────────────────────────────────

    async def _event_loop(self) -> None:
        chat = self.query_one("#chat-area", VerticalScroll)

        async def mount(widget) -> None:
            await chat.mount(widget)
            chat.scroll_end(animate=False)

        def reset_response() -> None:
            self._current_md = None
            self._response_buf = ""
            self._last_event_type = None

        def reset_thinking() -> None:
            self._thinking_md = None
            self._thinking_buf = ""
            self._last_event_type = None

        async for event in self.api.stream_events():
            match event:
                case UserInputEvent(text=t):
                    reset_response()
                    reset_thinking()
                    await mount(Static(f"[bold]>[/bold] {t}", classes="user-msg"))

                case ThinkingEvent(fragment=f):
                    if not self._show_thinking:
                        continue
                    if self._last_event_type != "thinking":
                        self._thinking_md = None
                        self._thinking_buf = ""
                    self._last_event_type = "thinking"
                    self._thinking_buf += f
                    if self._thinking_md is None:
                        self._thinking_md = Static(self._thinking_buf, classes="thinking")
                        await mount(self._thinking_md)
                    else:
                        self._thinking_md.update(self._thinking_buf)
                    chat.scroll_end(animate=False)

                case ResponseEvent(fragment=f):
                    if self._last_event_type != "response":
                        self._current_md = None
                        self._response_buf = ""
                    self._last_event_type = "response"
                    self._response_buf += f
                    if self._current_md is None:
                        self._current_md = Markdown(self._response_buf, classes="response")
                        await mount(self._current_md)
                    else:
                        await self._current_md.update(self._response_buf)
                    chat.scroll_end(animate=False)

                case ToolStartEvent(tool_name=name, tool_input=inp):
                    # Reset so the next ResponseEvent mounts after this tool's output
                    reset_response()
                    await mount(Static(f"[bold]⏺ {name}[/bold]({_fmt_tool_input(inp)})", classes="tool-call"))

                case ToolOutputEvent(result=result, output_format=fmt):
                    match fmt:
                        case "all":
                            text = result
                        case "truncate":
                            if len(result) > MAX_TOOL_OUTPUT:
                                result = result[:MAX_TOOL_OUTPUT] + "… [truncated]"
                            text = "\n".join(f"  {line}" for line in result.splitlines())
                        case _:
                            continue
                    await mount(Static(text, classes="tool-output"))

                case ToolErrorEvent(tool_name=name, error=err):
                    await mount(Static(f"  {name}: {err}", classes="tool-error"))

                case DoneEvent(error=error):
                    reset_response()
                    reset_thinking()
                    if error:
                        await mount(Static(f"[red]Error: {error}[/red]"))
                    self._turn_in_progress = False
                    self._unlock_input()

    # ── Input handling ────────────────────────────────────────────────────────

    def on_input_area_submitted(self, msg: InputArea.Submitted) -> None:
        text = msg.text
        if text.startswith("/"):
            self.run_worker(self._dispatch_command(text), exclusive=False)
        else:
            self._history.append(text)
            self.query_one("#input-area", InputArea).load_history(self._history)
            self.api.feed(str(uuid.uuid4()), text)
            self._turn_in_progress = True
            self._lock_input()

    def on_input_area_text_changed(self, msg: InputArea.TextChanged) -> None:
        text = msg.text
        overlay = self.query_one(InputOverlay)
        if text.startswith("/"):
            after_slash = text[1:]
            if " " in after_slash:
                overlay.hide()
                return
            query = after_slash.lower() if after_slash.strip() else ""
            matches = [(f"/{cmd}", desc) for cmd, desc in _ALL_COMMANDS if cmd.startswith(query)]
            overlay.show_completions(matches)
        else:
            overlay.hide()

    def action_maybe_quit(self) -> None:
        if self._turn_in_progress:
            self.api.cancel_current()
        else:
            self.exit()

    # ── Schema-driven command dispatch (fully inline, no panels) ──────────────

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

        field_map = {f.name: f for f in FIELDS}
        field = field_map.get(cmd)
        if not field:
            self._log(f"[yellow]Unknown command /{cmd}. Type /help.[/yellow]")
            return

        match field.menu:
            case ToggleMenu() as m:
                if arg in ("on", "off"):
                    m.set_current(self, arg == "on")
                    self._log(f"[dim]/{field.name} → {arg}[/dim]")
                    return
                current = "on" if m.get_current(self) else "off"
                overlay = self.query_one(InputOverlay)
                await overlay.show_command(
                    "choice",
                    ["on", "off"],
                    current,
                    False,
                    lambda v: m.set_current(self, v == "on"),
                )

            case TextInputMenu() as m:
                if not arg:
                    current = m.get_current(self)
                    overlay = self.query_one(InputOverlay)
                    await overlay.show_command(
                        "text",
                        [],
                        current,
                        m.nullable,
                        lambda v: m.set_current(self, None if (m.nullable and v and v.lower() == "none") else v),
                    )
                    return
                val = None if (m.nullable and arg.lower() == "none") else arg
                m.set_current(self, val)
                self._log(f"[dim]/{field.name} → {val!r}[/dim]")

            case DialogueMenu() as m:
                if arg:
                    m.set_current(self, arg)
                    self._log(f"[dim]/{field.name} → {arg}[/dim]")
                    return
                choices = await resolve_choices(m.choices, self)
                current = m.get_current(self)
                overlay = self.query_one(InputOverlay)

                if field.name == "provider":

                    def provider_callback(v: str) -> None:
                        if v == CUSTOM_PROVIDER_LABEL:
                            self.run_worker(
                                self._show_custom_provider_input(overlay, m),
                                exclusive=False,
                            )
                        else:
                            m.set_current(self, v)

                    await overlay.show_command("choice", choices, current, False, provider_callback)
                else:
                    await overlay.show_command(
                        "choice",
                        choices,
                        current,
                        False,
                        lambda v: m.set_current(self, v),
                    )

            case MultiSelectMenu() as m:
                if arg:
                    selected = [s.strip() for s in arg.replace(",", " ").split() if s.strip()]
                    m.set_current(self, selected)
                    self._log(f"[dim]/{field.name} → {selected}[/dim]")
                    return
                choices = await resolve_choices(m.choices, self)
                current = list(m.get_current(self) or [])
                overlay = self.query_one(InputOverlay)
                await overlay.show_command(
                    "multiselect",
                    choices,
                    current,
                    False,
                    lambda v: m.set_current(self, v),
                )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _show_custom_provider_input(self, overlay: InputOverlay, menu: DialogueMenu) -> None:
        """Show text input for custom provider URL after user selects 'Custom...'."""
        current = menu.get_current(self)
        await overlay.show_command(
            "text",
            [],
            current,
            False,
            lambda url: menu.set_current(self, url),
        )

    def _lock_input(self) -> None:
        self.query_one("#input-area", InputArea).disabled = True

    def _unlock_input(self) -> None:
        inp = self.query_one("#input-area", InputArea)
        inp.disabled = False
        inp.focus()

    def _log(self, msg: str) -> None:
        chat = self.query_one("#chat-area", VerticalScroll)
        widget = Static(msg)
        self.run_worker(chat.mount(widget), exclusive=False)
        chat.scroll_end(animate=False)

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
        self._log("\n".join(lines))

    def run_app(self) -> None:
        async def _run() -> None:
            async with self.api:
                # The issue with inline mode is that there are white borders around
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
