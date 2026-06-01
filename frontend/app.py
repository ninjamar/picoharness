from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
import uuid
from pathlib import Path

from pylatexenc.latex2text import LatexNodes2Text

_latex_converter = LatexNodes2Text()


def _render_latex(text: str) -> str:
    def replace_block(m: re.Match) -> str:
        try:
            return f"\n```\n{_latex_converter.latex_to_text(m.group(1))}\n```\n"
        except Exception:
            return m.group(0)

    def replace_inline(m: re.Match) -> str:
        try:
            return f"`{_latex_converter.latex_to_text(m.group(1))}`"
        except Exception:
            return m.group(0)

    text = re.sub(r'\$\$(.*?)\$\$', replace_block, text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', replace_inline, text)
    return text

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
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
    FIELDS_SHOW_UI,
    DialogueMenu,
    MultiSelectMenu,
    TextInputMenu,
    ToggleMenu,
    resolve_choices,
)
from frontend.services_cmd import services_main
from frontend.widgets import CommandPanel, CompletionMenu, InputArea

MAX_TOOL_OUTPUT = 500
SYSTEM_PROMPT_PATH = Path(__file__).parent / "files" / "system_prompt.md"
UPDATE_INTERVAL = 0.05  # seconds between re-renders


def _fmt_tool_input(inp: dict | str) -> str:
    if not isinstance(inp, dict):
        return repr(inp)
    items = list(inp.items())
    if len(items) == 1:
        return repr(items[0][1])
    return ", ".join(f"{k}={v!r}" for k, v in items)


class InputOverlay(Container):
    """Container for CompletionMenu and CommandPanel."""

    pass


class ChatApp(App):
    """Textual app for PicoHarness chat interface with inline config panels."""

    TITLE = "PicoHarness"
    CSS_PATH = "style.tcss"
    theme = "nord" # type: ignore

    BINDINGS = []

    def __init__(self, api: BackendAPI, show_think: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.api = api
        self.show_think = show_think

    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield VerticalScroll(id="chat-area")

        with InputOverlay(id="input-overlay"):
            yield CompletionMenu(id="completion")
            yield CommandPanel(id="command-panel")

        with Horizontal(id="input-line"):
            yield Static(">", id="input-prompt", expand=False)
            yield InputArea(id="input-area")

    def on_mount(self) -> None:
        """Initialize the app."""
        self.title = "PicoHarness"
        # Initialize streaming state
        self._current_md: Markdown | None = None
        self._response_buf: str = ""
        self._thinking_md: Static | None = None
        self._thinking_buf: str = ""
        self._last_event_type: str | None = None
        self._generating = False  # Track if LLM is generating
        self._last_ctrlc_time: float = 0.0

        # Populate the completion menu with all commands
        completion_menu = self.query_one("#completion", CompletionMenu)
        all_commands = [(f"/{f.name}", f.description) for f in FIELDS_SHOW_UI]
        all_commands += [("/help", "Show help"), ("/quit", "Exit")]
        completion_menu.set_commands(all_commands)

        input_area = self.query_one("#input-area", InputArea)
        input_area.focus()
        self._print_header()
        self._start_event_loop_worker()

    def set_show_think(self, value: bool) -> None:
        self.show_think = value

    def action_help_quit(self) -> None:
        # Overrides a builtin...
        """Handle Ctrl+C with priority: interrupt → clear → double-press exit."""
        input_area = self.query_one("#input-area", InputArea)
        if self._generating:
            self.api.cancel_current()
            self._print_system_feedback("Cancelled (Ctrl+C again to exit)")
            self._last_ctrlc_time = 0.0
        elif input_area.text.strip():
            input_area.text = ""
            input_area._history_index = None
            input_area._saved_input = ""
            self._print_system_feedback("Input cleared (Ctrl+C again to exit)")
            self._last_ctrlc_time = 0.0
        else:
            now = time.time()
            if now - self._last_ctrlc_time <= 2.0:
                self._print_system_feedback("Exiting…")
                self.exit()
            else:
                self._last_ctrlc_time = now
                self._print_system_feedback("Press Ctrl+C again to exit")

    def on_input_area_submitted(self, msg: InputArea.Submitted) -> None:
        """Handle text submission from input area."""
        text = msg.text.strip()
        if not text:
            return

        if text.startswith("/"):
            asyncio.create_task(self._dispatch_command(text))
        else:
            input_id = str(uuid.uuid4())
            self.api.feed(input_id, text)
            self._mount_user_message(text)

    def on_text_area_changed(self, event) -> None:
        """Handle text changes in the input area."""
        if event.text_area.id != "input-area":
            return

        raw = event.text_area.text
        completion_menu = self.query_one("#completion", CompletionMenu)

        # Hide completion if no slash or if there's a space (argument mode)
        if not raw.lstrip().startswith("/") or " " in raw:
            completion_menu.hide()
            return

        # Extract the prefix (everything after / and before space)
        prefix = raw.lstrip()[1:].lower()
        completion_menu.filter(prefix)

    def on_input_area_completion_requested(self) -> None:
        """Handle completion selection via Tab or Enter."""
        completion_menu = self.query_one("#completion", CompletionMenu)
        input_area = self.query_one("#input-area", InputArea)

        selected = completion_menu.get_selected()
        if selected:
            input_area.text = selected + " "
            completion_menu.hide()

    def on_command_panel_dismissed(self, msg: CommandPanel.Dismissed) -> None:
        """Handle config panel dismissal and apply the value."""
        command_panel = self.query_one("#command-panel", CommandPanel)
        input_area = self.query_one("#input-area", InputArea)

        if not hasattr(self, "_pending_field"):
            command_panel.hide()
            input_area.focus()
            return

        field = self._pending_field

        # If we're in custom provider mode, this is the URL input
        if hasattr(self, "_custom_provider_mode") and self._custom_provider_mode:
            self._custom_provider_mode = False
            command_panel.hide()
            if msg.value:
                field.menu.set_current(self, msg.value)
                self._print_dim_feedback(f"/{field.name} → {msg.value!r}")
            del self._pending_field
            input_area.focus()
            return

        # Check if this is a custom provider selection (show URL input instead)
        if field.name == "provider" and msg.value == CUSTOM_PROVIDER_LABEL:
            self._custom_provider_mode = True
            asyncio.create_task(self._show_custom_provider_input())
            input_area.focus()
            return

        # Normal case: apply the value
        command_panel.hide()
        if msg.value is not None:
            field.menu.set_current(self, msg.value)
            self._print_dim_feedback(f"/{field.name} → {msg.value!r}")
        del self._pending_field
        input_area.focus()

    async def _dispatch_command(self, raw: str) -> None:
        """Dispatch a slash command."""
        parts = raw[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "quit":
            self.exit()
            return

        if cmd == "help":
            self._show_help()
            return

        field_map = {f.name: f for f in FIELDS_SHOW_UI}
        field = field_map.get(cmd)
        if not field:
            self._print_warning(f"Unknown command /{cmd}. Type /help.")
            return

        # If inline arg provided, apply directly without a panel
        match field.menu:
            case ToggleMenu() as m:
                if arg in ("on", "off"):
                    m.set_current(self, arg == "on")
                    self._print_dim_feedback(f"/{field.name} → {arg}")
                else:
                    # Show panel
                    await self._show_command_panel(field)

            case TextInputMenu() as m:
                if arg:
                    val = None if (m.nullable and arg.lower() == "none") else arg
                    m.set_current(self, val)
                    self._print_dim_feedback(f"/{field.name} → {val!r}")
                else:
                    # Show panel
                    await self._show_command_panel(field)

            case DialogueMenu() as m:
                if arg:
                    m.set_current(self, arg)
                    self._print_dim_feedback(f"/{field.name} → {arg}")
                else:
                    # Show panel
                    await self._show_command_panel(field)

            case MultiSelectMenu() as m:
                if arg:
                    selected = [s.strip() for s in arg.replace(",", " ").split() if s.strip()]
                    m.set_current(self, selected)
                    self._print_dim_feedback(f"/{field.name} → {selected}")
                else:
                    # Show panel
                    await self._show_command_panel(field)

    async def _show_command_panel(self, field) -> None:
        """Show the command panel for a field."""
        command_panel = self.query_one("#command-panel", CommandPanel)
        self._pending_field = field

        match field.menu:
            case ToggleMenu():
                current = field.menu.get_current(self)
                await command_panel.show_toggle(current, field.description)

            case TextInputMenu() as m:
                current = m.get_current(self)
                await command_panel.show_text_input(current, m.nullable, field.description)

            case DialogueMenu() as m:
                choices = await resolve_choices(m.choices, self)
                current = m.get_current(self)
                await command_panel.show_choice(choices, current, field.description)

            case MultiSelectMenu() as m:
                choices = await resolve_choices(m.choices, self)
                current = m.get_current(self) or []
                await command_panel.show_multiselect(choices, current, field.description)

    async def _show_custom_provider_input(self) -> None:
        """Show text input for custom provider URL."""
        command_panel = self.query_one("#command-panel", CommandPanel)
        await command_panel.show_text_input(
            current=None, nullable=False, label="Enter provider URL (e.g., http://localhost:8000)"
        )

    def _start_event_loop_worker(self) -> None:
        """Start background worker for streaming responses."""
        self.run_worker(self._event_loop(), exclusive=True)

    async def _event_loop(self) -> None:
        """Background task to consume and render streaming events."""
        # This event loop runs forever
        async for event in self.api.stream_events():
            match event:
                case UserInputEvent():
                    pass  # User message already mounted

                case ThinkingEvent(fragment=fragment):
                    if not self._generating:
                        self._generating = True
                        self.sub_title = "generating…"
                    if self.show_think:
                        self._thinking_buf += fragment
                        if self._thinking_md is None:
                            # First thinking fragment - mount widget
                            chat_area = self.query_one("#chat-area", VerticalScroll)
                            self._thinking_md = Static(self._thinking_buf, classes="thinking")
                            chat_area.mount(self._thinking_md)
                            chat_area.scroll_end(animate=False)
                        else:
                            # Update existing thinking widget
                            self._thinking_md.update(self._thinking_buf)
                            chat_area = self.query_one("#chat-area", VerticalScroll)
                            chat_area.scroll_end(animate=False)

                case ResponseEvent(fragment=fragment):
                    if not self._generating:
                        self._generating = True
                        self.sub_title = "generating…"
                    # Reset thinking if transitioning from thinking to response
                    if self._last_event_type == "thinking":
                        self._thinking_buf = ""
                        self._thinking_md = None

                    self._response_buf += fragment
                    if self._current_md is None:
                        # First response fragment - mount widget
                        chat_area = self.query_one("#chat-area", VerticalScroll)
                        self._current_md = Markdown(self._response_buf, classes="response")
                        await chat_area.mount(self._current_md)
                        chat_area.scroll_end(animate=False)
                    else:
                        # Update existing response widget
                        self._current_md.update(self._response_buf)
                        chat_area = self.query_one("#chat-area", VerticalScroll)
                        chat_area.scroll_end(animate=False)
                    self._last_event_type = "response"

                case ToolStartEvent(tool_name=name, tool_input=inp):
                    # Finalize current response/thinking
                    self._reset_response()
                    self._reset_thinking()
                    self._mount_tool_call(name, inp)

                case ToolOutputEvent(result=result, output_format=fmt):
                    self._mount_tool_output(result, fmt)

                case ToolErrorEvent(tool_name=name, error=error):
                    self._mount_tool_error(name, error)

                case DoneEvent(error=error):
                    self._generating = False
                    self.sub_title = ""
                    # Finalize any remaining response/thinking
                    self._reset_response()
                    self._reset_thinking()
                    if error:
                        self._mount_error(error)

    def _reset_response(self) -> None:
        """Finalize and clear response accumulator."""
        if self._current_md is not None:
            self._current_md.update(_render_latex(self._response_buf))
            self._current_md = None
        self._response_buf = ""

    def _reset_thinking(self) -> None:
        """Finalize and clear thinking accumulator."""
        if self._thinking_md is not None:
            self._thinking_md = None
        self._thinking_buf = ""

    def _mount_user_message(self, text: str) -> None:
        """Mount a user message in the chat area."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        widget = Static(text, classes="user-msg")
        chat_area.mount(widget)
        chat_area.scroll_end(animate=False)

    def _mount_tool_call(self, name: str, inp: dict | str) -> None:
        """Mount a tool call notification."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        text = f"⏺ {name}({_fmt_tool_input(inp)})"
        widget = Static(text, classes="tool-call")
        chat_area.mount(widget)
        chat_area.scroll_end(animate=False)

    def _mount_tool_output(self, result: str, fmt: str) -> None:
        """Mount tool output."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        match fmt:
            case "all":
                text = result
            case "truncate":
                if len(result) > MAX_TOOL_OUTPUT:
                    result = result[:MAX_TOOL_OUTPUT] + "… [truncated]"
                text = "\n".join(f"  {line}" for line in result.splitlines())
            case _:
                return  # "none" format, don't display
        widget = Static(text, classes="tool-output")
        chat_area.mount(widget)
        chat_area.scroll_end(animate=False)

    def _mount_tool_error(self, name: str, error: str) -> None:
        """Mount a tool error message."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        text = f"{name}: {error}"
        widget = Static(text, classes="tool-error")
        chat_area.mount(widget)
        chat_area.scroll_end(animate=False)

    def _mount_error(self, error: str) -> None:
        """Mount an error message."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        text = f"Error: {error}"
        widget = Static(text, classes="tool-error")
        chat_area.mount(widget)
        chat_area.scroll_end(animate=False)

    def _mount_response(self, text: str) -> None:
        """Mount a response (markdown) to the chat area."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        widget = Markdown(_render_latex(text), classes="response")
        chat_area.mount(widget)
        chat_area.scroll_end(animate=False)

    def _print_header(self) -> None:
        """Mount a header message."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        header = Static("[bold]PicoHarness[/bold]", id="header")
        chat_area.mount(header)
        hint = Static("[dim]Ctrl+C to interrupt/exit · /help for commands[/dim]", id="hint")
        chat_area.mount(hint)

    def _show_help(self) -> None:
        """Display help message."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        lines = ["[bold underline]Commands[/bold underline]"]
        for field in FIELDS_SHOW_UI:
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

        help_text = "\n".join(lines)
        widget = Static(help_text)
        chat_area.mount(widget)
        chat_area.scroll_end(animate=False)

    def _print_warning(self, text: str) -> None:
        """Print a warning message."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        widget = Static(f"[yellow]{text}[/yellow]")
        chat_area.mount(widget)
        chat_area.scroll_end(animate=False)

    def _print_dim_feedback(self, text: str) -> None:
        """Print dim feedback about a config change."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        widget = Static(f"[dim]{text}[/dim]")
        chat_area.mount(widget)
        chat_area.scroll_end(animate=False)

    def _print_system_feedback(self, text: str) -> None:
        """Print dim feedback about a config change."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        widget = Static(f"[dim orange]{text}[/dim orange]")
        chat_area.mount(widget)
        chat_area.scroll_end(animate=False)


def cli() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "services":
        services_main(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description="PicoHarness TUI")
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
    tool_names = cfg.tools if cfg.tools else list(tool_name_map.keys())
    for name in tool_names:
        if name not in tool_name_map:
            raise SystemExit(f"Unknown tool '{name}'. Valid: {list(tool_name_map.keys())}")
        tools.append(tool_name_map[name])

    provider = (
        OllamaProvider()
        if cfg.provider == "ollama"
        else OpenAICompatibleProvider(base_url=f"http://{cfg.provider}/v1", api_key=cfg.api_key or "")
    )

    system_prompt = None
    prompt_path = Path(cfg.system_prompt_path) if cfg.system_prompt_path else SYSTEM_PROMPT_PATH
    if prompt_path.exists():
        system_prompt = prompt_path.read_text()

    api = BackendAPI(
        provider=provider,
        model=cfg.model,
        think=cfg.think,
        context_length=cfg.context_length,
        tool_classes=tools,
        system_prompt=system_prompt,
        system_prompt_path=cfg.system_prompt_path,
        api_key=cfg.api_key or "",
        searxng_url=cfg.searxng_url,
        jina_reader_url=cfg.jina_reader_url,
    )

    async def run_app():
        async with api:
            app = ChatApp(api=api, show_think=cfg.show_think)
            await app.run_async()

    asyncio.run(run_app())
