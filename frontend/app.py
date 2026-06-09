from __future__ import annotations

import asyncio
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Literal, Optional

import typer
from pylatexenc.latex2text import LatexNodes2Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Markdown, Static

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
from backend.sessions import SessionManager
from frontend.config_io import generate_config, load_config
from frontend.doctor_cmd import doctor_main
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
SESSION_PICKER_PREVIEW_LENGTH = 60

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

    text = re.sub(r"\$\$(.*?)\$\$", replace_block, text, flags=re.DOTALL)
    text = re.sub(r"\$(.*?)\$", replace_inline, text)
    return text


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
    theme = "nord"

    BINDINGS = []

    def __init__(
        self,
        api: BackendAPI,
        show_think: bool = True,
        show_picker: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api = api
        self.show_think = show_think
        self.show_picker = show_picker
        self.session_name = ""

    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield VerticalScroll(id="chat-area")

        with InputOverlay(id="input-overlay"):
            yield CompletionMenu(id="completion")
            yield CommandPanel(id="command-panel")

        with Container(id="input-section"):
            yield Static("", id="status-bar")
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
        self._auto_scroll: bool = True  # Auto-scroll active when True
        self._response_dirty: bool = False
        self._thinking_dirty: bool = False
        self._last_response_buf: str = ""
        self._last_thinking_buf: str = ""
        self._working_dots: int = 0  # Cycles through 0, 1, 2, 3

        # Populate the completion menu with all commands
        completion_menu = self.query_one("#completion", CompletionMenu)
        all_commands = [(f"/{f.name}", f.description) for f in FIELDS_SHOW_UI]
        all_commands += [
            ("/name", "Rename the current session"),
            ("/help", "Show help"),
            ("/quit", "Exit"),
        ]
        completion_menu.set_commands(all_commands)

        input_area = self.query_one("#input-area", InputArea)
        input_area.focus()

        # Register scroll position watcher
        chat_area = self.query_one("#chat-area", VerticalScroll)
        self.watch(chat_area, "scroll_y", self._on_chat_area_scroll_y_changed)

        # Update status bar immediately and on interval
        self._update_status_bar()
        self.set_interval(1.0, self._update_status_bar)
        self.set_interval(UPDATE_INTERVAL, self._flush_pending_updates)

        self._print_header()

        if self.show_picker:

            async def show_session_picker():
                command_panel = self.query_one("#command-panel", CommandPanel)
                sessions = await asyncio.to_thread(self.api.session_manager.list_sessions)

                self._session_pick_map: dict[str, str | None] = {"+ New Chat": None}
                for s in sessions:
                    preview = (
                        s.first_user_message[:SESSION_PICKER_PREVIEW_LENGTH] if s.first_user_message else "(empty)"
                    )
                    label = f"{s.name}  —  {preview}"
                    self._session_pick_map[label] = s.name

                self._pending_session_pick = True
                await command_panel.show_choice(list(self._session_pick_map.keys()), None, "Select session")

            self.run_worker(show_session_picker())
        else:

            async def start_new():
                self.api.start_session()
                self.session_name = ""
                self._start_event_loop_worker()

            self.run_worker(start_new())

    def _on_session_selected(self, session_name: str | None) -> None:
        """Handle session selection."""

        async def load_and_start():
            if session_name:
                events = await self.api.resume_session(session_name)
                self.session_name = session_name
                await self._replay_session(events)
                await asyncio.sleep(0.1)
                self.query_one("#chat-area", VerticalScroll).scroll_end(animate=False)
            else:
                self.api.start_session()
                self.session_name = ""
            self._start_event_loop_worker()

        self.run_worker(load_and_start())

    def set_show_think(self, value: bool) -> None:
        self.show_think = value

    def exit(self, *args, **kwargs):
        super().exit()
        self.api.finalize()

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

        # Handle session picker
        if getattr(self, "_pending_session_pick", False):
            self._pending_session_pick = False
            command_panel.hide()
            session_name = self._session_pick_map.get(msg.value) if msg.value else None
            self._on_session_selected(session_name)
            input_area.focus()
            return

        # Handle session rename
        if hasattr(self, "_pending_rename") and self._pending_rename:
            self._pending_rename = False
            command_panel.hide()
            if msg.value:
                try:
                    self.api.rename_session(self.session_name, msg.value)
                    self.session_name = msg.value
                    self._print_dim_feedback(f"Session renamed to {msg.value!r}")
                except FileExistsError:
                    self._print_warning(f"Session '{msg.value}' already exists")
            input_area.focus()
            return

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

        if cmd == "name":
            if arg:
                try:
                    self.api.rename_session(self.session_name, arg)
                    self.session_name = arg
                    self._print_dim_feedback(f"Session renamed to {arg!r}")
                except FileExistsError:
                    self._print_warning(f"Session '{arg}' already exists")
            else:
                await self._show_name_input()
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

    async def _show_name_input(self) -> None:
        """Show text input for renaming the session."""
        command_panel = self.query_one("#command-panel", CommandPanel)
        self._pending_rename = True
        await command_panel.show_text_input(current=self.session_name, nullable=False, label="Enter new session name")

    def _start_event_loop_worker(self) -> None:
        """Start background worker for streaming responses."""
        self.run_worker(self._event_loop(), exclusive=True)

    async def _handle_event(self, event: Event, is_replay: bool = False) -> None:
        try:
            match event:
                case UserInputEvent(text=text):
                    if is_replay:
                        self._mount_user_message(text)

                case ThinkingEvent(fragment=fragment):
                    # if not self._generating:
                    if not is_replay:
                        self._generating = True
                        self._update_status_bar()
                    if self.show_think:
                        self._thinking_buf += fragment
                        if self._thinking_md is None:
                            # First thinking fragment - mount widget
                            chat_area = self.query_one("#chat-area", VerticalScroll)
                            self._thinking_md = Static(self._thinking_buf, classes="thinking")
                            await chat_area.mount(self._thinking_md)
                            self._scroll_to_bottom_if_following()
                        else:
                            self._thinking_dirty = True

                case ResponseEvent(fragment=fragment):
                    # if not self._generating:
                    if not is_replay:
                        self._generating = True
                        self._update_status_bar()
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
                        self._scroll_to_bottom_if_following()
                    else:
                        self._response_dirty = True
                    self._last_event_type = "response"

                case ToolStartEvent(tool_name=name, tool_input=inp):
                    if not is_replay:
                        self._update_status_bar()

                    self._generating = True
                    # Finalize current response/thinking
                    await self._reset_response()
                    self._reset_thinking()
                    self._mount_tool_call(name, inp)

                case ToolOutputEvent(result=result, output_format=fmt):
                    self._mount_tool_output(result, fmt)

                case ToolErrorEvent(tool_name=name, error=error):
                    self._mount_tool_error(name, error)

                case DoneEvent(error=error):
                    await self._finish_generating()
                    if error:
                        self._mount_error(error)
        except Exception as e:
            self._mount_error(f"Internal error: {e}")

    async def _event_loop(self) -> None:
        """Background task to consume and render streaming events."""
        try:
            async for event in self.api.stream_events():
                await self._handle_event(event)
        except Exception as e:
            if self._generating:
                self._mount_error(f"Stream error: {e}")
        finally:
            await self._finish_generating()

    def _flush_pending_updates(self) -> None:
        """Flush throttled widget updates at UPDATE_INTERVAL cadence."""
        if self._response_dirty and self._current_md is not None:
            if self._response_buf != self._last_response_buf:
                self._current_md.update(self._response_buf)
                self._last_response_buf = self._response_buf
                self._scroll_to_bottom_if_following()
            self._response_dirty = False
        if self._thinking_dirty and self._thinking_md is not None:
            if self._thinking_buf != self._last_thinking_buf:
                self._thinking_md.update(self._thinking_buf)
                self._last_thinking_buf = self._thinking_buf
                self._scroll_to_bottom_if_following()
            self._thinking_dirty = False

    async def _finish_generating(self) -> None:
        """Reset all generation state. Safe to call even if not currently generating."""
        if not self._generating:
            return
        self._generating = False
        self.sub_title = ""
        self._update_status_bar()
        await self._reset_response()
        self._reset_thinking()
        try:
            self.query_one("#input-area", InputArea).focus()
        except Exception:
            pass

    async def _reset_response(self) -> None:
        """Finalize and clear response accumulator."""
        if self._current_md is not None:
            rendered = await asyncio.to_thread(_render_latex, self._response_buf)
            self._current_md.update(rendered)
            self._current_md = None
        self._response_buf = ""
        self._response_dirty = False

    def _reset_thinking(self) -> None:
        """Finalize and clear thinking accumulator."""
        if self._thinking_md is not None:
            self._thinking_md = None
        self._thinking_buf = ""
        self._thinking_dirty = False

    def _on_chat_area_scroll_y_changed(self, scroll_y: float) -> None:
        """Update auto-scroll flag based on scroll position."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        self._auto_scroll = (chat_area.max_scroll_y - scroll_y) <= 3

    def _scroll_to_bottom_if_following(self) -> None:
        """Scroll to bottom only when auto-scroll is active."""
        if self._auto_scroll:
            self.query_one("#chat-area", VerticalScroll).scroll_end(animate=False)

    def _update_status_bar(self) -> None:
        """Update the status bar with context window and generation status."""
        trimmed, max_ctx, total_size = self.api.context_window
        status = f"total: {total_size} | context window: {trimmed}/{max_ctx}"
        if self.session_name:
            status += f" | session: {self.session_name}"
        if self._generating:
            dots = "." * self._working_dots
            status += f" | working{dots}"
            self._working_dots = (self._working_dots + 1) % 4
        self.query_one("#status-bar", Static).update(status)

    def _mount_user_message(self, text: str) -> None:
        """Mount a user message in the chat area."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        widget = Static(text, classes="user-msg")
        chat_area.mount(widget)
        self._auto_scroll = True
        chat_area.scroll_end(animate=False)

    def _mount_tool_call(self, name: str, inp: dict | str) -> None:
        """Mount a tool call notification."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        text = f"⏺ {name}({_fmt_tool_input(inp)})"
        widget = Static(text, classes="tool-call")
        chat_area.mount(widget)
        self._scroll_to_bottom_if_following()

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
        self._scroll_to_bottom_if_following()

    def _mount_tool_error(self, name: str, error: str) -> None:
        """Mount a tool error message."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        text = f"{name}: {error}"
        widget = Static(text, classes="tool-error")
        chat_area.mount(widget)
        self._scroll_to_bottom_if_following()

    def _mount_error(self, error: str) -> None:
        """Mount an error message."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        text = f"Error: {error}"
        widget = Static(text, classes="tool-error")
        chat_area.mount(widget)
        self._scroll_to_bottom_if_following()

    def _mount_response(self, text: str) -> None:
        """Mount a response (markdown) to the chat area."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        widget = Markdown(_render_latex(text), classes="response")
        chat_area.mount(widget)
        self._scroll_to_bottom_if_following()

    def _print_header(self) -> None:
        """Mount a header message."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        header = Static("[bold]PicoHarness[/bold]", id="header")
        chat_area.mount(header)
        hint = Static("[dim]Ctrl+C to interrupt/exit · /help for commands[/dim]", id="hint")
        chat_area.mount(hint)

    async def _replay_session(self, events: list[Event]) -> None:
        """Replay session events into the chat area."""
        for i, event in enumerate(events):
            await self._handle_event(event, is_replay=True)
            if i % 10 == 9:
                await asyncio.sleep(0)

        self._current_md = None
        self._response_buf = ""
        self._thinking_md = None
        self._thinking_buf = ""
        self._last_event_type = None
        self._generating = False
        self._response_dirty = False
        self._thinking_dirty = False
        self._last_response_buf = ""
        self._last_thinking_buf = ""
        self._update_status_bar()

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
        lines.append("  [cyan]/name[/cyan]  Rename the current session  [dim]/name <new-name>[/dim]")
        lines.append("  [cyan]/help[/cyan]  Show this message")
        lines.append("  [cyan]/quit[/cyan]  Exit")

        help_text = "\n".join(lines)
        widget = Static(help_text)
        chat_area.mount(widget)
        self._scroll_to_bottom_if_following()

    def _print_warning(self, text: str) -> None:
        """Print a warning message."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        widget = Static(f"[yellow]{text}[/yellow]")
        chat_area.mount(widget)
        self._scroll_to_bottom_if_following()

    def _print_dim_feedback(self, text: str) -> None:
        """Print dim feedback about a config change."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        widget = Static(f"[dim]{text}[/dim]")
        chat_area.mount(widget)
        self._scroll_to_bottom_if_following()

    def _print_system_feedback(self, text: str) -> None:
        """Print dim feedback about a config change."""
        chat_area = self.query_one("#chat-area", VerticalScroll)
        widget = Static(f"[dim orange]{text}[/dim orange]")
        chat_area.mount(widget)
        self._scroll_to_bottom_if_following()


DEFAULT_CONFIG = Path.home() / ".ph" / "config.toml"

app = typer.Typer(help="PicoHarness — local LLM agent TUI", add_completion=False)


def _load_config(config_path: Optional[str], preset: Optional[str]):
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise typer.Exit(f"Config not found: {path}\nRun `ph init` to create one.", code=1)
    return load_config(path, preset)


def _build_api(cfg, session_manager: SessionManager) -> BackendAPI:
    tool_name_map: dict[str, type] = {tool.name: tool for tool in ALL_TOOLS}
    tools = []
    tool_names = cfg.tools if cfg.tools else list(tool_name_map.keys())
    for name in tool_names:
        if name not in tool_name_map:
            raise typer.Exit(f"Unknown tool '{name}'. Valid: {list(tool_name_map.keys())}", code=1)
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

    return BackendAPI(
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
        summarizer_model=cfg.summarizer_model,
        session_manager=session_manager,
    )


@app.command()
def init(path: Optional[str] = typer.Argument(None, help="Config file path")):
    """Write a sample config file."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    if config_path.exists():
        if not typer.confirm(f"{config_path} already exists. Overwrite?"):
            typer.echo("Aborted.")
            raise typer.Exit()
    generate_config(config_path)


@app.command()
def chat(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path"),
    preset: Optional[str] = typer.Argument(None, help="Config preset"),
    resume: bool = typer.Option(False, "--resume", help="Resume an existing session"),
):
    """Start the TUI chat interface."""
    cfg = _load_config(config, preset)
    session_manager = SessionManager(Path(cfg.session_save_location).expanduser())
    api = _build_api(cfg, session_manager)

    async def run_chat():
        async with api:
            chat_app = ChatApp(api=api, show_think=cfg.show_think, show_picker=resume)
            await chat_app.run_async()

    asyncio.run(run_chat())


@app.command()
def service(action: Literal["start", "stop"] = typer.Argument(..., help="Action to perform")):
    """Start or stop Docker services."""
    if action == "start":
        services_main(["up", "-d"])
    else:
        services_main(["down"])


@app.command()
def doctor(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path"),
    preset: Optional[str] = typer.Argument(None, help="Config preset"),
):
    """Check that provider is reachable."""
    _load_config(config, preset)
    config_path = Path(config) if config else DEFAULT_CONFIG
    sys.exit(asyncio.run(doctor_main(config_path, preset)))


def cli() -> None:
    app()
