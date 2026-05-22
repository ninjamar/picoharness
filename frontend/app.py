from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path

from pyratatui import (
    AsyncTerminal,
    Block,
    Constraint,
    Direction,
    Layout,
    Paragraph,
    TextArea,
)

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
from frontend.widgets import (
    CommandMode,
    CommandPanelState,
    CompletionState,
    draw_command_panel,
    draw_completion_menu,
)

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


class ChatApp:
    def __init__(self, api: BackendAPI, show_think: bool = True) -> None:
        self.api = api
        self._show_thinking = show_think
        self._turn_in_progress = False

        self._chat_lines: list[str] = []
        self._scroll_offset: int = 0
        self._response_buf: str = ""
        self._thinking_buf: str = ""
        self._last_event_type: str | None = None

        self._textarea = TextArea()
        self._input_locked = False
        self._history: list[str] = []
        self._history_index: int = -1
        self._saved_input: str = ""

        self._completion = CompletionState()
        self._command = CommandPanelState()
        self._should_exit = False

    @property
    def show_think(self) -> bool:
        return self._show_thinking

    def set_show_think(self, value: bool) -> None:
        self._show_thinking = value

    def _overlay_height(self) -> int:
        if self._completion.visible and self._completion.items:
            return min(10, len(self._completion.items) + 2)
        elif self._command.visible:
            return min(12, len(self._command.choices) + 4) if self._command.choices else 7
        return 0

    def _add_chat_line(self, text: str) -> None:
        self._chat_lines.append(text)

    def _reset_response(self) -> None:
        self._response_buf = ""
        self._last_event_type = None

    def _reset_thinking(self) -> None:
        self._thinking_buf = ""

    async def _consume_events(self) -> None:
        async for event in self.api.stream_events():
            match event:
                case UserInputEvent(text=t):
                    self._reset_response()
                    self._reset_thinking()
                    self._add_chat_line(f"> {t}")

                case ThinkingEvent(fragment=f):
                    if not self._show_thinking:
                        continue
                    if self._last_event_type != "thinking":
                        self._thinking_buf = ""
                        self._add_chat_line("")
                    self._last_event_type = "thinking"
                    self._thinking_buf += f
                    if self._chat_lines:
                        self._chat_lines[-1] = f"💭 {self._thinking_buf}"

                case ResponseEvent(fragment=f):
                    if self._last_event_type != "response":
                        self._response_buf = ""
                        self._add_chat_line("")
                    self._last_event_type = "response"
                    self._response_buf += f
                    if self._chat_lines:
                        self._chat_lines[-1] = f"{self._response_buf}"

                case ToolStartEvent(tool_name=name, tool_input=inp):
                    self._reset_response()
                    self._add_chat_line(f"⏺ {name}({_fmt_tool_input(inp)})")

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
                    self._add_chat_line(text)

                case ToolErrorEvent(tool_name=name, error=err):
                    self._add_chat_line(f"  {name}: {err}")

                case DoneEvent(error=error):
                    self._reset_response()
                    self._reset_thinking()
                    if error:
                        self._add_chat_line(f"Error: {error}")
                    self._turn_in_progress = False
                    self._unlock_input()

    def _handle_key(self, ev) -> None:
        if ev is None:
            return

        key_code = getattr(ev, "code", None)
        ctrl = getattr(ev, "ctrl", False)
        alt = getattr(ev, "alt", False)
        shift = getattr(ev, "shift", False)

        if key_code == "c" and ctrl:
            if self._turn_in_progress:
                self.api.cancel_current()
            else:
                self._should_exit = True
            return

        if self._input_locked:
            return

        if self._completion.visible:
            self._handle_completion_key(key_code)
            return

        if self._command.visible:
            self._handle_command_key(key_code)
            return

        if key_code == "Enter":
            text = "\n".join(self._textarea.lines()).strip()
            if text:
                self._history.append(text)
                self._history_index = -1
                self._saved_input = ""
                self.api.feed(str(uuid.uuid4()), text)
                self._turn_in_progress = True
                self._lock_input()
                self._textarea = TextArea()
        elif key_code == "Up":
            self._history_up()
        elif key_code == "Down":
            self._history_down()
        elif key_code == "j" and ctrl:
            self._textarea.insert_str("\n")
        elif key_code is not None:
            self._textarea.input_key(key_code, ctrl, alt, shift)

        self._update_completions()

    def _handle_completion_key(self, key_code: str | None) -> None:
        if not key_code:
            return
        if key_code == "Up":
            if self._completion.index > 0:
                self._completion.index -= 1
                self._completion.list_state.select(self._completion.index)
        elif key_code == "Down":
            if self._completion.index < len(self._completion.items) - 1:
                self._completion.index += 1
                self._completion.list_state.select(self._completion.index)
        elif key_code == "Tab" or key_code == "Enter":
            if self._completion.items:
                cmd = self._completion.items[self._completion.index][0]
                self._apply_completion(cmd)
                self._completion.visible = False
        elif key_code == "Esc":
            self._completion.visible = False

    def _handle_command_key(self, key_code: str | None) -> None:
        if not key_code:
            return

        if key_code == "Up":
            self._command.list_state.select_previous()
        elif key_code == "Down":
            self._command.list_state.select_next()
        elif key_code == "Enter":
            selected_idx = self._command.list_state.selected
            if self._command.mode == CommandMode.CHOICE:
                if selected_idx is not None and selected_idx < len(self._command.choices):
                    if self._command.callback:
                        self._command.callback(self._command.choices[selected_idx])
                self._command.visible = False
            elif self._command.mode == CommandMode.MULTISELECT:
                selected_choices = [
                    self._command.choices[i] for i in range(len(self._command.choices)) if i in self._command.selected
                ]
                if self._command.callback:
                    self._command.callback(selected_choices)
                self._command.visible = False
            elif self._command.mode == CommandMode.TEXT:
                if self._command.callback:
                    self._command.callback("\n".join(self._command.textarea.lines()))
                self._command.visible = False
        elif key_code == "Esc":
            self._command.visible = False
        elif key_code == " " and self._command.mode == CommandMode.MULTISELECT:
            selected_idx = self._command.list_state.selected
            if selected_idx is not None:
                choice = self._command.choices[selected_idx]
                if choice in self._command.selected:
                    self._command.selected.discard(choice)
                else:
                    self._command.selected.add(choice)
        elif self._command.mode == CommandMode.TEXT:
            self._command.textarea.input_key(key_code, False, False, False)

    def _update_completions(self) -> None:
        text = "\n".join(self._textarea.lines())
        if text.startswith("/"):
            after_slash = text[1:]
            if " " in after_slash:
                self._completion.visible = False
                return
            query = after_slash.lower() if after_slash.strip() else ""
            matches = [(f"/{cmd}", desc) for cmd, desc in _ALL_COMMANDS if cmd.startswith(query)]
            if matches:
                self._completion.items = matches
                self._completion.index = 0
                self._completion.list_state.select(0)
                self._completion.visible = True
            else:
                self._completion.visible = False
        else:
            self._completion.visible = False

    def _apply_completion(self, cmd: str) -> None:
        self._textarea = TextArea()
        self._textarea.insert_str(cmd + " ")

    def _history_up(self) -> None:
        if not self._history:
            return
        if self._history_index == -1:
            self._saved_input = "\n".join(self._textarea.lines())
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        self._textarea = TextArea.from_lines(self._history[self._history_index].split("\n"))

    def _history_down(self) -> None:
        if self._history_index == -1:
            return
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self._textarea = TextArea.from_lines(self._history[self._history_index].split("\n"))
        else:
            self._history_index = -1
            self._textarea = TextArea()
            if self._saved_input:
                self._textarea.insert_str(self._saved_input)

    def _lock_input(self) -> None:
        self._input_locked = True

    def _unlock_input(self) -> None:
        self._input_locked = False

    def _log(self, msg: str) -> None:
        self._add_chat_line(msg)

    def _show_help(self) -> None:
        lines = ["Commands:"]
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
            lines.append(f"  /{field.name}  {field.description}  {usage}")
        lines.append("  /help  Show this message")
        lines.append("  /quit  Exit")
        self._log("\n".join(lines))

    async def _dispatch_command(self, text: str) -> None:
        parts = text[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "help":
            self._show_help()
            return
        if cmd == "quit":
            self._should_exit = True
            return

        field_map = {f.name: f for f in FIELDS}
        field = field_map.get(cmd)
        if not field:
            self._log(f"Unknown command /{cmd}. Type /help.")
            return

        match field.menu:
            case ToggleMenu() as m:
                if arg in ("on", "off"):
                    m.set_current(self, arg == "on")
                    self._log(f"/{field.name} → {arg}")
                    return
                current = "on" if m.get_current(self) else "off"
                await self._show_choice_dialog(["on", "off"], current, lambda v: m.set_current(self, v == "on"))

            case TextInputMenu() as m:
                if not arg:
                    current = m.get_current(self)
                    await self._show_text_input_dialog(
                        current,
                        lambda v: m.set_current(self, None if (m.nullable and v and v.lower() == "none") else v),
                    )
                    return
                val = None if (m.nullable and arg.lower() == "none") else arg
                m.set_current(self, val)
                self._log(f"/{field.name} → {val!r}")

            case DialogueMenu() as m:
                if arg:
                    m.set_current(self, arg)
                    self._log(f"/{field.name} → {arg}")
                    return
                choices = await resolve_choices(m.choices, self)
                current = m.get_current(self)

                if field.name == "provider":

                    def provider_callback(v: str) -> None:
                        if v == CUSTOM_PROVIDER_LABEL:
                            self._command.visible = False
                            asyncio.create_task(self._show_custom_provider_input(m))
                        else:
                            m.set_current(self, v)

                    await self._show_choice_dialog(choices, current, provider_callback)
                else:
                    await self._show_choice_dialog(choices, current, lambda v: m.set_current(self, v))

            case MultiSelectMenu() as m:
                if arg:
                    selected = [s.strip() for s in arg.replace(",", " ").split() if s.strip()]
                    m.set_current(self, selected)
                    self._log(f"/{field.name} → {selected}")
                    return
                choices = await resolve_choices(m.choices, self)
                current = list(m.get_current(self) or [])
                await self._show_multiselect_dialog(choices, current, lambda v: m.set_current(self, v))

    async def _show_custom_provider_input(self, menu: DialogueMenu) -> None:
        current = menu.get_current(self)
        await self._show_text_input_dialog(current, lambda url: menu.set_current(self, url))

    async def _show_choice_dialog(self, choices: list[str], current: str, callback) -> None:
        self._command.mode = CommandMode.CHOICE
        self._command.choices = choices
        self._command.callback = callback
        self._command.list_state = type(self._command.list_state)()
        try:
            idx = choices.index(current)
            self._command.list_state.select(idx)
        except ValueError:
            self._command.list_state.select(0)
        self._command.visible = True

    async def _show_multiselect_dialog(self, choices: list[str], current: list[str], callback) -> None:
        self._command.mode = CommandMode.MULTISELECT
        self._command.choices = choices
        self._command.selected = set(current)
        self._command.callback = callback
        self._command.list_state = type(self._command.list_state)()
        self._command.visible = True

    async def _show_text_input_dialog(self, current: str | None, callback) -> None:
        self._command.mode = CommandMode.TEXT
        self._command.textarea = TextArea.from_lines([current or ""])
        self._command.callback = callback
        self._command.choices = []
        self._command.visible = True

    def run_app(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with self.api:
            async with AsyncTerminal() as term:
                term.hide_cursor()
                stream_task = asyncio.create_task(self._consume_events())

                try:
                    async for ev in term.events(fps=30):
                        if ev is not None and getattr(ev, "code", None) == "Enter":
                            text = "\n".join(self._textarea.lines()).strip()
                            if text.startswith("/") and not self._command.visible and not self._completion.visible:
                                await self._dispatch_command(text)
                                self._textarea = TextArea()
                                self._update_completions()
                                term.draw(self._build_ui())
                                continue

                        self._handle_key(ev)
                        term.draw(self._build_ui())

                        if self._should_exit:
                            break

                finally:
                    term.show_cursor()
                    stream_task.cancel()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass

    def _build_ui(self):
        # Snapshot state for closure
        chat_lines = list(self._chat_lines)
        textarea = self._textarea
        completion = self._completion
        command = self._command
        overlay_h = self._overlay_height()

        def ui(frame, _lines=chat_lines, _ta=textarea, _comp=completion, _cmd=command, _oh=overlay_h):
            area = frame.area
            chunks = (
                Layout()
                .direction(Direction.Vertical)
                .constraints([Constraint.fill(1), Constraint.length(_oh), Constraint.length(3)])
                .split(area)
            )
            chat_area, overlay_area, input_area = chunks

            if not _lines:
                frame.render_widget(
                    Paragraph.from_string("LocalAI  Ctrl+C to quit · Ctrl+J for newline · /help for commands").block(
                        Block().bordered()
                    ),
                    chat_area,
                )
            else:
                visible_height = int(chat_area.height)
                total_lines = len(_lines)
                if total_lines <= visible_height:
                    visible_lines = _lines
                else:
                    start = max(0, total_lines - visible_height)
                    visible_lines = _lines[start : start + visible_height]
                text = "\n".join(visible_lines)
                frame.render_widget(Paragraph.from_string(text).block(Block().bordered()), chat_area)

            if _comp.visible:
                draw_completion_menu(frame, overlay_area, _comp)
            elif _cmd.visible:
                draw_command_panel(frame, overlay_area, _cmd)

            frame.render_textarea(_ta, input_area)

        return ui


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
