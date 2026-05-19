import asyncio
from dataclasses import dataclass, field
from inspect import iscoroutinefunction
from typing import TYPE_CHECKING, Any

from prompt_toolkit import prompt
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.shortcuts.choice_input import ChoiceInput
from prompt_toolkit.widgets import CheckboxList

from frontend.style import _CMD_STYLE

if TYPE_CHECKING:
    from frontend.app import ChatFrontend


@dataclass
class BaseMenu:
    get_current: Any  # Callable[["ChatFrontend"], Any]
    set_current: Any  # Callable[["ChatFrontend", Any], None]
    label: str | None = None


@dataclass
class ToggleMenu(BaseMenu):
    pass


@dataclass
class TextInputMenu(BaseMenu):
    nullable: bool = False


@dataclass
class DialogueMenu(BaseMenu):
    choices: Any = field(default_factory=list)


@dataclass
class MultiSelectMenu(BaseMenu):
    choices: Any = field(default_factory=list)


@dataclass
class FieldDef:
    name: str
    type: type
    default: Any
    description: str
    menu: BaseMenu


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
                _print_cmd(cls.usage_message)

    @staticmethod
    async def _execute(frontend: "ChatFrontend", args: list[str]) -> None:
        raise NotImplementedError


async def _handle_toggle_menu(menu: ToggleMenu, frontend: "ChatFrontend", field_name: str, args: list[str]) -> None:
    """Handle toggle menu: requires on/off arg."""
    if not args or args[0].lower() not in ("on", "off"):
        raise _ShowUsageMessage()
    value = args[0].lower() == "on"
    menu.set_current(frontend, value)
    state = "enabled" if value else "disabled"
    _print_cmd(f"{field_name} {state}")


async def _handle_dialogue_menu(menu: DialogueMenu, frontend: "ChatFrontend", field_name: str, args: list[str]) -> None:
    """Handle dialogue menu: choices can include other menus."""
    raw_choices = await _resolve_choices(menu.choices, frontend)
    current = menu.get_current(frontend)

    # Partition into string choices and embedded menus
    str_choices: list[str] = []
    menu_choices: dict[str, BaseMenu] = {}
    for c in raw_choices:
        if isinstance(c, BaseMenu):
            lbl = c.label or repr(c)
            str_choices.append(lbl)
            menu_choices[lbl] = c
        else:
            str_choices.append(c)

    if args:
        selected_str = args[0]
        if selected_str not in str_choices:
            _print_cmd(f"Invalid choice. Options: {', '.join(str_choices)}")
            return
    else:
        selected_str = await _run_choice_input(
            f"Select {field_name} (current: {current})",
            [(c, c) for c in str_choices],
            current if current in str_choices else str_choices[0],
        )
        if selected_str is None:
            return

    # If the selected item is an embedded menu, recurse into it
    if selected_str in menu_choices:
        await _execute_menu(menu_choices[selected_str], frontend, selected_str, [])
        return

    if selected_str != current:
        menu.set_current(frontend, selected_str)
        _print_cmd(f"{field_name} → {selected_str}")


async def _handle_text_input_menu(
    menu: TextInputMenu, frontend: "ChatFrontend", field_name: str, args: list[str]
) -> None:
    """Handle text input menu: free text or nullable clear. Bug fix: interactive fallback when args empty."""
    if not args:
        # Interactive mode: prompt for text. Used when embedded in another menu or called without args.
        text = await _prompt_text(f"{menu.label or field_name}: ")
        if text is None:
            return
        menu.set_current(frontend, text)
        _print_cmd(f"{field_name} → {text}")
        return

    raw = " ".join(args)
    if menu.nullable and raw.lower() == "none":
        menu.set_current(frontend, None)
        _print_cmd(f"{field_name} cleared")
    else:
        menu.set_current(frontend, raw)
        _print_cmd(f"{field_name} → {raw}")


async def _handle_multi_select_menu(
    menu: MultiSelectMenu, frontend: "ChatFrontend", field_name: str, args: list[str]
) -> None:
    """Handle multi-select menu: always interactive, no subcommands."""
    # Always interactive; args are ignored
    raw_choices = await _resolve_choices(menu.choices, frontend)
    str_choices: list[str] = []
    for c in raw_choices:
        if isinstance(c, BaseMenu):
            str_choices.append(c.label or repr(c))
        else:
            str_choices.append(c)

    current: list[str] = list(menu.get_current(frontend))
    values = [(c, c) for c in str_choices]
    result = await _run_multi_choice_input(
        f"Select {field_name} (Space to toggle, Enter to confirm)",
        values,
        [c for c in current if c in str_choices],
    )
    if result is None:
        return
    menu.set_current(frontend, result)
    _print_cmd(f"{field_name}: {result}")


async def _execute_menu(menu: BaseMenu, frontend: "ChatFrontend", field_name: str, args: list[str]) -> None:
    """Central dispatch for menu execution based on type."""
    match menu:
        case ToggleMenu():
            await _handle_toggle_menu(menu, frontend, field_name, args)
        case DialogueMenu():
            await _handle_dialogue_menu(menu, frontend, field_name, args)
        case TextInputMenu():
            await _handle_text_input_menu(menu, frontend, field_name, args)
        case MultiSelectMenu():
            await _handle_multi_select_menu(menu, frontend, field_name, args)
        case _:
            raise TypeError(f"Unknown menu type: {type(menu)}")


# ── Registry building ─────────────────────────────────────────────────────────


def _make_cmd(
    name: str, description: str, usage: str | None, completions: dict[str, Any] | None, execute_fn
) -> type[Command]:
    """Dynamically create a Command subclass from parts."""
    return type(
        name.title().replace("-", "") + "Command",
        (Command,),
        {
            "name": name,
            "description": description,
            "usage_message": usage,
            "completions": completions,
            "_execute": staticmethod(execute_fn),
        },
    )


def _build_usage_for_field(f: FieldDef) -> str | None:
    """Build a usage string for a field based on its menu type."""
    if isinstance(f.menu, ToggleMenu):
        return f"/{f.name} on|off"
    if isinstance(f.menu, TextInputMenu):
        parts = ["<value>"] + (["none"] if f.menu.nullable else [])
        return f"/{f.name} " + "|".join(parts)
    return None


def _completions_for_menu(menu: BaseMenu) -> dict[str, Any] | None:
    """Return completions dict for a menu, or None if dynamic/interactive."""
    if isinstance(menu, ToggleMenu):
        return {"on": None, "off": None}
    return None


def build_registry(fields: list[FieldDef]) -> dict[str, type[Command]]:
    """Build a registry of Command subclasses from schema fields.

    Each field.name maps directly to one command (no grouping).
    """
    registry: dict[str, type[Command]] = {}
    for f in fields:

        async def _execute(frontend: "ChatFrontend", args: list[str], _f=f) -> None:
            await _execute_menu(_f.menu, frontend, _f.name, args)

        usage = _build_usage_for_field(f)
        completions = _completions_for_menu(f.menu)
        registry[f.name] = _make_cmd(f.name, f.description, usage, completions, _execute)
    return registry


def _print_cmd(msg: str) -> None:
    """Print a command response with styling."""
    print_formatted_text(FormattedText([("class:cmd", msg + "\n")]), style=_CMD_STYLE)


async def _resolve_choices(choices: Any, frontend: "ChatFrontend") -> list:
    """Resolve choices: list passthrough or call callable(frontend) - handles both async and sync."""
    if iscoroutinefunction(choices):
        return list(await choices(frontend))
    elif callable(choices):
        return list(choices(frontend))
    return list(choices)


async def _prompt_text(label: str = "Enter value: ") -> str | None:
    """Prompt for free-text value. Returns None if empty/cancelled."""

    text = (await asyncio.to_thread(prompt, label)).strip()
    return text if text else None


async def _run_choice_input(message: str, values: list[tuple[str, Any]], default: str) -> str | None:
    """Run an interactive ChoiceInput and return the selected value (or None if cancelled)."""
    escape_kb = KeyBindings()

    @escape_kb.add("escape")
    def _(event: Any) -> None:
        event.app.exit(result=None)

    @escape_kb.add("c-c")
    def _(event: Any) -> None:
        event.app.exit(result=None)

    choice_input = ChoiceInput(
        message=message,
        options=values,
        default=default,
        style=_CMD_STYLE,
        key_bindings=escape_kb,
    )
    app = choice_input._create_application()
    app.erase_when_done = True
    try:
        return await asyncio.to_thread(app.run)
    except KeyboardInterrupt:
        return None


async def _run_multi_choice_input(
    message: str, values: list[tuple[str, Any]], default_values: list[str]
) -> list[str] | None:
    """Run an interactive CheckboxList and return selected values (or None if cancelled)."""
    cb_list = CheckboxList(values=values, default_values=default_values)

    kb = KeyBindings()

    @kb.add("enter")
    def _confirm(event: Any) -> None:
        event.app.exit(result=cb_list.current_values)

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event: Any) -> None:
        event.app.exit(result=None)

    label = Window(
        content=FormattedTextControl(FormattedText([("class:cmd", message + "\n")])),
        dont_extend_height=True,
    )

    app = Application(
        layout=Layout(HSplit([label, cb_list])),
        key_bindings=kb,
        style=_CMD_STYLE,
        erase_when_done=True,
    )
    try:
        return await asyncio.to_thread(app.run)
    except KeyboardInterrupt:
        return None
