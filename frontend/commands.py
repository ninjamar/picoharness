from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from frontend.schema import (
    CUSTOM_PROVIDER_LABEL,
    FIELDS,
    DialogueMenu,
    MultiSelectMenu,
    TextInputMenu,
    ToggleMenu,
    resolve_choices,
)
from frontend.tui import open_dialog, open_text_dialog, CommandMode

if TYPE_CHECKING:
    from frontend.app import ChatApp


async def _open_custom_provider_input(app: ChatApp, menu: DialogueMenu) -> None:
    current = menu.get_current(app)
    open_text_dialog(app._command, current, lambda url: menu.set_current(app, url))


async def dispatch_command(text: str, app: ChatApp) -> None:
    parts = text[1:].split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "help":
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
        app._chat_log.lines.append("\n".join(lines))
        return
    if cmd == "quit":
        app._should_exit = True
        return

    field_map = {f.name: f for f in FIELDS}
    field = field_map.get(cmd)
    if not field:
        app._chat_log.lines.append(f"Unknown command /{cmd}. Type /help.")
        return

    match field.menu:
        case ToggleMenu() as m:
            if arg in ("on", "off"):
                m.set_current(app, arg == "on")
                app._chat_log.lines.append(f"/{field.name} → {arg}")
                return
            current = "on" if m.get_current(app) else "off"
            open_dialog(
                app._command,
                CommandMode.CHOICE,
                choices=["on", "off"],
                current=current,
                callback=lambda v: m.set_current(app, v == "on"),
            )

        case TextInputMenu() as m:
            if not arg:
                current = m.get_current(app)
                open_dialog(
                    app._command,
                    CommandMode.TEXT,
                    current=current,
                    callback=lambda v: m.set_current(app, None if (m.nullable and v and v.lower() == "none") else v),
                )
                return
            val = None if (m.nullable and arg.lower() == "none") else arg
            m.set_current(app, val)
            app._chat_log.lines.append(f"/{field.name} → {val!r}")

        case DialogueMenu() as m:
            if arg:
                m.set_current(app, arg)
                app._chat_log.lines.append(f"/{field.name} → {arg}")
                return
            choices = await resolve_choices(m.choices, app)
            current = m.get_current(app)

            if field.name == "provider":

                def provider_callback(v: str) -> None:
                    if v == CUSTOM_PROVIDER_LABEL:
                        app._command.visible = False
                        asyncio.create_task(_open_custom_provider_input(app, m))
                    else:
                        m.set_current(app, v)

                open_dialog(
                    app._command, CommandMode.CHOICE, choices=choices, current=current, callback=provider_callback
                )
            else:
                open_dialog(
                    app._command,
                    CommandMode.CHOICE,
                    choices=choices,
                    current=current,
                    callback=lambda v: m.set_current(app, v),
                )

        case MultiSelectMenu() as m:
            if arg:
                selected = [s.strip() for s in arg.replace(",", " ").split() if s.strip()]
                m.set_current(app, selected)
                app._chat_log.lines.append(f"/{field.name} → {selected}")
                return
            choices = await resolve_choices(m.choices, app)
            current = list(m.get_current(app) or [])
            open_dialog(
                app._command,
                CommandMode.MULTISELECT,
                choices=choices,
                selected=current,
                callback=lambda v: m.set_current(app, v),
            )
