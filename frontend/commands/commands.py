from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.shortcuts import print_formatted_text

from backend.provider.provider import OllamaProvider, OpenAICompatibleProvider
from frontend.style import _CMD_STYLE

from ._internal import (
    Command,
    DialogueMenu,
    FieldDef,
    MultiSelectMenu,
    TextInputMenu,
    ToggleMenu,
    build_registry,
)

if TYPE_CHECKING:
    from frontend.app import ChatFrontend


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


async def _fetch_models(frontend: "ChatFrontend") -> list[str]:
    """Fetch available model names from the backend at command-execution time."""
    model_infos = await frontend.backend.config.get_available_models()
    return [m.name for m in model_infos]


def _get_enabled_tools(frontend: "ChatFrontend") -> list[str]:
    """Fetch all available tool names from the backend."""
    return frontend.backend.config.get_all_tools()


def _set_provider(frontend: "ChatFrontend", value: str) -> None:
    """Create appropriate provider instance and set it in config."""
    if value == "ollama":
        provider = OllamaProvider()
    else:
        provider = OpenAICompatibleProvider(base_url=f"http://{value}/v1")
    frontend.backend.config.set_provider(provider)


_custom_provider_input = TextInputMenu(
    get_current=lambda f: f.backend.config.provider,
    set_current=_set_provider,
    label="Custom...",
)

FIELDS = [
    FieldDef(
        name="model",
        type=str,
        default="qwen3:2b",
        description="Ollama/OpenAI-compatible model identifier",
        menu=DialogueMenu(
            get_current=lambda f: f.backend.config.model,
            set_current=lambda f, v: f.backend.config.set_model(v),
            choices=_fetch_models,
        ),
    ),
    FieldDef(
        name="provider",
        type=str,
        default="ollama",
        description="Provider: 'ollama' or an OpenAI-compatible base URL",
        menu=DialogueMenu(
            get_current=lambda f: f.backend.config.provider,
            set_current=_set_provider,
            choices=lambda f: ["ollama", _custom_provider_input],
        ),
    ),
    FieldDef(
        name="think",
        type=bool,
        default=False,
        description="Enable chain-of-thought / extended thinking",
        menu=ToggleMenu(
            get_current=lambda f: f.backend.config.think,
            set_current=lambda f, v: f.backend.config.set_think(v),
        ),
    ),
    FieldDef(
        name="show_think",
        type=bool,
        default=True,
        description="Display thinking output in the terminal",
        menu=ToggleMenu(
            get_current=lambda f: f.show_think,
            set_current=lambda f, v: f.set_show_think(v),
        ),
    ),
    FieldDef(
        name="system_prompt_path",
        type=str | None,
        default=None,
        description="Path to a Markdown system prompt file",
        menu=TextInputMenu(
            get_current=lambda f: f.backend.config.system_prompt_path,
            set_current=lambda f, v: f.backend.config.set_system_prompt_path(v),
            nullable=True,
        ),
    ),
    FieldDef(
        name="enabled_tools",
        type=list,
        default=[],
        description="Tools available to the model",
        menu=MultiSelectMenu(
            get_current=lambda f: f.backend.config.enabled_tools,
            set_current=lambda f, v: f.backend.config.set_enabled_tools(v),
            choices=_get_enabled_tools,
        ),
    ),
]


def build_all_commands(frontend_ref) -> dict[str, type[Command]]:
    """Build the complete command dict: schema-driven + help + quit."""
    schema_commands = build_registry(FIELDS)
    return {"help": HelpCommand, "quit": QuitCommand, **schema_commands}
