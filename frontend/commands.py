import inspect
from collections.abc import Callable

from prompt_toolkit.completion import NestedCompleter

_commands = {}


def command(name: str, description: str | None = None) -> Callable:
    def inner(fn: Callable) -> Callable:
        _commands[name] = {"fn": fn, "description": description or inspect.getdoc(fn)}
        return fn

    return inner


class CommandDispatcher:
    def __init__(self) -> None:
        self._commands = {name: options for name, options in _commands.items()}

        self.completer = NestedCompleter.from_nested_dict(
            {f"/{name}": None for name, options in self._commands.items()}
        )

    def dispatch(self, raw: str) -> bool:
        if raw.startswith("/"):
            raw = raw[1:]

        items = raw.split()
        name = items[0]
        args = items[1:] if len(items) > 1 else []

        if name not in self._commands:
            print(f"Unknown command: /{name}. Type /help for available commands.")
            return False

        self._commands[name]["fn"](self, *args)
        return True

    @command("help")
    def help(self) -> None:
        """
        Show available commands
        """

        print("\nAvailable commands:")
        for name in sorted(self._commands.keys()):
            desc = self._commands[name]["description"]
            print(f"  /{name}")
            if desc:
                print(f"    {self._commands[name]['description']}")
        print()

    @property
    def command_names(self) -> list[str]:
        return list(self._commands.keys())
