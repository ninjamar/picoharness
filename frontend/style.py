from prompt_toolkit.styles import Style

_CMD_STYLE = Style.from_dict(
    {
        "cmd": "bold cyan",
        "desc": "italic",
        "head": "bold underline",
        "args": "dim",
        "selected-option": "fg:ansigreen bold",
        "number": "fg:ansicyan",
        "current-marker": "bold",
    }
)
