import os
import select
import sys
import termios
import tty

from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys


def detect_kitty() -> bool:
    """Return True if the terminal supports the kitty keyboard protocol."""
    if not sys.stdin.isatty():
        return False
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        sys.stdout.write("\x1b[?u")
        sys.stdout.flush()
        ready, _, _ = select.select([fd], [], [], 0.1)
        if not ready:
            return False
        resp = os.read(fd, 32).decode("ascii", errors="ignore")
        return resp.startswith("\x1b[?") and resp.endswith("u")
    except Exception:
        return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


_CTRL_KEYS = [
    Keys.ControlA,
    Keys.ControlB,
    Keys.ControlC,
    Keys.ControlD,
    Keys.ControlE,
    Keys.ControlF,
    Keys.ControlG,
    Keys.ControlH,
    Keys.ControlI,
    Keys.ControlJ,
    Keys.ControlK,
    Keys.ControlL,
    Keys.ControlM,
    Keys.ControlN,
    Keys.ControlO,
    Keys.ControlP,
    Keys.ControlQ,
    Keys.ControlR,
    Keys.ControlS,
    Keys.ControlT,
    Keys.ControlU,
    Keys.ControlV,
    Keys.ControlW,
    Keys.ControlX,
    Keys.ControlY,
    Keys.ControlZ,
]


def register_sequences() -> None:
    """Register kitty keyboard protocol escape sequences with prompt_toolkit."""
    ANSI_SEQUENCES["\x1b[13;2u"] = Keys.ControlJ
    for i, key in enumerate(_CTRL_KEYS):
        ANSI_SEQUENCES[f"\x1b[{97 + i};5u"] = key


def make_input_bindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("enter")
    def _(event):
        event.current_buffer.validate_and_handle()

    @kb.add("c-j")
    def _(event):
        event.current_buffer.insert_text("\n")

    return kb


def init_kitty() -> None:
    """Initialize kitty keyboard protocol: register sequences and enable protocol."""
    register_sequences()
    sys.stdout.write("\x1b[>1u")
    sys.stdout.flush()


def end_kitty() -> None:
    """Disable kitty keyboard protocol."""
    sys.stdout.write("\x1b[<u")
    sys.stdout.flush()
