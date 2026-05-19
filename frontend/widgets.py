from textual import events
from textual.message import Message
from textual.widgets import TextArea


class InputArea(TextArea):
    """TextArea that submits on Enter and inserts newline on Ctrl+J."""

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            if text := self.text.strip():
                self.post_message(self.Submitted(text))
                self.clear()
        elif event.key == "ctrl+j":
            event.prevent_default()
            self.insert("\n")
