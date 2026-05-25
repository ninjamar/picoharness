from collections.abc import Callable
from typing import Any

from textual.containers import Container
from textual.message import Message
from textual.widgets import (
    Input,
    Label,
    ListItem,
    ListView,
    OptionList,
    SelectionList,
    TextArea,
)
from textual.widgets._option_list import Option


class InputArea(TextArea):
    """Multi-line text input with history, slash-command completion, and message dispatch.

    Maintains:
    - _history: all submitted commands
    - _history_index: position in history (None = current input)
    - _saved_input: unsaved text when navigating history
    """

    class Submitted(Message):
        def __init__(self, text: str):
            self.text = text
            super().__init__()

    class CompletionRequested(Message):
        """User pressed Tab/Enter to apply a completion."""

        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._history: list[str] = []
        self._history_index: int | None = None
        self._saved_input: str = ""

    @property
    def _completion_visible(self) -> bool:
        """Check if completion menu is visible (DOM-based)."""
        try:
            return self.app.query_one("#completion").has_class("visible")
        except Exception:
            return False

    @property
    def _panel_visible(self) -> bool:
        """Check if command panel is visible (DOM-based)."""
        try:
            return self.app.query_one("#command-panel").has_class("visible")
        except Exception:
            return False

    def on_key(self, event) -> None:
        """Handle key events."""
        key = event.key

        # Handle Enter
        if key == "enter":
            if self._completion_visible:
                # Run the selected command as normal input
                event.prevent_default()
                selected = self.app.query_one("#completion", CompletionMenu).get_selected()
                if selected:
                    # Update history and clear state (same as normal submit)
                    self._history.append(selected)
                    self._history_index = None
                    self._saved_input = ""
                    self.text = ""
                    # Post Submitted to dispatch through normal flow
                    self.post_message(self.Submitted(selected))
                return
            if self._panel_visible:
                # CommandPanel owns enter
                return
            # Submit the text
            text = self.text
            if text:
                self._history.append(text)
            self._history_index = None
            self._saved_input = ""
            self.post_message(self.Submitted(text))
            self.text = ""
            event.prevent_default()
            return

        # Handle Tab
        if key == "tab":
            if self._completion_visible:
                # Tab applies completion
                event.prevent_default()
                self.post_message(self.CompletionRequested())
                return
            # Default tab behavior

        # Handle Up/Down arrows
        if key == "up":
            if self._completion_visible:
                # Navigate completion menu
                event.prevent_default()
                self.app.query_one("#completion", CompletionMenu).navigate(-1)
                return
            if self._panel_visible:
                # CommandPanel owns Up
                return
            # History navigation
            if not self._history:
                return
            if self._history_index is None:
                self._saved_input = self.text
                self._history_index = len(self._history) - 1
            else:
                self._history_index = max(0, self._history_index - 1)
            self.text = self._history[self._history_index]
            event.prevent_default()
            return

        if key == "down":
            if self._completion_visible:
                # Navigate completion menu
                event.prevent_default()
                self.app.query_one("#completion", CompletionMenu).navigate(1)
                return
            if self._panel_visible:
                # CommandPanel owns Down
                return
            # History navigation
            if not self._history:
                return
            if self._history_index is None:
                return
            self._history_index += 1
            if self._history_index >= len(self._history):
                self._history_index = None
                self.text = self._saved_input
            else:
                self.text = self._history[self._history_index]
            event.prevent_default()
            return

        # Handle Escape
        if key == "escape":
            if self._panel_visible:
                # CommandPanel owns escape
                return
            if self._completion_visible:
                # Dismiss completion menu
                event.prevent_default()
                self.app.query_one("#completion", CompletionMenu).hide()
                return
            # Clear input
            self.text = ""
            event.prevent_default()
            return


class CompletionMenu(ListView):
    """Dropdown menu for slash-command completions.

    Built once at startup with all commands. Filter by showing/hiding items.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._commands: dict[str, str] = {}  # Maps item id to command name

    def set_commands(self, items: list[tuple[str, str]]) -> None:
        """Call once at mount time to populate the full list."""
        for idx, (cmd, desc) in enumerate(items):
            item_id = f"cmd{idx}"
            self._commands[item_id] = cmd
            label = f"[bold]{cmd}[/bold]  [dim]{desc}[/dim]"
            self.append(ListItem(Label(label), id=item_id))

    def filter(self, prefix: str) -> None:
        """Show/hide items by prefix, show menu if any match."""
        any_visible = False
        for item in self.query(ListItem):
            cmd = self._commands.get(item.id, "")
            visible = cmd.lstrip("/").startswith(prefix)
            item.display = visible
            if visible:
                any_visible = True
        if any_visible:
            self._select_first_visible()
            self.add_class("visible")
        else:
            self.remove_class("visible")

    def hide(self) -> None:
        """Hide the menu."""
        self.remove_class("visible")

    def get_selected(self) -> str | None:
        """Return the currently selected command name."""
        if self.highlighted_child is None or not self.highlighted_child.display:
            return None
        return self._commands.get(self.highlighted_child.id)

    def navigate(self, direction: int) -> None:
        """Move selection, skipping hidden items."""
        all_items = list(self.query(ListItem))
        visible_items = [i for i in all_items if i.display]
        if not visible_items:
            return
        if self.highlighted_child not in visible_items:
            # Select first visible if nothing selected
            self.index = all_items.index(visible_items[0])
            return
        idx = visible_items.index(self.highlighted_child)
        new_idx = max(0, min(idx + direction, len(visible_items) - 1))
        self.index = all_items.index(visible_items[new_idx])

    def _select_first_visible(self) -> None:
        """Move selection to first visible item."""
        for i, item in enumerate(self.query(ListItem)):
            if item.display:
                self.index = i
                return


class CommandPanel(Container):
    """Modal panel for config dialogs: choice (OptionList), multiselect (SelectionList), or text (Input).

    Shows one widget at a time, takes focus when visible, posts Dismissed message on completion.
    """

    class Dismissed(Message):
        def __init__(self, value: Any | None):
            self.value = value
            super().__init__()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_widget: OptionList | SelectionList | Input | None = None
        self._current_callback: Callable[[Any], None] | None = None

    async def show_choice(self, choices: list[str], current: str | None, label: str = "Select") -> None:
        """Show a single-select dialog with OptionList."""
        await self._clear_children()
        options = [Option(choice, choice) for choice in choices]
        widget = OptionList(*options)
        if current and current in choices:
            widget.highlighted = choices.index(current)
        self._current_widget = widget
        await self.mount(widget)
        self.add_class("visible")
        widget.focus()

    async def show_multiselect(self, choices: list[str], current: list[str], label: str = "Select") -> None:
        """Show a multi-select dialog with SelectionList."""
        await self._clear_children()
        items = [(choice, choice) for choice in choices]
        widget = SelectionList(*items)
        # Pre-select current items by toggling them
        for choice in choices:
            if choice in current:
                widget.toggle(choice)
        self._current_widget = widget
        await self.mount(widget)
        self.add_class("visible")
        widget.focus()

    async def show_text_input(self, current: str | None, nullable: bool = False, label: str = "Enter text") -> None:
        """Show a text input dialog with Input widget."""
        await self._clear_children()
        widget = Input(value=current or "")
        self._current_widget = widget
        await self.mount(widget)
        self.add_class("visible")
        widget.focus()

    async def show_toggle(self, current: bool, label: str = "Toggle") -> None:
        """Show a toggle dialog (on/off choice)."""
        choices = ["on", "off"]
        current_str = "on" if current else "off"
        await self.show_choice(choices, current_str, label)

    def hide(self) -> None:
        """Hide the panel."""
        self.remove_class("visible")

    async def _clear_children(self) -> None:
        """Remove all child widgets."""
        self._current_widget = None
        # Remove any existing child widgets
        for widget in list(self.children):
            await widget.remove()

    def _on_option_list_option_selected(self, event) -> None:
        """OptionList item selected."""
        value = event.option.prompt
        self.post_message(self.Dismissed(value))
        self.hide()

    def _on_input_submitted(self, event) -> None:
        """Input widget submitted (Enter pressed)."""
        value = event.value
        self.post_message(self.Dismissed(value))
        self.hide()

    def on_key(self, event) -> None:
        """Handle Escape and Enter for SelectionList."""
        if event.key == "escape":
            event.prevent_default()
            self.post_message(self.Dismissed(None))
            self.hide()
        elif event.key == "enter" and isinstance(self._current_widget, SelectionList):
            event.prevent_default()
            # Get selected items - selected property is a tuple of Selection objects
            selected_values = list(self._current_widget.selected)
            self.post_message(self.Dismissed(selected_values))
            self.hide()
