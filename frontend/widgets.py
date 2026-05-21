from __future__ import annotations

from typing import Any, Callable

from textual import events
from textual.containers import Container
from textual.message import Message
from textual.widgets import (
    Input,
    Label,
    ListItem,
    ListView,
    OptionList,
    SelectionList,
    Static,
    TextArea,
)
from textual.widgets.selection_list import Selection


class CompletionMenu(ListView):
    """Slash-command completion menu shown above the input."""

    def show_completions(self, items: list[tuple[str, str]]) -> None:
        """Update list with (command, description) pairs and make visible."""
        self.clear()
        for cmd, desc in items:
            self.append(ListItem(Label(f"[bold]{cmd}[/bold]  [dim]{desc}[/dim]"), name=cmd))
        if items:
            self.add_class("visible")
            self.index = 0
        else:
            self.remove_class("visible")

    def hide(self) -> None:
        self.remove_class("visible")
        self.clear()

    def get_selected_command(self) -> str | None:
        if not self.has_class("visible"):
            return None
        item = self.highlighted_child
        if item is None:
            return None
        return item.name


class CommandPanel(Container):
    """Interactive command UI panel. Renders OptionList, SelectionList, or Input based on command type."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_widget: OptionList | SelectionList | Input | None = None
        self._callback: Callable[[Any], None] | None = None

    async def show_choice_dialog(self, choices: list[str], current: str, callback: Callable[[str], None]) -> None:
        """Show OptionList for DialogueMenu."""
        self._callback = callback
        self.remove_children()
        widget = OptionList(*choices)
        self._current_widget = widget
        self.add_class("visible")
        try:
            idx = choices.index(current)
            widget.highlighted = idx
        except ValueError, IndexError:
            pass
        await self.mount(widget)
        widget.focus()

    async def show_multiselect_dialog(
        self, choices: list[str], current: list[str], callback: Callable[[list[str]], None]
    ) -> None:
        """Show SelectionList for MultiSelectMenu."""
        self._callback = callback
        self.remove_children()
        selections = [Selection(c, c, c in current) for c in choices]
        widget = SelectionList(*selections)
        self._current_widget = widget
        self.add_class("visible")
        await self.mount(widget)
        widget.focus()

    async def show_text_input_dialog(self, current: str | None, callback: Callable[[str | None], None]) -> None:
        """Show Input for TextInputMenu."""
        self._callback = callback
        self.remove_children()
        value = current or ""
        widget = Input(value=value)
        self._current_widget = widget
        self.add_class("visible")
        await self.mount(widget)
        widget.focus()

    def hide(self) -> None:
        self.remove_class("visible")
        self._current_widget = None
        self._callback = None

    def _on_input_submitted(self, msg: Input.Submitted) -> None:
        if self._callback:
            self._callback(msg.value)
        self.hide()

    def _on_option_list_option_selected(self, msg: OptionList.OptionSelected) -> None:
        if self._callback:
            self._callback(msg.option.prompt)
        self.hide()

    def _on_selection_list_selected_changed(self, msg: SelectionList.SelectedChanged) -> None:
        pass

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter" and isinstance(self._current_widget, SelectionList):
            event.prevent_default()
            if self._callback:
                self._callback(list(self._current_widget.selected))
            self.hide()
        elif event.key == "escape":
            event.prevent_default()
            self.hide()


class InputOverlay(Container):
    """Container that manages CompletionMenu and CommandPanel mutual exclusivity."""

    def compose(self) -> Any:
        yield CompletionMenu(id="completion")
        yield CommandPanel(id="command-panel")

    async def on_mount(self) -> None:
        try:
            completion = self.query_one("#completion", CompletionMenu)
            completion.remove_class("visible")
        except Exception:
            pass
        try:
            command = self.query_one("#command-panel", CommandPanel)
            command.remove_class("visible")
        except Exception:
            pass

    def show_completions(self, matches: list[tuple[str, str]]) -> None:
        """Show CompletionMenu, hide CommandPanel."""
        try:
            command = self.query_one("#command-panel", CommandPanel)
            command.hide()
        except Exception:
            pass
        try:
            completion = self.query_one("#completion", CompletionMenu)
            completion.show_completions(matches)
        except Exception:
            pass

    async def show_command(
        self,
        menu_type: str,
        choices: list[str],
        current: Any,
        nullable: bool,
        callback: Callable[[Any], None],
    ) -> None:
        """Show CommandPanel with appropriate widget, hide CompletionMenu."""
        try:
            completion = self.query_one("#completion", CompletionMenu)
            completion.hide()
        except Exception:
            pass
        try:
            command = self.query_one("#command-panel", CommandPanel)
            if menu_type == "choice":
                await command.show_choice_dialog(choices, current, callback)
            elif menu_type == "multiselect":
                await command.show_multiselect_dialog(choices, current, callback)
            elif menu_type == "text":
                await command.show_text_input_dialog(current, callback)
        except Exception:
            pass

    def hide(self) -> None:
        """Hide both menus."""
        try:
            completion = self.query_one("#completion", CompletionMenu)
            completion.hide()
        except Exception:
            pass
        try:
            command = self.query_one("#command-panel", CommandPanel)
            command.hide()
        except Exception:
            pass

    def get_completion_menu(self) -> CompletionMenu | None:
        try:
            return self.query_one("#completion", CompletionMenu)
        except Exception:
            return None

    def get_command_panel(self) -> CommandPanel | None:
        try:
            return self.query_one("#command-panel", CommandPanel)
        except Exception:
            return None


class InputArea(TextArea):
    """TextArea that submits on Enter, supports history recall and completion navigation."""

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    class TextChanged(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._history_index: int = -1
        self._saved_input: str = ""

    def load_history(self, history: list[str]) -> None:
        self._history = history

    async def _on_key(self, event: events.Key) -> None:
        menu = self._get_completion_menu()

        if event.key == "enter":
            event.prevent_default()
            command_panel = self._get_command_panel()
            if command_panel is not None and command_panel.has_class("visible"):
                return
            if menu is not None and menu.has_class("visible"):
                cmd = menu.get_selected_command()
                if cmd:
                    self._apply_completion(cmd)
                    menu.hide()
                return
            if text := self.text.strip():
                self._history_index = -1
                self._saved_input = ""
                self.post_message(self.Submitted(text))
                self.clear()

        elif event.key == "tab":
            event.prevent_default()
            if menu is not None and menu.has_class("visible"):
                cmd = menu.get_selected_command()
                if cmd:
                    self._apply_completion(cmd)
                    menu.hide()

        elif event.key == "escape":
            event.prevent_default()
            command_panel = self._get_command_panel()
            if command_panel is not None and command_panel.has_class("visible"):
                command_panel.hide()
            elif menu is not None and menu.has_class("visible"):
                menu.hide()

        elif event.key == "up":
            if menu is not None and menu.has_class("visible"):
                event.prevent_default()
                idx = menu.index or 0
                if idx > 0:
                    menu.index = idx - 1
            else:
                event.prevent_default()
                self._history_up()

        elif event.key == "down":
            if menu is not None and menu.has_class("visible"):
                event.prevent_default()
                idx = menu.index or 0
                if idx < len(menu.children) - 1:
                    menu.index = idx + 1
            else:
                event.prevent_default()
                self._history_down()

        elif event.key == "ctrl+j":
            event.prevent_default()
            self.insert("\n")
            self.post_message(self.TextChanged(self.text))

        else:
            # Let TextArea handle it, then notify of change
            self.call_after_refresh(self._notify_changed)

    def _notify_changed(self) -> None:
        self.post_message(self.TextChanged(self.text))

    def _apply_completion(self, cmd: str) -> None:
        self.clear()
        self.insert(cmd + " ")
        self.post_message(self.TextChanged(self.text))

    def _history_up(self) -> None:
        if not self._history:
            return
        if self._history_index == -1:
            self._saved_input = self.text
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        self._load_history_entry()

    def _history_down(self) -> None:
        if self._history_index == -1:
            return
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self._load_history_entry()
        else:
            self._history_index = -1
            self.clear()
            if self._saved_input:
                self.insert(self._saved_input)

    def _load_history_entry(self) -> None:
        self.clear()
        self.insert(self._history[self._history_index])

    def _get_completion_menu(self) -> CompletionMenu | None:
        try:
            overlay = self.app.query_one(InputOverlay)
            return overlay.get_completion_menu()
        except Exception:
            return None

    def _get_command_panel(self) -> CommandPanel | None:
        try:
            overlay = self.app.query_one(InputOverlay)
            return overlay.get_command_panel()
        except Exception:
            return None
