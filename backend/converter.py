from __future__ import annotations

from typing import Any

from backend.events import (
    DoneEvent,
    Event,
    ResponseEvent,
    ThinkingEvent,
    ToolErrorEvent,
    ToolOutputEvent,
    ToolStartEvent,
    UserInputEvent,
)


class EventToMessageConverter:
    """Converts Events to OpenAI message format.

    Stateful: accumulates fragments into complete messages.
    """

    def __init__(self) -> None:
        self.pending_response = ""
        self.pending_tool_calls: list[dict[str, Any]] = []

    def feed(self, event: Event) -> dict[str, Any] | None:
        """Feed an event, return complete message if one finalizes, else None."""
        match event:
            case UserInputEvent(text=text):
                return {"role": "user", "content": text}

            case ResponseEvent(fragment=fragment):
                self.pending_response += fragment
                return None

            case ToolStartEvent(tool_id=tool_id, tool_name=name, tool_input=inp):
                # Finalize any pending response first
                msg = self._finalize()
                # Add this tool call to pending
                self.pending_tool_calls.append(
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {"name": name, "arguments": inp},
                    }
                )
                return msg

            case ToolOutputEvent(tool_id=tool_id, result=result):
                return {"role": "tool", "tool_call_id": tool_id, "content": result}

            case ToolErrorEvent(tool_id=tool_id, error=error):
                return {"role": "tool", "tool_call_id": tool_id, "content": f"Error: {error}"}

            case ThinkingEvent():
                return None

            case DoneEvent():
                return self._finalize()

    def feed_all(self, events: list[Event]) -> list[dict[str, Any]]:
        """Batch convert events to messages."""
        messages = []
        for event in events:
            if msg := self.feed(event):
                messages.append(msg)
        if msg := self._finalize():
            messages.append(msg)
        return messages

    def _finalize(self) -> dict[str, Any] | None:
        """Complete any pending state and return message if exists."""
        if self.pending_response or self.pending_tool_calls:
            msg: dict[str, Any] = {"role": "assistant", "content": self.pending_response}
            if self.pending_tool_calls:
                msg["tool_calls"] = self.pending_tool_calls
            self.pending_response = ""
            self.pending_tool_calls = []
            return msg
        return None
