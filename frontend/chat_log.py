from __future__ import annotations

from backend.api import (
    DoneEvent,
    ResponseEvent,
    ThinkingEvent,
    ToolErrorEvent,
    ToolOutputEvent,
    ToolStartEvent,
    UserInputEvent,
)

MAX_TOOL_OUTPUT = 500


def _fmt_tool_input(inp: dict | str) -> str:
    if not isinstance(inp, dict):
        return repr(inp)
    items = list(inp.items())
    if len(items) == 1:
        return repr(items[0][1])
    return ", ".join(f"{k}={v!r}" for k, v in items)


class ChatLog:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._response_buf: str = ""
        self._thinking_buf: str = ""
        self._last_event_type: str | None = None

    def append(self, text: str) -> None:
        self.lines.append(text)

    def process(self, event, show_thinking: bool) -> bool:
        match event:
            case UserInputEvent(text=t):
                self._response_buf = ""
                self._last_event_type = None
                self._thinking_buf = ""
                self.lines.append(f"> {t}")

            case ThinkingEvent(fragment=f):
                if not show_thinking:
                    return False
                if self._last_event_type != "thinking":
                    self._thinking_buf = ""
                    self.lines.append("")
                self._last_event_type = "thinking"
                self._thinking_buf += f
                if self.lines:
                    self.lines[-1] = f"💭 {self._thinking_buf}"

            case ResponseEvent(fragment=f):
                if self._last_event_type != "response":
                    self._response_buf = ""
                    self.lines.append("")
                self._last_event_type = "response"
                self._response_buf += f
                if self.lines:
                    self.lines[-1] = f"{self._response_buf}"

            case ToolStartEvent(tool_name=name, tool_input=inp):
                self._response_buf = ""
                self._last_event_type = None
                self.lines.append(f"⏺ {name}({_fmt_tool_input(inp)})")

            case ToolOutputEvent(result=result, output_format=fmt):
                match fmt:
                    case "all":
                        text = result
                    case "truncate":
                        if len(result) > MAX_TOOL_OUTPUT:
                            result = result[:MAX_TOOL_OUTPUT] + "… [truncated]"
                        text = "\n".join(f"  {line}" for line in result.splitlines())
                    case _:
                        return False
                self.lines.append(text)

            case ToolErrorEvent(tool_name=name, error=err):
                self.lines.append(f"  {name}: {err}")

            case DoneEvent(error=error):
                self._response_buf = ""
                self._last_event_type = None
                self._thinking_buf = ""
                if error:
                    self.lines.append(f"Error: {error}")
                return True

        return False
