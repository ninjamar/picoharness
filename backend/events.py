from dataclasses import dataclass

__all__ = ["Event", "UserInputEvent", "ThinkingEvent", "ResponseEvent", "ToolStartEvent", "ToolFinishEvent"]
@dataclass
class Event:
    id: str # ID of sequence
    text: str | None # Text
    error: str | None # Error message

@dataclass
class UserInputEvent(Event):
    # Pass your id and message (through text)
    pass

@dataclass
class ThinkingEvent(Event):
    # Model thinks in text
    pass

@dataclass
class ResponseEvent(Event):
    # Response in section text
    pass


@dataclass
class ToolStartEvent(Event):
    tool_id: str # To match with tool finish event
    tool_name: str
    tool_input: dict


@dataclass
class ToolFinishEvent(Event):
    tool_id: str
    tool_name: str
    tool_output: dict
