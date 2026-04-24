from dataclasses import dataclass


@dataclass
class Event:
    pass


@dataclass
class UserInputEvent(Event):
    """User input into model"""

    text: str


@dataclass
class ThinkingEvent(Event):
    """A fragment of the model's internal reasoning."""

    text: str


@dataclass
class ResponseEvent(Event):
    """A fragment of the model's visible reply."""

    text: str


@dataclass
class ToolStartEvent(Event):
    """A tool execution has been kicked off."""

    id: str
    name: str
    input: dict


@dataclass
class ToolEndEvent(Event):
    """A tool execution has completed."""

    id: str
    output: str
