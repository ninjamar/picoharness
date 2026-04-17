from dataclasses import dataclass


@dataclass
class ThinkingEvent:
    """A fragment of the model's internal reasoning."""

    text: str


@dataclass
class ResponseEvent:
    """A fragment of the model's visible reply."""

    text: str


@dataclass
class ToolStartEvent:
    """A tool execution has been kicked off."""

    id: str
    name: str
    input: dict


@dataclass
class ToolEndEvent:
    """A tool execution has completed."""

    id: str
    output: str
