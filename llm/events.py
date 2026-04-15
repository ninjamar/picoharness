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
class ToolEvent:
    name: str
    input: dict
    output: str
