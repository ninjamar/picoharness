from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "Event",
    "UserInputEvent",
    "ThinkingEvent",
    "ResponseEvent",
    "ToolStartEvent",
    "ToolOutputEvent",
    "ToolErrorEvent",
    "DoneEvent",
]


@dataclass
class Event:
    id: str


@dataclass
class UserInputEvent(Event):
    text: str


@dataclass
class ThinkingEvent(Event):
    fragment: str


@dataclass
class ResponseEvent(Event):
    fragment: str


@dataclass
class ToolStartEvent(Event):
    tool_id: str
    tool_name: str
    tool_input: dict


@dataclass
class ToolOutputEvent(Event):
    tool_id: str
    tool_name: str
    result: str
    output_format: Literal["all", "truncate", "none"]


@dataclass
class ToolErrorEvent(Event):
    tool_id: str
    tool_name: str
    error: str


@dataclass
class DoneEvent(Event):
    error: str | None
    interrupted: bool  # if user interrupted or not
