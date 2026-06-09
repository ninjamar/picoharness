from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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


@dataclass
class SessionInfo:
    name: str
    created_at: datetime
    first_user_message: str


# All event classes for deserialization
_EVENT_CLASSES = {
    cls.__name__: cls
    for cls in [
        UserInputEvent,
        ThinkingEvent,
        ResponseEvent,
        ToolStartEvent,
        ToolOutputEvent,
        ToolErrorEvent,
        DoneEvent,
    ]
}


def _serialize_event(event: Event) -> dict[str, Any]:
    """Serialize an event to a JSON-compatible dict."""
    data = dataclasses.asdict(event)
    data["type"] = type(event).__name__
    return data


def _deserialize_event(data: dict[str, Any]) -> Event:
    """Deserialize an event from a JSON-compatible dict."""
    event_type = data.pop("type", None)
    if not event_type or event_type not in _EVENT_CLASSES:
        raise ValueError(f"Unknown event type: {event_type}")
    cls = _EVENT_CLASSES[event_type]
    return cls(**{k: v for k, v in data.items() if k in {f.name for f in dataclasses.fields(cls)}})


class SessionManager:
    def __init__(self, save_dir: Path) -> None:
        self.save_dir = Path(save_dir).expanduser()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._current_session_name: str | None = None
        self._events: list[Event] = []

    def start_session(self, name: str) -> None:
        """Start a new session or resume an existing one."""
        self._current_session_name = name
        self._events = []
        session_file = self.save_dir / f"{name}.jsonl"

        # If resuming, file already exists — don't overwrite metadata
        if not session_file.exists():
            # Create new session: write metadata line
            metadata = {
                "name": name,
                "created_at": datetime.now().isoformat(),
            }
            with open(session_file, "w") as f:
                f.write(json.dumps(metadata) + "\n")

    def _write(self, event: Event, name: str) -> None:
        """Write an event to a session file."""
        session_file = self.save_dir / f"{name}.jsonl"
        with open(session_file, "a") as f:
            f.write(json.dumps(_serialize_event(event)) + "\n")

    def send(self, event: Event) -> None:
        """Stream-write an event to the current session file immediately."""
        if not self._current_session_name:
            return
        self._events.append(event)
        self._write(event, self._current_session_name)

    def load(self, name: str) -> list[Event]:
        """Load events from a JSONL session file."""
        session_file = self.save_dir / f"{name}.jsonl"
        if not session_file.exists():
            raise FileNotFoundError(f"Session '{name}' not found")

        events = []
        with open(session_file, "r") as f:
            lines = f.readlines()
            # First line is metadata, skip it
            for line in lines[1:]:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        events.append(_deserialize_event(data))
                    except (json.JSONDecodeError, ValueError):
                        # Skip malformed lines
                        continue

        # Repair: synthesize DoneEvent if missing and write to file
        self._current_session_name = name
        self.repair(events)
        self._current_session_name = None

        return events

    def repair(self, events: list[Event]) -> None:
        """Synthesize a DoneEvent if the session ends without one, and write it."""
        if not events or isinstance(events[-1], DoneEvent):
            return
        done = DoneEvent(id=events[-1].id, error=None, interrupted=True)
        events.append(done)
        if self._current_session_name:
            self._write(done, self._current_session_name)

    def finalize(self) -> None:
        """On app exit, repair the current session if needed."""
        self.repair(self._events)

    def list_sessions(self) -> list[SessionInfo]:
        """List all saved sessions."""
        sessions = []
        for session_file in sorted(self.save_dir.glob("*.jsonl"), reverse=True):
            try:
                with open(session_file, "r") as f:
                    # First line is metadata
                    metadata_line = f.readline().strip()
                    if not metadata_line:
                        continue
                    metadata = json.loads(metadata_line)
                    name = metadata.get("name", session_file.stem)
                    created_at_str = metadata.get("created_at", "")
                    created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now()

                    # Find first user message
                    first_user_msg = ""
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                event_data = json.loads(line)
                                if event_data.get("type") == "UserInputEvent":
                                    text = event_data.get("text", "")
                                    # Get first sentence
                                    for sent in text.split("."):
                                        sent = sent.strip()
                                        if sent:
                                            first_user_msg = sent[:100]
                                            break
                                    break
                            except json.JSONDecodeError:
                                continue

                    sessions.append(SessionInfo(name=name, created_at=created_at, first_user_message=first_user_msg))
            except Exception:
                continue

        return sorted(sessions, key=lambda s: s.created_at, reverse=True)

    def rename(self, old_name: str, new_name: str) -> None:
        """Rename a session file and update its metadata."""
        old_file = self.save_dir / f"{old_name}.jsonl"
        new_file = self.save_dir / f"{new_name}.jsonl"

        if not old_file.exists():
            raise FileNotFoundError(f"Session '{old_name}' not found")
        if new_file.exists():
            raise FileExistsError(f"Session '{new_name}' already exists")

        # Read all lines
        with open(old_file, "r") as f:
            lines = f.readlines()

        # Update metadata (first line)
        if lines:
            metadata = json.loads(lines[0])
            metadata["name"] = new_name
            lines[0] = json.dumps(metadata) + "\n"

        # Write to new file
        with open(new_file, "w") as f:
            f.writelines(lines)

        # Delete old file
        old_file.unlink()

        # Update current session name if renaming the active session
        if self._current_session_name == old_name:
            self._current_session_name = new_name

    def exists(self, name: str) -> bool:
        return (self.save_dir / f"{name}.jsonl").exists()
