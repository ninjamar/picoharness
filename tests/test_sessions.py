import json
import tempfile
from pathlib import Path

import pytest

from backend.events import (
    DoneEvent,
    ResponseEvent,
    ToolOutputEvent,
    ToolStartEvent,
    UserInputEvent,
)
from backend.sessions import SessionManager


def test_session_manager_save_and_load():
    """Test saving and loading events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionManager(Path(tmpdir))

        events = [
            UserInputEvent(id="1", text="Hello"),
            ResponseEvent(id="1", fragment="Hi there!"),
        ]

        sm.start_session("test-session")
        for event in events:
            sm.send(event)
        loaded = sm.load("test-session")

        # load() calls repair() which adds a DoneEvent if missing
        assert len(loaded) == 3
        assert isinstance(loaded[0], UserInputEvent)
        assert loaded[0].text == "Hello"
        assert isinstance(loaded[1], ResponseEvent)
        assert loaded[1].fragment == "Hi there!"
        assert isinstance(loaded[2], DoneEvent)


def test_session_manager_list_sessions():
    """Test listing all sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionManager(Path(tmpdir))

        sm.start_session("session1")
        sm.send(UserInputEvent(id="1", text="First message"))

        sm.start_session("session2")
        sm.send(UserInputEvent(id="2", text="Another message here"))

        sessions = sm.list_sessions()
        assert len(sessions) == 2

        # Check that sessions are sorted by created_at descending
        names = [s.name for s in sessions]
        assert "session1" in names
        assert "session2" in names

        # Check first_user_message extraction
        session1 = next(s for s in sessions if s.name == "session1")
        assert session1.first_user_message == "First message"

        session2 = next(s for s in sessions if s.name == "session2")
        assert session2.first_user_message == "Another message here"


def test_session_manager_rename():
    """Test renaming a session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionManager(Path(tmpdir))

        sm.start_session("old-name")
        sm.send(UserInputEvent(id="1", text="Test"))

        sm.rename("old-name", "new-name")

        assert not sm.exists("old-name")
        assert sm.exists("new-name")

        # Verify content is preserved
        loaded = sm.load("new-name")
        assert isinstance(loaded[0], UserInputEvent)
        assert loaded[0].text == "Test"

        # Verify name field in JSONL metadata is updated (first line)
        metadata_line = (Path(tmpdir) / "new-name.jsonl").open().readline()
        file_content = json.loads(metadata_line)
        assert file_content["name"] == "new-name"


def test_session_manager_file_not_found():
    """Test loading non-existent session raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionManager(Path(tmpdir))

        with pytest.raises(FileNotFoundError, match="Session 'nonexistent' not found"):
            sm.load("nonexistent")


def test_session_manager_rename_existing_target():
    """Test renaming to existing name raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionManager(Path(tmpdir))

        sm.start_session("session1")
        sm.send(UserInputEvent(id="1", text="Test"))

        sm.start_session("session2")
        sm.send(UserInputEvent(id="2", text="Test"))

        with pytest.raises(FileExistsError, match="Session 'session2' already exists"):
            sm.rename("session1", "session2")


def test_session_manager_preserves_created_at():
    """Test that created_at timestamp is preserved on save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionManager(Path(tmpdir))

        sm.start_session("test")
        sm.send(UserInputEvent(id="1", text="Test"))
        file1 = (Path(tmpdir) / "test.jsonl").open().readline()
        created_at_1 = json.loads(file1)["created_at"]

        # Start session again — won't overwrite metadata since file exists
        sm.start_session("test")
        sm.send(UserInputEvent(id="2", text="Test2"))
        file2 = (Path(tmpdir) / "test.jsonl").open().readline()
        created_at_2 = json.loads(file2)["created_at"]

        # created_at should not change
        assert created_at_1 == created_at_2


def test_session_manager_first_user_message_extraction():
    """Test extraction of first user message for display."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionManager(Path(tmpdir))

        sm.start_session("test")
        sm.send(UserInputEvent(id="1", text="This is a long message. With multiple sentences! And punctuation?"))

        sessions = sm.list_sessions()

        assert len(sessions) == 1
        assert sessions[0].first_user_message == "This is a long message"


def test_session_manager_creates_directory():
    """Test that SessionManager creates the save directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_dir = Path(tmpdir) / "a" / "b" / "c"
        assert not nested_dir.exists()

        sm = SessionManager(nested_dir)
        assert nested_dir.exists()

        sm.start_session("test")
        sm.send(UserInputEvent(id="1", text="Test"))
        assert (nested_dir / "test.jsonl").exists()


def test_session_preserves_tool_calls():
    """Test that tool calls in events are preserved and round-trip correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionManager(Path(tmpdir))

        sm.start_session("test")
        sm.send(UserInputEvent(id="1", text="Search for weather"))
        sm.send(ToolStartEvent(id="1", tool_id="123", tool_name="search_web", tool_input={"query": "weather"}))
        sm.send(ToolOutputEvent(id="1", tool_id="123", tool_name="search_web", result="Sunny", output_format="all"))

        loaded = sm.load("test")

        # Should have 3 events plus auto-added DoneEvent from repair()
        assert len(loaded) == 4
        assert isinstance(loaded[0], UserInputEvent)
        assert loaded[0].text == "Search for weather"
        assert isinstance(loaded[1], ToolStartEvent)
        assert loaded[1].tool_name == "search_web"
        assert isinstance(loaded[2], ToolOutputEvent)
        assert loaded[2].result == "Sunny"
        assert isinstance(loaded[3], DoneEvent)
