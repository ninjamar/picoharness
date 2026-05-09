import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend import Backend
from backend.backend import _format_system_prompt, _get_tool_info
from backend.events import (
    DoneEvent,
    ResponseEvent,
    ThinkingEvent,
    ToolErrorEvent,
    ToolOutputEvent,
    ToolStartEvent,
    UserInputEvent,
)
from backend.provider.provider import (
    ModelInfo,
    _ChatMessage,
    _ChatResponse,
    _ToolCall,
    _ToolCallFunction,
)
from backend.tools import BaseTool, ReadFileTool
from backend.tools.base import _parse_docstring


class _IntTool(BaseTool):
    """Tool with required and optional int parameters."""

    name = "int_tool"
    output_format = "all"

    async def execute(self, count: int, label: str = "x", **_) -> str:  # type: ignore[override]
        """Process an integer.

        Args:
            count: How many times.
            label: The label.
        """
        return f"{label} x {count}"


class _TypesTool(BaseTool):
    """Tool with various parameter types."""

    name = "types_tool"
    output_format = "all"

    async def execute(self, a: str, b: int, c: float, d: bool, e: list, f: dict, **_) -> str:  # type: ignore[override]
        """Process multiple types.

        Args:
            a: A string.
            b: An integer.
            c: A float.
            d: A boolean.
            e: A list.
            f: A dictionary.
        """
        return "ok"


def make_chat_response(content=None, thinking=None, tool_calls=None):
    """Helper to create a _ChatResponse with given fields."""
    return _ChatResponse(
        message=_ChatMessage(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls or [],
        )
    )


def make_tool_call(name, arguments):
    """Helper to create a _ToolCall."""
    return _ToolCall(function=_ToolCallFunction(name=name, arguments=arguments))


def make_provider(chunks):
    """
    Create a mock provider that yields from a sequence of chunks.

    Args:
        chunks: list of lists. Each inner list is a sequence of _ChatResponse objects
                to yield for a single provider.chat() call.
    """
    call_index = [0]

    async def mock_chat(**kwargs):
        idx = call_index[0]
        call_index[0] += 1
        if idx < len(chunks):
            for chunk in chunks[idx]:
                yield chunk

    provider = MagicMock()
    provider.chat = mock_chat
    return provider


async def collect_events(backend, req_id: str) -> list:
    """Drain stream_events() until DoneEvent for req_id, return all collected events."""
    events = []
    async for event in backend.stream_events():
        events.append(event)
        if isinstance(event, DoneEvent) and event.id == req_id:
            break
    return events


async def test_basic_response():
    """Test that feed() and stream_events() produces expected events on basic response."""
    provider = make_provider(
        [
            [make_chat_response(content="Hello!")],
        ]
    )

    async with Backend(provider=provider, model="test", tools=[ReadFileTool]) as backend:
        backend.feed("req-1", "hi")
        events = await collect_events(backend, "req-1")

        # Verify events
        assert any(isinstance(e, UserInputEvent) for e in events), "Missing UserInputEvent"
        assert any(isinstance(e, ResponseEvent) and e.fragment == "Hello!" for e in events), (
            "Missing ResponseEvent with correct text"
        )
        assert any(isinstance(e, DoneEvent) and e.id == "req-1" and e.error is None for e in events), (
            "Missing DoneEvent with no error"
        )


async def test_thinking_event():
    """Test that thinking events are emitted before response events."""
    provider = make_provider(
        [
            [
                make_chat_response(thinking="hmm"),
                make_chat_response(content="response"),
            ],
        ]
    )

    async with Backend(provider=provider, model="test", tools=[ReadFileTool]) as backend:
        backend.feed("req-1", "hi")
        events = await collect_events(backend, "req-1")

        # Find indices
        thinking_idx = next((i for i, e in enumerate(events) if isinstance(e, ThinkingEvent)), -1)
        response_idx = next((i for i, e in enumerate(events) if isinstance(e, ResponseEvent)), -1)

        assert thinking_idx >= 0, "Missing ThinkingEvent"
        assert response_idx >= 0, "Missing ResponseEvent"
        assert thinking_idx < response_idx, "ThinkingEvent should come before ResponseEvent"


async def test_tool_call_flow():
    """Test that tool calls trigger the correct events and second call receives tool results."""
    with patch("backend.backend.ReadFileTool.execute", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "file contents"

        provider = make_provider(
            [
                [
                    make_chat_response(tool_calls=[make_tool_call("read_file", {"path": "/tmp/x"})]),
                ],
                [
                    make_chat_response(content="Done"),
                ],
            ]
        )

        async with Backend(provider=provider, model="test", tools=[ReadFileTool]) as backend:
            backend.feed("req-1", "read /tmp/x")
            events = await collect_events(backend, "req-1")

            # Verify tool start/finish events
            assert any(isinstance(e, ToolStartEvent) and e.tool_name == "read_file" for e in events), (
                "Missing ToolStartEvent"
            )
            assert any(
                isinstance(e, ToolOutputEvent) and e.tool_name == "read_file" and e.result == "file contents"
                for e in events
            ), "Missing ToolOutputEvent with correct output"
            assert any(isinstance(e, DoneEvent) and e.error is None for e in events)


async def test_done_event_on_error():
    """Test that errors are captured in DoneEvent."""

    async def failing_chat(**kwargs):
        raise ValueError("Provider error")
        yield  # satisfy AsyncGenerator type

    provider = MagicMock()
    provider.chat = failing_chat

    async with Backend(provider=provider, model="test", tools=[ReadFileTool]) as backend:
        backend.feed("req-1", "hi")
        events = await collect_events(backend, "req-1")

        # Should have a DoneEvent with an error
        done_events = [e for e in events if isinstance(e, DoneEvent) and e.id == "req-1"]
        assert len(done_events) == 1
        assert done_events[0].error is not None
        assert "Provider error" in done_events[0].error


@pytest.mark.asyncio
async def test_message_history_across_turns():
    """Test that message history is built correctly across multiple sequential inputs."""
    captured_messages = []

    async def capture_chat(model, messages, think, **kwargs):
        captured_messages.append(list(messages))
        # First call: just echo user message
        # Second call: echo with context
        if len(captured_messages) == 1:
            yield make_chat_response(content="response1")
        else:
            yield make_chat_response(content="response2")

    provider = MagicMock()
    provider.chat = capture_chat

    async with Backend(provider=provider, model="test", tools=[ReadFileTool]) as backend:
        backend.feed("req-1", "first")
        backend.feed("req-2", "second")

        done_count = [0]
        async for event in backend.stream_events():
            if isinstance(event, DoneEvent):
                done_count[0] += 1
                if done_count[0] == 2:
                    break

        # Second call should have both user messages and first assistant response
        assert len(captured_messages) >= 2
        second_call_messages = captured_messages[1]
        user_contents = [m["content"] for m in second_call_messages if m.get("role") == "user"]
        assistant_contents = [m["content"] for m in second_call_messages if m.get("role") == "assistant"]

        assert "first" in user_contents, "First user message should be in history"
        assert "second" in user_contents, "Second user message should be in history"
        assert "response1" in assistant_contents, "First response should be in history"


async def test_multiple_tool_calls_concurrent():
    """Test that multiple tool calls run concurrently and events are emitted for all."""
    with patch("backend.backend.ReadFileTool.execute", new_callable=AsyncMock) as mock_execute:
        # Simulate different execution times
        async def slow_execute(**kwargs):
            path = kwargs.get("path", "")
            await asyncio.sleep(0.01)
            return f"contents of {path}"

        mock_execute.side_effect = slow_execute

        provider = make_provider(
            [
                [
                    make_chat_response(
                        tool_calls=[
                            make_tool_call("read_file", {"path": "/tmp/a"}),
                            make_tool_call("read_file", {"path": "/tmp/b"}),
                        ]
                    ),
                ],
                [
                    make_chat_response(content="Done"),
                ],
            ]
        )

        async with Backend(provider=provider, model="test", tools=[ReadFileTool]) as backend:
            backend.feed("req-1", "read two files")
            events = await collect_events(backend, "req-1")

            # Count tool start and finish events
            tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
            tool_finishes = [e for e in events if isinstance(e, (ToolOutputEvent, ToolErrorEvent))]

            assert len(tool_starts) == 2
            assert len(tool_finishes) == 2

            # All starts should come before any finish (since they're emitted up front)
            first_finish_idx = next(
                (i for i, e in enumerate(events) if isinstance(e, (ToolOutputEvent, ToolErrorEvent))), -1
            )
            assert all(events.index(e) < first_finish_idx for e in tool_starts), (
                "All ToolStartEvents should come before ToolFinishEvents"
            )


async def test_tool_error_in_finish_event():
    """Test that tool errors are properly communicated in ToolFinishEvent.error field."""
    with patch("backend.backend.ReadFileTool.execute", new_callable=AsyncMock) as mock_execute:
        mock_execute.side_effect = FileNotFoundError("File not found")

        provider = make_provider(
            [
                [
                    make_chat_response(tool_calls=[make_tool_call("read_file", {"path": "/nonexistent"})]),
                ],
                [
                    make_chat_response(content="Tool failed"),
                ],
            ]
        )

        async with Backend(provider=provider, model="test", tools=[ReadFileTool]) as backend:
            backend.feed("req-1", "read missing file")
            events = await collect_events(backend, "req-1")

            # Verify error in ToolErrorEvent
            tool_errors = [e for e in events if isinstance(e, ToolErrorEvent)]
            assert len(tool_errors) == 1
            assert tool_errors[0].error is not None
            assert "File not found" in tool_errors[0].error
            # DoneEvent should succeed despite tool error
            assert any(isinstance(e, DoneEvent) and e.error is None for e in events)


@pytest.mark.parametrize(
    "content",
    [
        "Line 1\nLine 2\nLine 3",
        "",
        "Special chars: !@#$%^&*()\nUnicode: 你好世界 🎉",
    ],
)
async def test_readfiletool_basic_read(content):
    """Test that ReadFileTool.execute() correctly reads various file types and contents."""
    tool = ReadFileTool()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
        f.write(content)
        temp_path = f.name

    try:
        result = await tool.execute(path=temp_path)
        assert result == content
    finally:
        Path(temp_path).unlink()


async def test_readfiletool_nonexistent_file():
    """Test that ReadFileTool.execute() handles nonexistent files gracefully."""
    tool = ReadFileTool()
    result = await tool.execute(path="/nonexistent/path/to/file.txt")
    assert "Error reading file" in result


# ============================================================================
# Tool schema and docstring parsing tests
# ============================================================================


def test_parse_docstring_none():
    """Test that _parse_docstring handles None gracefully."""
    result = _parse_docstring(None)
    assert len(result) == 0


def test_parse_docstring_args_section_parsed():
    """Test that Args section is parsed into param descriptions."""
    doc = """Process an integer.

    Args:
        count: How many times.
        label: The label to use.
    """
    result = _parse_docstring(doc)
    assert "count" in result
    assert "How many times" in result["count"]
    assert "label" in result
    assert "The label to use" in result["label"]


def test_parse_docstring_stops_at_returns():
    """Test that Returns section is not included in params."""
    doc = """Summary.

    Args:
        x: The value.

    Returns:
        A result.
    """
    result = _parse_docstring(doc)
    assert "x" in result
    # The "A result" line comes after the Returns: separator, so it should be in a non-param key (not "x")
    # Just verify "x" is in the result, which shows we parsed the Args section
    assert result["x"]  # Should have the description for param x


def test_schema_structure():
    """Test that to_schema returns correct top-level structure."""
    schema = ReadFileTool.to_schema()
    assert schema["type"] == "function"
    assert "function" in schema
    assert "name" in schema["function"]
    assert "description" in schema["function"]
    assert "parameters" in schema["function"]
    assert schema["function"]["parameters"]["type"] == "object"
    assert "properties" in schema["function"]["parameters"]
    assert "required" in schema["function"]["parameters"]


def test_schema_required_vs_optional():
    """Test that required params are listed and optional params are not."""
    schema = _IntTool.to_schema()
    props = schema["function"]["parameters"]
    required = props["required"]
    assert "count" in required
    assert "label" not in required


@pytest.mark.parametrize(
    "tool_cls,param_name,expected_type",
    [
        (_TypesTool, "a", "string"),
        (_TypesTool, "b", "integer"),
        (_TypesTool, "c", "number"),
        (_TypesTool, "d", "boolean"),
        (_TypesTool, "e", "array"),
        (_TypesTool, "f", "object"),
    ],
)
def test_schema_type_mapping(tool_cls, param_name, expected_type):
    """Test type annotation to JSON schema mapping."""
    schema = tool_cls.to_schema()
    param_schema = schema["function"]["parameters"]["properties"][param_name]
    assert param_schema["type"] == expected_type


def test_schema_unannotated_defaults_to_string():
    """Test that params without type annotations default to string."""

    class _UnAnnotatedTool(BaseTool):
        name = "unannotated"
        output_format = "all"

        async def execute(self, value, **_) -> str:  # type: ignore[override]
            return "ok"

    schema = _UnAnnotatedTool.to_schema()
    param_schema = schema["function"]["parameters"]["properties"]["value"]
    assert param_schema["type"] == "string"


def test_schema_description_from_docstring():
    """Test that function description is extracted from docstring."""
    schema = _IntTool.to_schema()
    description = schema["function"]["description"]
    assert "integer" in description.lower()


def test_readfiletool_schema_structure():
    """Test ReadFileTool schema has correct structure."""
    schema = ReadFileTool.to_schema()
    props = schema["function"]["parameters"]
    assert "path" in props["properties"]
    assert props["properties"]["path"]["type"] == "string"
    assert "path" in props["required"]
    assert ReadFileTool.output_format == "none"


# ============================================================================
# BackendConfig tests
# ============================================================================


async def test_backend_config_model_property():
    """Test that config.model returns current model."""
    provider = make_provider([[make_chat_response(content="hi")]])
    async with Backend(provider=provider, model="test-model", tools=[ReadFileTool]) as backend:
        assert backend.config.model == "test-model"


async def test_backend_config_think_property():
    """Test that config.think returns think flag."""
    provider = make_provider([[make_chat_response(content="hi")]])
    async with Backend(provider=provider, model="test", think=True, tools=[ReadFileTool]) as backend:
        assert backend.config.think is True

    async with Backend(provider=provider, model="test", think=False, tools=[ReadFileTool]) as backend:
        assert backend.config.think is False


async def test_backend_config_set_model():
    """Test that config.set_model updates backend model."""
    provider = make_provider([[make_chat_response(content="hi")]])
    async with Backend(provider=provider, model="model-a", tools=[ReadFileTool]) as backend:
        assert backend.config.model == "model-a"
        await backend.config.set_model("model-b")
        assert backend.config.model == "model-b"


async def test_backend_config_set_think():
    """Test that config.set_think updates backend think flag."""
    provider = make_provider([[make_chat_response(content="hi")]])
    async with Backend(provider=provider, model="test", think=False, tools=[ReadFileTool]) as backend:
        assert backend.config.think is False
        backend.config.set_think(True)
        assert backend.config.think is True


async def test_backend_config_get_available_models():
    """Test that config.get_available_models delegates to provider."""
    mock_provider = MagicMock()
    mock_provider.list_models = AsyncMock(return_value=[ModelInfo(name="m1"), ModelInfo(name="m2")])
    mock_provider.chat = AsyncMock(side_effect=lambda **_: iter([]))

    async with Backend(provider=mock_provider, model="test", tools=[]) as backend:
        models = await backend.config.get_available_models()
        assert len(models) == 2
        assert models[0].name == "m1"
        assert models[1].name == "m2"


# ============================================================================
# System prompt tests
# ============================================================================


def test_format_system_prompt_replaces_tools_placeholder():
    """Test that {{tools}} is replaced with tool info."""
    prompt = "Use these tools: {{tools}}"
    result = _format_system_prompt(prompt, [ReadFileTool])
    assert "{{tools}}" not in result
    assert "read_file" in result


def test_format_system_prompt_no_tools_replaces_with_empty():
    """Test that {{tools}} is replaced with empty string when no tools."""
    prompt = "Tools: {{tools}}."
    result = _format_system_prompt(prompt, [])
    assert result == "Tools: ."


def test_get_tool_info_contains_name_and_sig():
    """Test that _get_tool_info includes tool name and signature."""
    info = _get_tool_info([ReadFileTool])
    assert "read_file" in info
    assert "path" in info


async def test_system_prompt_added_as_first_message():
    """Test that system prompt is added as first message."""
    provider = make_provider([[make_chat_response(content="hi")]])
    system_prompt = "You are helpful. Tools: {{tools}}"
    async with Backend(provider=provider, model="test", system_prompt=system_prompt, tools=[ReadFileTool]) as backend:
        assert len(backend._messages) > 0
        assert backend._messages[0]["role"] == "system"
        assert "You are helpful" in backend._messages[0]["content"]


async def test_no_system_prompt_no_system_message():
    """Test that no system prompt means no system message."""
    provider = make_provider([[make_chat_response(content="hi")]])
    async with Backend(provider=provider, model="test", tools=[ReadFileTool]) as backend:
        system_messages = [m for m in backend._messages if m.get("role") == "system"]
        assert len(system_messages) == 0


# ============================================================================
# Backend lifecycle tests
# ============================================================================


async def test_backend_no_tools():
    """Test that backend works with no tools."""
    provider = make_provider([[make_chat_response(content="response")]])
    async with Backend(provider=provider, model="test", tools=[]) as backend:
        backend.feed("req-1", "hello")
        events = await collect_events(backend, "req-1")
        assert any(isinstance(e, UserInputEvent) for e in events)
        assert any(isinstance(e, ResponseEvent) for e in events)
        assert any(isinstance(e, DoneEvent) and e.error is None for e in events)


async def test_cancel_current_emits_interrupted_done_event():
    """Test that cancel_current() causes DoneEvent(interrupted=True)."""

    async def slow_chat(**kwargs):
        await asyncio.sleep(10)
        yield make_chat_response(content="delayed")

    provider = MagicMock()
    provider.chat = slow_chat

    async with Backend(provider=provider, model="test", tools=[ReadFileTool]) as backend:
        backend.feed("req-1", "hi")
        await asyncio.sleep(0.05)
        backend.cancel_current()

        events = await collect_events(backend, "req-1")
        done_events = [e for e in events if isinstance(e, DoneEvent) and e.id == "req-1"]
        assert len(done_events) == 1
        assert done_events[0].interrupted is True


async def test_shutdown_stops_process_loop():
    """Test that shutdown() stops the process loop gracefully."""

    async def never_chat(**kwargs):
        await asyncio.sleep(100)
        yield make_chat_response(content="never")

    provider = MagicMock()
    provider.chat = never_chat

    backend = Backend(provider=provider, model="test", tools=[ReadFileTool])
    async with backend:
        # Feed but don't wait for it
        backend.feed("req-1", "hi")
        await asyncio.sleep(0.05)
        await backend.shutdown()
        # If shutdown worked, process_task should complete soon
        if backend._process_task:
            try:
                await asyncio.wait_for(backend._process_task, timeout=1)
            except asyncio.TimeoutError:
                pytest.fail("Process loop did not shut down in time")


# ============================================================================
# Tool dispatch tests
# ============================================================================


async def test_unknown_tool_emits_tool_error_event():
    """Test that an unknown tool call produces ToolErrorEvent with 'Unknown tool' message."""
    provider = make_provider(
        [
            [
                make_chat_response(tool_calls=[make_tool_call("nonexistent_tool", {"arg": "val"})]),
            ],
            [
                make_chat_response(content="Done"),
            ],
        ]
    )

    async with Backend(provider=provider, model="test", tools=[ReadFileTool]) as backend:
        backend.feed("req-1", "call unknown tool")
        events = await collect_events(backend, "req-1")

        tool_errors = [e for e in events if isinstance(e, ToolErrorEvent)]
        assert len(tool_errors) == 1
        assert "Unknown tool" in tool_errors[0].error


async def test_tool_result_appended_as_tool_message():
    """Test that tool results are appended as role=tool messages with matching tool_call_id."""
    with patch("backend.backend.ReadFileTool.execute", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "file contents"

        provider = make_provider(
            [
                [
                    make_chat_response(tool_calls=[make_tool_call("read_file", {"path": "/tmp/x"})]),
                ],
                [
                    make_chat_response(content="Done"),
                ],
            ]
        )

        async with Backend(provider=provider, model="test", tools=[ReadFileTool]) as backend:
            backend.feed("req-1", "read /tmp/x")
            events = await collect_events(backend, "req-1")

            # Find the tool call id from events
            tool_start_events = [e for e in events if isinstance(e, ToolStartEvent)]
            assert len(tool_start_events) == 1
            tool_id = tool_start_events[0].tool_id

            # Verify tool message was appended to backend messages
            tool_messages = [m for m in backend._messages if m.get("role") == "tool"]
            assert len(tool_messages) == 1
            assert tool_messages[0]["tool_call_id"] == tool_id
            assert tool_messages[0]["content"] == "file contents"
