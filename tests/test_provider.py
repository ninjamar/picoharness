from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from backend.provider.provider import (
    ModelInfo,
    OllamaProvider,
    OpenAICompatibleProvider,
)


def make_ollama_chunk(content=None, thinking=None, tool_calls=None):
    """Helper to create a fake Ollama response chunk."""
    chunk = MagicMock()
    chunk.message.content = content
    chunk.message.thinking = thinking
    chunk.message.tool_calls = tool_calls or []
    return chunk


def make_ollama_tool_call(name, arguments):
    """Helper to create a fake Ollama tool call."""
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def make_openai_chunk(content=None, reasoning_content=None, tool_calls=None):
    """Helper to create a fake OpenAI delta chunk."""
    delta = MagicMock()
    delta.content = content
    delta.reasoning_content = reasoning_content
    delta.reasoning = None
    delta.thinking = None
    delta.tool_calls = tool_calls or []

    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=delta)]
    return chunk


def make_openai_tool_call_delta(index, name=None, arguments=None):
    """Helper to create a fake OpenAI tool call delta."""
    tc = MagicMock()
    tc.index = index
    if name:
        tc.function.name = name
    if arguments:
        tc.function.arguments = arguments
    return tc


@contextmanager
def make_ollama_provider(chunks):
    """Create a mock OllamaProvider that streams given chunks."""
    with patch("ollama.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        async def async_gen(**_kwargs):
            for chunk in chunks:
                yield chunk

        mock_client.chat = AsyncMock(side_effect=lambda **kwargs: async_gen(**kwargs))
        yield OllamaProvider()


@contextmanager
def make_openai_provider(chunks):
    """Create a mock OpenAICompatibleProvider that streams given chunks."""
    with patch("backend.provider.provider.AsyncOpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        async def async_gen(**_kwargs):
            for chunk in chunks:
                yield chunk

        mock_client.chat.completions.create = AsyncMock(side_effect=lambda **kwargs: async_gen(**kwargs))
        yield OpenAICompatibleProvider(base_url="http://localhost:8000")


async def test_ollama_streams_content():
    """Test that OllamaProvider correctly yields content from streamed chunks."""
    chunks = [
        make_ollama_chunk(content="Hello "),
        make_ollama_chunk(content="world!"),
    ]

    with make_ollama_provider(chunks) as provider:
        responses = []
        async for response in provider.chat(model="test", messages=[], think=False):
            responses.append(response)

        assert len(responses) == 2
        assert responses[0].message.content == "Hello "
        assert responses[1].message.content == "world!"


async def test_ollama_streams_thinking():
    """Test that OllamaProvider correctly yields thinking from streamed chunks."""
    chunks = [
        make_ollama_chunk(thinking="Let me think"),
        make_ollama_chunk(content="The answer is"),
    ]

    with make_ollama_provider(chunks) as provider:
        responses = []
        async for response in provider.chat(model="test", messages=[], think=True):
            responses.append(response)

        assert len(responses) == 2
        assert responses[0].message.thinking == "Let me think"
        assert responses[1].message.content == "The answer is"


async def test_ollama_streams_tool_calls():
    """Test that OllamaProvider correctly yields tool calls."""
    tool_calls = [make_ollama_tool_call("read_file", {"path": "/tmp/x"})]
    chunks = [
        make_ollama_chunk(tool_calls=tool_calls),
    ]

    with make_ollama_provider(chunks) as provider:
        responses = []
        async for response in provider.chat(model="test", messages=[], think=False):
            responses.append(response)

        assert len(responses) == 1
        assert len(responses[0].message.tool_calls) == 1
        assert responses[0].message.tool_calls[0].function.name == "read_file"
        assert responses[0].message.tool_calls[0].function.arguments == {"path": "/tmp/x"}


async def test_openai_streams_content():
    """Test that OpenAICompatibleProvider correctly yields content from streamed chunks."""
    chunks = [
        make_openai_chunk(content="Hello "),
        make_openai_chunk(content="world!"),
    ]

    with make_openai_provider(chunks) as provider:
        responses = []
        async for response in provider.chat(model="test", messages=[], think=False):
            responses.append(response)

        assert len(responses) == 2
        assert responses[0].message.content == "Hello "
        assert responses[1].message.content == "world!"


async def test_openai_streams_thinking():
    """Test that OpenAICompatibleProvider correctly yields thinking from streamed chunks."""
    chunks = [
        make_openai_chunk(reasoning_content="Thinking..."),
        make_openai_chunk(content="Answer"),
    ]

    with make_openai_provider(chunks) as provider:
        responses = []
        async for response in provider.chat(model="test", messages=[], think=True):
            responses.append(response)

        assert len(responses) == 2
        assert responses[0].message.thinking == "Thinking..."
        assert responses[1].message.content == "Answer"


async def test_openai_assembles_tool_calls():
    """Test that OpenAICompatibleProvider correctly assembles tool call arguments across chunks."""
    # Tool call arguments come in fragments
    chunks = [
        make_openai_chunk(
            tool_calls=[
                make_openai_tool_call_delta(index=0, name="read_file", arguments='{"path":'),
            ]
        ),
        make_openai_chunk(
            tool_calls=[
                make_openai_tool_call_delta(index=0, arguments=' "/tmp/x"}'),
            ]
        ),
    ]

    with make_openai_provider(chunks) as provider:
        responses = []
        async for response in provider.chat(model="test", messages=[], think=False):
            responses.append(response)

        # OpenAI provider emits tool calls in a final response after all fragments are assembled
        assert len(responses) >= 1
        final_response = responses[-1]
        assert len(final_response.message.tool_calls) == 1
        assert final_response.message.tool_calls[0].function.name == "read_file"
        assert final_response.message.tool_calls[0].function.arguments == {"path": "/tmp/x"}


# ============================================================================
# OllamaProvider additional tests
# ============================================================================


async def test_ollama_list_models():
    """Test that OllamaProvider.list_models returns correctly structured ModelInfo."""
    with patch("ollama.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the list() response
        mock_model = MagicMock()
        mock_model.model = "llama2"
        mock_model.details.parameter_size = "7B"
        mock_model.details.quantization_level = "Q4"

        mock_client.list = AsyncMock(return_value=MagicMock(models=[mock_model]))

        provider = OllamaProvider()
        models = await provider.list_models()

        assert len(models) == 1
        assert models[0].name == "llama2"
        assert models[0].parameter_size == "7B"
        assert models[0].quantization_level == "Q4"


async def test_ollama_passes_tool_schemas():
    """Test that OllamaProvider passes tool_schemas to the client."""
    with patch("ollama.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        chunk = make_ollama_chunk(content="hi")

        async def async_gen(**kwargs):
            yield chunk

        mock_client.chat = AsyncMock(side_effect=lambda **kwargs: async_gen(**kwargs))

        provider = OllamaProvider()
        provider.tool_schemas = [{"type": "function", "function": {"name": "test_tool"}}]

        responses = []
        async for response in provider.chat(model="test", messages=[], think=False):
            responses.append(response)

        # Verify chat was called with tools kwarg
        mock_client.chat.assert_called_once()
        call_kwargs = mock_client.chat.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == provider.tool_schemas


async def test_ollama_none_tool_calls_no_crash():
    """Test that OllamaProvider handles None tool_calls gracefully."""
    chunks = [
        make_ollama_chunk(content="response", tool_calls=None),
    ]

    with make_ollama_provider(chunks) as provider:
        responses = []
        async for response in provider.chat(model="test", messages=[], think=False):
            responses.append(response)

        assert len(responses) == 1
        assert responses[0].message.content == "response"
        assert responses[0].message.tool_calls == []


async def test_ollama_think_flag_passed():
    """Test that OllamaProvider passes think flag to client."""
    with patch("ollama.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        chunk = make_ollama_chunk(content="response")

        async def async_gen(**kwargs):
            yield chunk

        mock_client.chat = AsyncMock(side_effect=lambda **kwargs: async_gen(**kwargs))

        provider = OllamaProvider()
        responses = []
        async for response in provider.chat(model="test", messages=[], think=True):
            responses.append(response)

        # Verify think=True was passed to client
        mock_client.chat.assert_called_once()
        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs.get("think") is True


# ============================================================================
# OpenAICompatibleProvider additional tests
# ============================================================================


async def test_openai_list_models():
    """Test that OpenAICompatibleProvider.list_models returns correctly structured ModelInfo."""
    with patch("backend.provider.provider.AsyncOpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_model = MagicMock()
        mock_model.id = "gpt-4"

        mock_client.models.list = AsyncMock(return_value=MagicMock(data=[mock_model]))

        provider = OpenAICompatibleProvider(base_url="http://localhost:8000")
        models = await provider.list_models()

        assert len(models) == 1
        assert models[0].name == "gpt-4"


async def test_openai_reasoning_field_maps_to_thinking():
    """Test that OpenAI 'reasoning' field is mapped to thinking."""

    def make_openai_chunk_with_reasoning(reasoning_text):
        delta = MagicMock()
        delta.content = None
        delta.reasoning_content = None
        delta.reasoning = reasoning_text
        delta.thinking = None
        delta.tool_calls = []

        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=delta)]
        return chunk

    chunks = [
        make_openai_chunk_with_reasoning("Let me reason about this"),
    ]

    with make_openai_provider(chunks) as provider:
        responses = []
        async for response in provider.chat(model="test", messages=[], think=True):
            responses.append(response)

        assert len(responses) >= 1
        thinking_responses = [r for r in responses if r.message.thinking]
        assert len(thinking_responses) >= 1
        assert "Let me reason" in thinking_responses[0].message.thinking


async def test_openai_thinking_field_maps_to_thinking():
    """Test that OpenAI 'thinking' field is mapped to thinking."""

    def make_openai_chunk_with_thinking(thinking_text):
        delta = MagicMock()
        delta.content = None
        delta.reasoning_content = None
        delta.reasoning = None
        delta.thinking = thinking_text
        delta.tool_calls = []

        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=delta)]
        return chunk

    chunks = [
        make_openai_chunk_with_thinking("Processing information..."),
    ]

    with make_openai_provider(chunks) as provider:
        responses = []
        async for response in provider.chat(model="test", messages=[], think=True):
            responses.append(response)

        assert len(responses) >= 1
        thinking_responses = [r for r in responses if r.message.thinking]
        assert len(thinking_responses) >= 1
        assert "Processing information" in thinking_responses[0].message.thinking


async def test_openai_think_true_sets_reasoning_effort_high():
    """Test that think=True sets reasoning_effort='high'."""
    with patch("backend.provider.provider.AsyncOpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        chunk = make_openai_chunk(content="result")

        async def async_gen(**kwargs):
            yield chunk

        mock_client.chat.completions.create = AsyncMock(side_effect=lambda **kwargs: async_gen(**kwargs))

        provider = OpenAICompatibleProvider(base_url="http://localhost:8000")
        responses = []
        async for response in provider.chat(model="test", messages=[], think=True):
            responses.append(response)

        # Verify reasoning_effort='high' was passed
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("reasoning_effort") == "high"


async def test_openai_think_false_sets_reasoning_effort_none():
    """Test that think=False sets reasoning_effort='none'."""
    with patch("backend.provider.provider.AsyncOpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        chunk = make_openai_chunk(content="result")

        async def async_gen(**kwargs):
            yield chunk

        mock_client.chat.completions.create = AsyncMock(side_effect=lambda **kwargs: async_gen(**kwargs))

        provider = OpenAICompatibleProvider(base_url="http://localhost:8000")
        responses = []
        async for response in provider.chat(model="test", messages=[], think=False):
            responses.append(response)

        # Verify reasoning_effort='none' was passed
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("reasoning_effort") == "none"


async def test_openai_serializes_dict_tool_call_arguments():
    """Test that dict tool call arguments are JSON-serialized before sending."""
    with patch("backend.provider.provider.AsyncOpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        chunk = make_openai_chunk(content="done")

        async def async_gen(**kwargs):
            yield chunk

        mock_client.chat.completions.create = AsyncMock(side_effect=lambda **kwargs: async_gen(**kwargs))

        provider = OpenAICompatibleProvider(base_url="http://localhost:8000")
        messages = [
            {
                "role": "assistant",
                "content": "I'll read the file",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": {"path": "/tmp/x"}},
                    }
                ],
            }
        ]

        responses = []
        async for response in provider.chat(model="test", messages=messages, think=False):
            responses.append(response)

        # Verify arguments were serialized
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        sent_messages = call_kwargs.get("messages", [])

        # Find the assistant message with tool_calls
        assistant_msg = next((m for m in sent_messages if m.get("role") == "assistant"), None)
        assert assistant_msg is not None
        assert len(assistant_msg.get("tool_calls", [])) > 0
        tool_call = assistant_msg["tool_calls"][0]
        # Arguments should be a JSON string, not a dict
        assert isinstance(tool_call["function"]["arguments"], str)
        assert '"/tmp/x"' in tool_call["function"]["arguments"]


async def test_openai_multiple_tool_calls_assembled():
    """Test that multiple parallel tool calls (different indices) are assembled correctly."""
    chunks = [
        make_openai_chunk(
            tool_calls=[
                make_openai_tool_call_delta(index=0, name="tool1", arguments='{"a":'),
                make_openai_tool_call_delta(index=1, name="tool2", arguments='{"b":'),
            ]
        ),
        make_openai_chunk(
            tool_calls=[
                make_openai_tool_call_delta(index=0, arguments=" 1}"),
                make_openai_tool_call_delta(index=1, arguments=" 2}"),
            ]
        ),
    ]

    with make_openai_provider(chunks) as provider:
        responses = []
        async for response in provider.chat(model="test", messages=[], think=False):
            responses.append(response)

        # Final response should have both tool calls
        final_response = responses[-1]
        assert len(final_response.message.tool_calls) == 2

        # Find tool1 and tool2
        tool_calls = {tc.function.name: tc for tc in final_response.message.tool_calls}
        assert "tool1" in tool_calls
        assert "tool2" in tool_calls
        assert tool_calls["tool1"].function.arguments == {"a": 1}
        assert tool_calls["tool2"].function.arguments == {"b": 2}


async def test_openai_null_tool_calls_delta_no_crash():
    """Test that null tool_calls in delta doesn't cause crash."""
    chunks = [
        make_openai_chunk(content="response", tool_calls=None),
    ]

    with make_openai_provider(chunks) as provider:
        responses = []
        async for response in provider.chat(model="test", messages=[], think=False):
            responses.append(response)

        assert len(responses) >= 1
        assert responses[0].message.content == "response"
