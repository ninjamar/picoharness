from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from backend.provider.provider import OllamaProvider, OpenAICompatibleProvider


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
        yield OllamaProvider(tools=[])


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
        yield OpenAICompatibleProvider(base_url="http://localhost:8000", tools=[])


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
