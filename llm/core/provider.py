import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import ollama
from openai import AsyncOpenAI


@dataclass
class ToolCallFunction:
    name: str
    arguments: dict


@dataclass
class ToolCall:
    function: ToolCallFunction


@dataclass
class ChatMessage:
    content: str | None = None
    thinking: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ChatResponse:
    message: ChatMessage


class BaseProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    async def chat(self, *, model, messages, stream, think, tools):
        """Call the chat API with the given parameters."""
        pass


class OllamaProvider(BaseProvider):
    """Manages the Ollama AsyncClient connection."""

    def __init__(self):
        self.client = ollama.AsyncClient()

    async def chat(self, *, model, messages, stream, think, tools):
        """Call the Ollama chat API with the given parameters."""
        async for part in await self.client.chat(
            model=model,
            messages=messages,
            stream=stream,
            think=think,
            tools=tools,
        ):
            tool_calls = [
                ToolCall(function=ToolCallFunction(name=tc.function.name, arguments=tc.function.arguments))
                for tc in (part.message.tool_calls or [])
            ]
            yield ChatResponse(
                message=ChatMessage(
                    content=part.message.content,
                    thinking=part.message.thinking,
                    tool_calls=tool_calls,
                )
            )


class OpenAICompatibleProvider(BaseProvider):
    """Manages OpenAI-compatible API client connection."""

    def __init__(self, base_url: str, api_key: str = ""):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def chat(self, *, model, messages, stream, think, tools):
        """Call the OpenAI-compatible chat API with the given parameters."""
        accum: dict[int, dict] = {}

        async for chunk in await self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            tools=tools,
            reasoning_effort="high" if think else "none",
        ):
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            thinking = getattr(delta, "reasoning_content", None) or getattr(delta, "thinking", None)
            if thinking:
                yield ChatResponse(message=ChatMessage(thinking=thinking))

            if delta.content:
                yield ChatResponse(message=ChatMessage(content=delta.content))

            for tc in delta.tool_calls or []:
                if tc.index not in accum:
                    accum[tc.index] = {"name": tc.function.name, "arguments": ""}
                accum[tc.index]["arguments"] += tc.function.arguments or ""

        if accum:
            tool_calls = [
                ToolCall(function=ToolCallFunction(name=v["name"], arguments=json.loads(v["arguments"])))
                for v in accum.values()
            ]
            yield ChatResponse(message=ChatMessage(tool_calls=tool_calls))
