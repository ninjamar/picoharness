import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Generator

import ollama
from openai import AsyncOpenAI


@dataclass
class ToolCallFunction:
    name: str
    arguments: dict[str, Any]


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
    async def chat(
        self, *, model: str, messages: list[dict[str, Any]], think: bool, tools_schemas: list[dict]
    ) -> AsyncGenerator[ChatResponse, None]:
        """Call the chat API with the given parameters."""
        raise NotImplementedError
        yield  # Need to satisfy AsyncGenerator type annotation


class OllamaProvider(BaseProvider):
    """Manages the Ollama AsyncClient connection."""

    def __init__(self) -> None:
        self.client = ollama.AsyncClient()

    async def chat(
        self, *, model: str, messages: list[dict[str, Any]], think: bool, tools_schemas: list[dict]
    ) -> AsyncGenerator[ChatResponse, None]:
        """Call the Ollama chat API with the given parameters."""
        async for part in await self.client.chat(
            model=model,
            messages=messages,
            stream=True,  # always Stream
            think=think,
            tools=tools_schemas,
        ):
            tool_calls = [
                ToolCall(function=ToolCallFunction(name=tc.function.name, arguments=dict(tc.function.arguments)))
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
    def __init__(self, base_url: str, api_key: str = "") -> None:
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def chat(
        self, *, model: str, messages: list[dict[str, Any]], think: bool, tools_schemas: list[dict]
    ) -> AsyncGenerator[ChatResponse, None]:
        accum: dict[int, dict[str, str]] = {}

        async for chunk in await self.client.chat.completions.create(  # type: ignore
            model=model,
            messages=messages,
            stream=True,
            tools=tools_schemas,
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
