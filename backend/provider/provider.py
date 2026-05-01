import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import ollama
from openai import AsyncOpenAI

from backend.tools import BaseTool
from backend import events

@dataclass
class _ToolCallFunction:
    name: str
    arguments: dict[str, Any]


@dataclass
class _ToolCall:
    function: _ToolCallFunction


@dataclass
class _ChatMessage:
    content: str | None = None
    thinking: str | None = None
    tool_calls: list[_ToolCall] = field(default_factory=list)


@dataclass
class _ChatResponse:
    message: _ChatMessage


class BaseProvider(ABC):
    def __init__(self, tools: list[type[BaseTool]]) -> None:
        self.tools_schemas = [tool.to_schema() for tool in tools]

    @abstractmethod
    async def chat(
        self, model: str, messages: list[dict[str, Any]], think: bool
    ) -> AsyncGenerator[_ChatResponse, None]:
        raise NotImplementedError
        yield  # Need to satisfy AsyncGenerator type annotation

    # async def stream_events(self, *, id, model: str, messages: list[dict[str, Any]], think: bool) -> AsyncGenerator[Event, None]:
    #     async for part in self._chat(model, messages, think):
    #         # part is a chat response
    #         msg = part.message
    #         if msg.content is not None:
    #             yield events.ResponseEvent(id, msg.content, None)
    #         if msg.thinking is not None:
    #             yield events.ThinkingEvent(id, msg.thinking, None)
    #         if len(msg.tool_calls) > 0:
    #             pass


class OllamaProvider(BaseProvider):
    def __init__(self, tools: list[type[BaseTool]]) -> None:
        super().__init__(tools)
        self.client = ollama.AsyncClient()

    async def chat(
        self, model: str, messages: list[dict[str, Any]], think: bool
    ) -> AsyncGenerator[_ChatResponse, None]:
        async for part in await self.client.chat(
            model=model,
            messages=messages,
            stream=True,  # always Stream
            think=think,
            tools=self.tools_schemas,
        ):
            tool_calls = [
                _ToolCall(function=_ToolCallFunction(name=tc.function.name, arguments=dict(tc.function.arguments)))
                for tc in (part.message.tool_calls or [])
            ]
            yield _ChatResponse(
                message=_ChatMessage(
                    content=part.message.content,
                    thinking=part.message.thinking,
                    tool_calls=tool_calls,
                )
            )


class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, base_url: str, tools: list[type[BaseTool]], api_key: str = "") -> None:
        super().__init__(tools)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def chat(
        self, model: str, messages: list[dict[str, Any]], think: bool
    ) -> AsyncGenerator[_ChatResponse, None]:

        serialized_messages = []
        for message in messages:
            if "tool_calls" in message:
                msg_copy = dict(message)
                msg_copy["tool_calls"] = [
                    {
                        **tc,
                        "function": {
                            **tc["function"],
                            "arguments": (
                                tc["function"]["arguments"]
                                if isinstance(tc["function"]["arguments"], str)
                                else json.dumps(tc["function"]["arguments"])
                            ),
                        },
                    }
                    for tc in message["tool_calls"]
                ]
                serialized_messages.append(msg_copy)
            else:
                serialized_messages.append(message)

        accum: dict[int, dict[str, str]] = {}

        async for chunk in await self.client.chat.completions.create(  # type: ignore
            model=model,
            messages=serialized_messages,
            stream=True,
            tools=self.tools_schemas,
            reasoning_effort="high" if think else "none",
        ):
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            # print(delta)
            thinking = (
                getattr(delta, "reasoning_content", None)
                or getattr(delta, "reasoning", None)
                or getattr(delta, "thinking", None)
            )
            if thinking:
                yield _ChatResponse(message=_ChatMessage(thinking=thinking))

            if delta.content:
                yield _ChatResponse(message=_ChatMessage(content=delta.content))

            for tc in delta.tool_calls or []:  # delta.tool_calls could be None
                if tc.index not in accum:
                    accum[tc.index] = {"name": tc.function.name, "arguments": ""}
                accum[tc.index]["arguments"] += tc.function.arguments or ""

        if accum:
            tool_calls = [
                _ToolCall(function=_ToolCallFunction(name=v["name"], arguments=json.loads(v["arguments"])))
                for v in accum.values()
            ]
            yield _ChatResponse(message=_ChatMessage(tool_calls=tool_calls))
