from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from tarfile import data_filter
from typing import TYPE_CHECKING, Any

from backend.backend import Backend
from backend.events import *
from backend.provider import (
    BaseProvider,
    ModelInfo,
    OllamaProvider,
    OpenAICompatibleProvider,
)
from backend.tools import BaseTool, ReadFileTool
from backend.tools.web.browse import ReadWebPage, SearchAndReadWeb, SearchWeb
from backend.tools.web.wikipedia import GetWikipediaPage, SearchWikipedia

ALL_TOOLS: list[type[BaseTool]] = [
    ReadFileTool,
    ReadWebPage,
    SearchWeb,
    SearchAndReadWeb,
    SearchWikipedia,
    GetWikipediaPage,
]

__all__ = [
    "ALL_TOOLS",
    "BackendAPI",
    "ModelInfo",
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
class BackendAPI:
    provider: BaseProvider
    model: str
    think: bool
    context_length: int

    tool_classes: list[type[BaseTool]] | None
    system_prompt: str | None
    system_prompt_path: str | None

    _backend: Backend = field(init=False)

    # TODO: Investigate default
    def __post_init__(self):
        self._backend = Backend.from_config(self)

    async def __aenter__(self) -> BackendAPI:
        await self._backend.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._backend.__aexit__(*args)

    # Streaming proxies
    def feed(self, input_id: str, text: str) -> None:
        self._backend.feed(input_id, text)

    async def stream_events(self) -> AsyncGenerator[Event]:
        async for event in self._backend.stream_events():
            yield event

    def cancel_current(self) -> None:
        self._backend.cancel_current()

    # Setters
    def set_model(self, model: str) -> None:
        self.model = model
        self._backend._model = model

    def set_think(self, value: bool) -> None:
        self.think = value
        self._backend._think = value

    def set_provider(self, provider: BaseProvider) -> None:
        self.provider = provider
        self._backend._provider = provider

    def set_system_prompt_path(self, value: str | None) -> None:
        self.system_prompt_path = value
        self._backend._system_prompt_path = value

    def set_enabled_tools(self, value: list[str]) -> None:
        if value == ["*"]:
            new_classes = list(ALL_TOOLS)
        else:
            name_map = {t.name: t for t in ALL_TOOLS}
            new_classes = [name_map[n] for n in value if n in name_map]
        self.tool_classes = new_classes
        self._backend._tool_classes = new_classes
        self._backend._tool_instances = [cls() for cls in new_classes]
        self._backend._tool_schemas = [cls.to_schema() for cls in new_classes]
        self.provider.tool_schemas = self._backend._tool_schemas

    def set_context_length(self, value: int) -> None:
        self.context_length = value
        self.provider._context_length = value

    # Getters

    # @property
    # def provider(self) -> str:
    #     if isinstance(self._provider, OllamaProvider):
    #         return "ollama"
    #     elif isinstance(self._provider, OpenAICompatibleProvider):
    #         base_url = str(self._provider.client.base_url)
    #         return base_url.replace("/v1", "") if base_url.endswith("/v1") else base_url
    #     return "unknown"

    def get_provider_type(self) -> str:
        if isinstance(self.provider, OllamaProvider):
            return "ollama"
        elif isinstance(self.provider, OpenAICompatibleProvider):
            base_url = str(self.provider.client.base_url)
            return base_url.replace("/v1", "") if base_url.endswith("/v1") else base_url
        raise

    def get_enabled_tools(self) -> list[str]:
        return [t.name for t in self.tool_classes]

    def get_all_tools(self) -> list[str]:
        return [t.name for t in ALL_TOOLS]

    async def get_available_models(self) -> list[ModelInfo]:
        return await self.provider.list_models()
