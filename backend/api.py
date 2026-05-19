from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from backend.backend import Backend

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


class BackendAPI:
    """
    Single interface between frontend and backend.

    Owns all configuration state. Frontend holds only a BackendAPI reference —
    never a Backend directly.
    """

    def __init__(
        self,
        *,
        provider: BaseProvider,
        model: str,
        think: bool = False,
        tool_classes: list[type[BaseTool]] | None = None,
        system_prompt: str | None = None,
        system_prompt_path: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._think = think
        self._system_prompt = system_prompt
        self._system_prompt_path = system_prompt_path
        self._tool_classes: list[type[BaseTool]] = list(tool_classes or [])

        # Eagerly create the Backend — no lazy init
        from backend.backend import Backend

        self._backend: Backend = Backend.from_config(self)

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> BackendAPI:
        await self._backend.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._backend.__aexit__(*args)

    # ── Streaming proxies ─────────────────────────────────────────────────────

    def feed(self, input_id: str, text: str) -> None:
        self._backend.feed(input_id, text)

    async def stream_events(self) -> AsyncGenerator[Event, None]:
        async for event in self._backend.stream_events():
            yield event

    def cancel_current(self) -> None:
        self._backend.cancel_current()

    # ── Config properties ─────────────────────────────────────────────────────

    @property
    def model(self) -> str:
        return self._model

    @property
    def think(self) -> bool:
        return self._think

    @property
    def provider(self) -> str:
        if isinstance(self._provider, OllamaProvider):
            return "ollama"
        elif isinstance(self._provider, OpenAICompatibleProvider):
            base_url = str(self._provider.client.base_url)
            return base_url.replace("/v1", "") if base_url.endswith("/v1") else base_url
        return "unknown"

    @property
    def system_prompt_path(self) -> str | None:
        return self._system_prompt_path

    @property
    def enabled_tools(self) -> list[str]:
        return [t.name for t in self._tool_classes]

    # ── Config setters (keep running Backend in sync) ─────────────────────────

    def set_model(self, model: str) -> None:
        self._model = model
        self._backend._model = model

    def set_think(self, value: bool) -> None:
        self._think = value
        self._backend._think = value

    def set_provider(self, provider: BaseProvider) -> None:
        self._provider = provider
        self._backend._provider = provider

    def set_system_prompt_path(self, value: str | None) -> None:
        self._system_prompt_path = value
        self._backend._system_prompt_path = value

    def set_enabled_tools(self, value: list[str]) -> None:
        if value == ["*"]:
            new_classes = list(ALL_TOOLS)
        else:
            name_map = {t.name: t for t in ALL_TOOLS}
            new_classes = [name_map[n] for n in value if n in name_map]
        self._tool_classes = new_classes
        self._backend._tool_classes = new_classes
        self._backend._tool_instances = [cls() for cls in new_classes]
        self._backend._tool_schemas = [cls.to_schema() for cls in new_classes]
        self._provider.tool_schemas = self._backend._tool_schemas

    # ── Query methods ─────────────────────────────────────────────────────────

    def get_all_tools(self) -> list[str]:
        return [t.name for t in ALL_TOOLS]

    async def get_available_models(self) -> list[ModelInfo]:
        return await self._provider.list_models()
