from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from backend.agent import Agent
from backend.backend import Backend
from backend.converter import EventToMessageConverter
from backend.events import *
from backend.provider import (
    BaseProvider,
    ModelInfo,
    OllamaProvider,
    OpenAICompatibleProvider,
)
from backend.sessions import SessionManager
from backend.tools import BaseTool, ReadFileTool
from backend.tools.web.browse import (
    ReadWebPage,
    SearchAndReadWeb,
    SearchAndSummarizeWeb,
    SearchWeb,
    SummarizeWebPage,
)
from backend.tools.web.wikipedia import GetWikipediaPage, SearchWikipedia

ALL_TOOLS: list[type[BaseTool]] = [
    ReadFileTool,
    ReadWebPage,
    SearchWeb,
    SearchAndReadWeb,
    SummarizeWebPage,
    SearchAndSummarizeWeb,
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

    searxng_url: str
    jina_reader_url: str

    session_manager: SessionManager

    api_key: str = ""
    summarizer_model: str | None = None

    _backend: Backend = field(init=False)

    def __post_init__(self):
        # Initialize SessionManager with default path if not provided
        if self.session_manager is None:
            self.session_manager = SessionManager(Path.home() / ".ph" / "sessions")
        self._backend = Backend.from_config(self)

    def set_session_save_location(self, path: str) -> None:
        self.session_manager = SessionManager(Path(path).expanduser())

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
            self.session_manager.send(event)
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

    def set_searxng_url(self, value: str) -> None:
        self.searxng_url = value
        self._backend._tool_config.searxng_url = value
        self._backend._reinstantiate_tools()

    def set_jina_reader_url(self, value: str) -> None:
        self.jina_reader_url = value
        self._backend._tool_config.jina_reader_url = value
        self._backend._reinstantiate_tools()

    def set_summarizer_model(self, value: str | None) -> None:
        self.summarizer_model = value
        if value:
            fresh_provider = Backend._build_summarizer_provider(self.provider, self.api_key)
            self._backend._summarizer_agent = Agent.for_single_task(fresh_provider, value)
        else:
            self._backend._summarizer_agent = None
        self._backend._reinstantiate_tools()

    def set_enabled_tools(self, value: list[str]) -> None:
        if value == ["*"]:
            new_classes = list(ALL_TOOLS)
        else:
            name_map = {t.name: t for t in ALL_TOOLS}
            new_classes = [name_map[n] for n in value if n in name_map]
        self.tool_classes = new_classes
        self._backend._tool_classes = new_classes
        self._backend._reinstantiate_tools()
        self._backend._tool_schemas = [cls.to_schema() for cls in new_classes]
        self.provider.tool_schemas = self._backend._tool_schemas

    def set_context_length(self, value: int) -> None:
        self.context_length = value
        self.provider.context_length = value
        self._backend.messages.max_size = int(value * 0.75)

    def set_api_key(self, value: str | None) -> None:
        self.api_key = value or ""
        if isinstance(self.provider, OpenAICompatibleProvider):
            self.provider.client = AsyncOpenAI(base_url=str(self.provider.client.base_url), api_key=self.api_key)

    # Getters

    def get_provider_type(self) -> str:
        if isinstance(self.provider, OllamaProvider):
            return "ollama"
        elif isinstance(self.provider, OpenAICompatibleProvider):
            base_url = str(self.provider.client.base_url)
            return base_url.replace("/v1", "") if base_url.endswith("/v1") else base_url
        raise Exception("Invalid provider type")

    def get_enabled_tools(self) -> list[str]:
        return [t.name for t in self.tool_classes]

    def get_all_tools(self) -> list[str]:
        return [t.name for t in ALL_TOOLS]

    async def get_available_models(self) -> list[ModelInfo]:
        return await self.provider.list_models()

    @property
    def context_window(self) -> tuple[int, int, int]:
        """Returns (actual_tokens, total_messages, context_length)."""
        m = self._backend.messages
        return m.actual_size, self.context_length, m.total_size

    def rename_session(self, old_name: str, new_name: str) -> None:
        self.session_manager.rename(old_name, new_name)

    async def resume_session(self, name: str) -> list[Event]:
        """Load a session, reconstruct context, and return events for UI replay."""
        events = await asyncio.to_thread(self.session_manager.load, name)

        # Convert events to messages and inject into backend context
        converter = EventToMessageConverter()
        messages = converter.feed_all(events)
        for msg in messages:
            self._backend.messages.append(msg)

        # Start collecting new events for this session
        self.session_manager.start_session(name)

        return events

    def start_session(self, name: str | None = None) -> None:
        """Start a new session or continue an existing one."""
        if name is None:
            # Generate a default name for new sessions
            name = f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.session_manager.start_session(name)

    def finalize(self):
        self.session_manager.finalize()
