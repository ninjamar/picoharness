import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from backend.events import (
    DoneEvent,
    Event,
    ResponseEvent,
    ThinkingEvent,
    ToolFinishEvent,
    ToolStartEvent,
    UserInputEvent,
)
from backend.provider.provider import BaseProvider
from backend.tools import BaseTool, ReadFileTool

ALLOWED_TOOLS: list[type[BaseTool]] = [ReadFileTool]

_SENTINEL = object()


class Backend:
    def __init__(
        self,
        *,
        provider: BaseProvider,
        model: str,
        think: bool = False,
        tools: list[type[BaseTool]] | None = None,
        system_prompt: dict | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._think = think
        self._tool_classes: list[type[BaseTool]] = tools or []
        self._tool_instances: list[BaseTool] = []
        self._messages: list[dict[str, Any]] = [] if system_prompt is None else [system_prompt]
        self._input_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._event_queue: asyncio.Queue[Event | object] = asyncio.Queue()
        self._process_task: asyncio.Task | None = None

    async def __aenter__(self) -> "Backend":
        init_tasks = [asyncio.create_task(self._init_tool(cls)) for cls in self._tool_classes]
        self._tool_instances = list(await asyncio.gather(*init_tasks))
        self._process_task = asyncio.create_task(self._process_loop())
        return self

    async def __aexit__(self, *_) -> None:
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        await self._event_queue.put(_SENTINEL)

    @staticmethod
    async def _init_tool(tool_cls: type[BaseTool]) -> BaseTool:
        return tool_cls()

    def feed(self, input_id: str, text: str) -> None:
        self._input_queue.put_nowait((input_id, text))

    async def stream_events(self) -> AsyncGenerator[Event, None]:
        while True:
            item = await self._event_queue.get()
            if item is _SENTINEL:
                break
            yield item  # type: ignore[misc]

    async def _process_loop(self) -> None:
        while True:
            input_id, text = await self._input_queue.get()
            try:
                await self._run_turn(input_id, text)
            except Exception as exc:
                await self._event_queue.put(DoneEvent(id=input_id, text=None, error=str(exc)))

    async def _run_turn(self, input_id: str, text: str) -> None:
        self._messages.append({"role": "user", "content": text})
        await self._event_queue.put(UserInputEvent(id=input_id, text=text, error=None))

        while True:
            response_text = ""
            tool_calls: list[dict[str, Any]] = []

            async for part in self._provider.chat(
                model=self._model,
                messages=self._messages,
                think=self._think,
            ):
                if fragment := part.message.thinking:
                    await self._event_queue.put(ThinkingEvent(id=input_id, text=fragment, error=None))

                if fragment := part.message.content:
                    response_text += fragment
                    await self._event_queue.put(ResponseEvent(id=input_id, text=fragment, error=None))

                for tc in part.message.tool_calls:
                    tool_calls.append(
                        {
                            "id": str(uuid.uuid4()),
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    )

            msg: dict[str, Any] = {"role": "assistant", "content": response_text}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            self._messages.append(msg)

            if not tool_calls:
                break

            async def run_tool(tc: dict) -> tuple[str, str, str | None, str | None]:
                tool_id = tc["id"]
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                output, error = await self._execute_tool(name, args)
                return tool_id, name, output, error

            tasks = [asyncio.create_task(run_tool(tc)) for tc in tool_calls]

            for tc in tool_calls:
                await self._event_queue.put(
                    ToolStartEvent(
                        id=input_id,
                        text=None,
                        error=None,
                        tool_id=tc["id"],
                        tool_name=tc["function"]["name"],
                        tool_input=tc["function"]["arguments"],
                    )
                )

            results: dict[str, str] = {}
            for coro in asyncio.as_completed(tasks):
                tool_id, name, output, error = await coro
                if error:
                    results[tool_id] = ""
                else:
                    results[tool_id] = output or ""
                await self._event_queue.put(
                    ToolFinishEvent(
                        id=input_id,
                        text=None,
                        error=error,
                        tool_id=tool_id,
                        tool_name=name,
                        tool_output={"result": output} if output else {},
                    )
                )

            for tc in tool_calls:
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": results[tc["id"]],
                    }
                )

        await self._event_queue.put(DoneEvent(id=input_id, text=None, error=None))

    async def _execute_tool(self, name: str, args: dict) -> tuple[str | None, str | None]:
        """Returns (output, error). One will be None, the other will have a value."""
        for instance in self._tool_instances:
            if instance.name == name:
                try:
                    result = await instance.execute(**args)
                    return result, None
                except Exception as exc:
                    return None, str(exc)
        return None, f"Unknown tool: {name}"
