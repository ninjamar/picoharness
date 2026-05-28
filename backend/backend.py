from __future__ import annotations

import asyncio
import inspect
import uuid
from collections import namedtuple
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.api import BackendAPI

from backend.events import (
    DoneEvent,
    Event,
    ResponseEvent,
    ThinkingEvent,
    ToolErrorEvent,
    ToolOutputEvent,
    ToolStartEvent,
    UserInputEvent,
)
from backend.provider import BaseProvider
from backend.system_prompt import format_system_prompt
from backend.tools import BaseTool

_SENTINEL = object()


class _InputSentinel:
    """Signals _process_loop to exit cleanly on backend shutdown."""

    # In Python 3.15, use the new sentinel function: https://docs.python.org/3.15/library/functions.html#sentinel
    pass


ToolExecutionResult = namedtuple("ToolExecutionResult", ["result", "error", "output_format"])


class Backend:
    def __init__(
        self,
        *,
        provider: BaseProvider,
        model: str,
        think: bool = False,
        tools: list[type[BaseTool]] | None = None,
        system_prompt: str | None = None,
        system_prompt_path: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._think = think
        self._system_prompt_path = system_prompt_path

        self._enabled_tools_override: list[str] = []
        self._tool_classes: list[type[BaseTool]] = tools or []
        self._tool_schemas = [tool.to_schema() for tool in self._tool_classes]
        self._provider.tool_schemas = self._tool_schemas

        self._tool_instances: list[BaseTool] = []

        self._messages: list[dict[str, Any]] = (
            []
            if system_prompt is None
            else [{"role": "system", "content": format_system_prompt(system_prompt, self._tool_classes)}]
        )
        self._input_queue: asyncio.Queue[tuple[str, str] | _InputSentinel] = asyncio.Queue()
        self._event_queue: asyncio.Queue[Event | _InputSentinel] = asyncio.Queue()
        self._process_task: asyncio.Task | None = None
        self._current_turn_task: asyncio.Task | None = None

    @classmethod
    def from_config(cls, api: BackendAPI) -> Backend:
        return cls(
            provider=api.provider,
            model=api.model,
            think=api.think,
            tools=api.tool_classes,
            system_prompt=api.system_prompt,
            system_prompt_path=api.system_prompt_path,
        )

    async def __aenter__(self) -> Backend:
        init_tasks = [asyncio.create_task(self._init_tool(cls)) for cls in self._tool_classes]
        self._tool_instances = list(await asyncio.gather(*init_tasks))
        self._process_task = asyncio.create_task(self._process_loop())
        return self

    async def __aexit__(self, *_) -> None:
        if self._process_task:
            await self.shutdown()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        await self._event_queue.put(_SENTINEL)  # type: ignore

    @staticmethod
    async def _init_tool(tool_cls: type[BaseTool]) -> BaseTool:
        return tool_cls()

    def feed(self, input_id: str, text: str) -> None:
        self._input_queue.put_nowait((input_id, text))

    def cancel_current(self) -> None:
        if self._current_turn_task and not self._current_turn_task.done():
            self._current_turn_task.cancel()

    async def shutdown(self) -> None:
        if self._current_turn_task and not self._current_turn_task.done():
            self._current_turn_task.cancel()
        await self._input_queue.put(_InputSentinel())

    async def stream_events(self) -> AsyncGenerator[Event]:
        while True:
            item = await self._event_queue.get()
            if item is _SENTINEL:
                break
            yield item  # type: ignore[misc]

    async def _process_loop(self) -> None:
        while True:
            item = await self._input_queue.get()
            if isinstance(item, _InputSentinel):
                break
            input_id, text = item
            self._current_turn_task = asyncio.create_task(self._run_turn(input_id, text))
            try:
                await self._current_turn_task
            finally:
                self._current_turn_task = None

    async def _run_turn(self, input_id: str, text: str) -> None:
        # start_len = len(self._messages)

        self._messages.append({"role": "user", "content": text})

        try:
            await self._event_queue.put(UserInputEvent(id=input_id, text=text))

            while True:
                response_text = ""
                tool_calls: list[dict[str, Any]] = []

                async for part in self._provider.chat(
                    model=self._model,
                    messages=self._messages,
                    think=self._think,
                ):
                    if fragment := part.message.thinking:
                        await self._event_queue.put(ThinkingEvent(id=input_id, fragment=fragment))

                    if fragment := part.message.content:
                        response_text += fragment
                        await self._event_queue.put(ResponseEvent(id=input_id, fragment=fragment))

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

                async def run_tool(tc: dict) -> tuple[str, str, str | None, str | None, str | None]:
                    tool_id = tc["id"]
                    name = tc["function"]["name"]
                    args = tc["function"]["arguments"]
                    output, error, output_format = await self._execute_tool(name, args)
                    return tool_id, name, output, error, output_format

                tasks = [asyncio.create_task(run_tool(tc)) for tc in tool_calls]

                for tc in tool_calls:
                    await self._event_queue.put(
                        ToolStartEvent(
                            id=input_id,
                            tool_id=tc["id"],
                            tool_name=tc["function"]["name"],
                            tool_input=tc["function"]["arguments"],
                        )
                    )

                results: dict[str, str] = {}
                for coro in asyncio.as_completed(tasks):
                    tool_id, name, output, error, output_format = await coro
                    if error:
                        results[tool_id] = ""
                        await self._event_queue.put(
                            ToolErrorEvent(id=input_id, tool_id=tool_id, tool_name=name, error=error)
                        )
                    else:
                        results[tool_id] = output or ""
                        await self._event_queue.put(
                            ToolOutputEvent(
                                id=input_id,
                                tool_id=tool_id,
                                tool_name=name,
                                result=output or "",
                                output_format=output_format,  # type: ignore
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

            await self._event_queue.put(DoneEvent(id=input_id, error=None, interrupted=False))

        except asyncio.CancelledError:
            # User cancelled processing
            # Don't roll back messages
            # del self._messages[start_len:]

            # For consistency, the backend still sends DoneEvent with interrupt = true even
            # though the frontend initiates the interrupt
            await self._event_queue.put(DoneEvent(id=input_id, error=None, interrupted=True))
        except Exception as e:
            # Tell model there is an error so it hopefully doesn't keep trying to do the same
            # action on every next turn.
            self._messages.append({"role": "system", "content": f"Error: {str(e)}"})
            await self._event_queue.put(DoneEvent(id=input_id, error=str(e), interrupted=False))

    async def _execute_tool(self, name: str, args: dict) -> ToolExecutionResult:
        """Returns (output, error). One will be None, the other will have a value."""
        for instance in self._tool_instances:
            if instance.name == name:
                try:
                    result = await instance.execute(**args)
                    return ToolExecutionResult(result=result, error=None, output_format=instance.output_format)
                except Exception as exc:
                    return ToolExecutionResult(result=None, error=str(exc), output_format=None)
        return ToolExecutionResult(result=None, error=f"Unknown tool: {name}", output_format=None)
