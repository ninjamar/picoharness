import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from ..events import (Event, ResponseEvent, ThinkingEvent, ToolEndEvent,
                      ToolStartEvent)
from ..tools import BaseTool
from .provider import OllamaProvider, OpenAICompatibleProvider


class ChatBackend:
    def __init__(
        self,
        model: str,
        think: bool,
        system_prompt: dict[str, str] | None = None,
        tools: list[type[BaseTool]] | None = None,
    ) -> None:
        self.think = think

        self.model = model
        # self.client = OllamaProvider(tools=tools or [])
        self.client = OpenAICompatibleProvider("http://127.0.0.1:11434/v1", tools=tools or [])
        self.messages: list[dict[str, Any]] = [] if system_prompt is None else [system_prompt]

        self.tools_instances = [tool() for tool in tools] if tools else []

    async def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        for tool_instance in self.tools_instances:
            if tool_instance.name == tool_name:
                return await tool_instance.execute(**arguments)
        return f"Unknown tool: {tool_name}"

    async def stream(
        self, user_input: str | None = None
    ) -> AsyncGenerator[Event, None]:  # AsyncGenerator[SendType, RecvType]
        if user_input is not None:
            self.messages.append({"role": "user", "content": user_input})

        while True:
            response = ""
            tool_calls: list[dict[str, Any]] = []

            async for part in self.client.chat(
                model=self.model,
                messages=self.messages,
                think=self.think,
            ):
                if data := part.message.thinking:
                    yield ThinkingEvent(data)

                if data := part.message.content:
                    response += data
                    yield ResponseEvent(data)

                if part.message.tool_calls:
                    for tc in part.message.tool_calls:
                        tool_id = str(uuid.uuid4())
                        tool_calls.append(
                            {
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    # "arguments": json.dumps(tc.function.arguments)
                                    "arguments": tc.function.arguments,
                                },
                            }
                        )

            msg: dict[str, Any] = {"role": "assistant", "content": response}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            self.messages.append(msg)

            if not tool_calls:  # exit when no tools left
                break

            # Execute all tools concurrently
            async def run_tool(tool_id: str, name: str, args: dict) -> tuple[str, str, str]:
                result = await self._execute_tool(name, args)
                return tool_id, name, result

            tasks = []
            for tc in tool_calls:
                tool_id = tc["id"]
                name = tc["function"]["name"]
                # args = json.loads(tc["function"]["arguments"])
                args = tc["function"]["arguments"]
                task = asyncio.create_task(run_tool(tool_id, name, args))

                tasks.append((tool_id, name, args, task))

                yield ToolStartEvent(id=tool_id, name=name, input=args)

            # Collect results as tools complete, yield ToolEndEvent for each
            results = {}
            for coro in asyncio.as_completed([t for _, _, _, t in tasks]):
                tool_id, name, result = await coro
                results[tool_id] = result
                yield ToolEndEvent(id=tool_id, output=result)

            # Append tool results to messages in original order
            for tool_id, name, _, _ in tasks:
                self.messages.append({"role": "tool", "tool_call_id": tool_id, "content": results[tool_id]})

            # Loop continues: client.chat() is called again with updated messages
