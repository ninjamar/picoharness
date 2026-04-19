import asyncio
import uuid
from collections.abc import AsyncGenerator

from ..events import ResponseEvent, ThinkingEvent, ToolEndEvent, ToolStartEvent
from ..tools import BaseTool
from .provider import OllamaProvider, OpenAICompatibleProvider


class ChatBackend:
    """Manages ollama interactions and conversation history."""

    def __init__(
        self,
        config,
        model: str,
        think: bool,
        system_prompt: dict[str, str] | None = None,
        tools: list[type[BaseTool]] | None = None,
    ) -> None:
        self.config = config

        self.think = think

        self.model = model
        # self.client = OllamaProvider()
        self.client = OpenAICompatibleProvider("http://127.0.0.1:8000/v1")
        self.messages: list[dict[str, str]] = [] if system_prompt is None else [system_prompt]

        self.tools_instances = [tool(self.config) for tool in tools] if tools else None
        self.tools_schemas = [tool.to_schema() for tool in tools] if tools else None

    async def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return its result."""
        for tool_instance in self.tools_instances:
            if tool_instance.name == tool_name:
                return await tool_instance.execute(**arguments)
        return f"Unknown tool: {tool_name}"

    async def stream(
        self, user_input: str | None = None
    ) -> AsyncGenerator[ThinkingEvent | ResponseEvent | ToolStartEvent | ToolEndEvent, None]:
        """Stream response events from the model.

        Yields ThinkingEvent and ResponseEvent objects. Handles tool execution
        internally—when the model calls tools, it yields ToolStartEvent for each,
        executes them concurrently, yields ToolEndEvent as each completes,
        and continues generating the response based on the tool outputs.
        """
        if user_input is not None:
            self.messages.append({"role": "user", "content": user_input})

        while True:
            response = ""
            tool_calls: list[dict] = []

            async for part in self.client.chat(
                model=self.model,
                messages=self.messages,
                stream=True,
                think=self.think,
                tools=self.tools_schemas,
            ):
                if data := part.message.thinking:
                    yield ThinkingEvent(data)

                if data := part.message.content:
                    response += data
                    yield ResponseEvent(data)

                if part.message.tool_calls:
                    tool_calls.extend(
                        {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in part.message.tool_calls
                    )

            msg = {"role": "assistant", "content": response}
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
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                tool_id = str(uuid.uuid4())
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
                self.messages.append({"role": "tool", "tool_name": name, "content": results[tool_id]})

            # Loop continues: client.chat() is called again with updated messages
