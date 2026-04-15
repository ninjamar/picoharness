from collections.abc import AsyncGenerator

from llm.events import ResponseEvent, ThinkingEvent, ToolEvent
from llm.provider import OllamaProvider



class ChatBackend:
    """Manages ollama interactions and conversation history."""

    def __init__(self, model, system_prompt=None, tools: list = None) -> None:
        self.model = model
        self.client = OllamaProvider()
        self.messages: list[dict[str, str]] = [] if system_prompt is None else [system_prompt]
        self.tools = [] if tools is None else tools

    async def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return its result."""
        for tool_cls in self.tools:
            if tool_cls.name == tool_name:
                return await tool_cls.execute(**arguments)
        return f"Unknown tool: {tool_name}"

    async def stream(self, user_input: str | None = None) -> AsyncGenerator[ThinkingEvent | ResponseEvent, None]:
        """Stream response events from the model.

        Yields ThinkingEvent and ResponseEvent objects. Handles tool execution
        internally—when the model calls a tool, it executes it and continues
        generating the response based on the tool output.
        """
        if user_input is not None:
            self.messages.append({"role": "user", "content": user_input})

        while True:
            response = ""
            tool_calls: list[dict] = []

            async for part in await self.client.chat(
                model=self.model,
                messages=self.messages,
                stream=True,
                think=True,
                tools=[tool.to_ollama() for tool in self.tools] or None,
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

            # Execute tools and append results
            for tc in tool_calls:
                result = await self._execute_tool(tc["function"]["name"], tc["function"]["arguments"])

                yield ToolEvent(
                    name=tc["function"]["name"],
                    input=tc["function"]["arguments"],
                    output=result,
                )
                self.messages.append({"role": "tool", "tool_name": tc["function"]["name"], "content": result})

            # Loop continues: client.chat() is called again with updated messages
