import asyncio

from llm.backend import ChatBackend
from llm.client import Client
from llm.tools import BaseTool, ReadFileTool, WeatherApiTool

MODEL = "lfm2.5-thinking:latest"

async def main(tools: list[type[BaseTool]] | None = None):
    """Run the interactive chat application."""
    backend = ChatBackend(MODEL, tools=tools or [])
    ui = Client()

    print(f"Running model {MODEL}. Ensure the context window has been turned up for optimal usage")

    while True:
        try:
            user_input = await ui.get_input()
        except KeyboardInterrupt:
            break

        try:
            await ui.render_stream(backend.stream(user_input))
        except asyncio.CancelledError:
            print()
            continue


if __name__ == "__main__":
    asyncio.run(main([WeatherApiTool, ReadFileTool]))
