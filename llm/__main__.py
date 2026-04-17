import asyncio

from .config import Configuration
from .tools import BaseTool
from .tools.agent import AgentTool

DEFAULT_MODEL = "lfm2.5-thinking:latest"
DEFAULT_AGENT_MODEL = "lfm2.5-thinking:latest"


async def main(tools: list[type[BaseTool]] | None = None):
    """Run the interactive chat application."""
    config = Configuration(
        model=DEFAULT_MODEL,
        agent_model=DEFAULT_AGENT_MODEL,
        tools=tools or [],
    )
    backend = config.spawn_backend()
    ui = config.spawn_client()

    print(f"Running model {config.model}. Ensure the context window has been turned up for optimal usage")

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
    asyncio.run(main([AgentTool]))
