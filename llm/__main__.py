import argparse
import asyncio

from .config import Configuration
from .state import set_config
from .tools import BaseTool, ReadFileTool
from .tools.agent import AgentTool


async def main(
    model: str,
    agent_model: str,
    provider: str,
    tools: list[type[BaseTool]] | None = None,
    think: bool = True,
):
    """Run the interactive chat application."""
    config = Configuration(
        model=model,
        agent_model=agent_model,
        think=think,
        tools=tools or [],
        provider=provider,
    )
    set_config(config)
    backend = config.spawn_backend()
    ui = config.spawn_terminal_ui()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--agent-model", default=None)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--provider", required=True)
    args = parser.parse_args()

    args.agent_model = args.agent_model or args.model

    app = main(
        tools=[AgentTool, ReadFileTool],
        model=args.model,
        agent_model=args.agent_model,
        think=args.think,
        provider=args.provider,
    )
    asyncio.run(app)
