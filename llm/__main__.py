import argparse
import asyncio

from .config import Configuration
from .tools import BaseTool, ReadFileTool
from .tools.agent import AgentTool


#DEFAULT_MODEL = "lfm2.5-thinking:latest"
#DEFAULT_AGENT_MODEL = "lfm2.5-thinking:latest"
DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_AGENT_MODEL="qwen3.5:0.8b"

async def main(tools: list[type[BaseTool]] | None = None, model: str = DEFAULT_MODEL, agent_model: str = DEFAULT_AGENT_MODEL, think: bool = True):
    """Run the interactive chat application."""
    config = Configuration(
        model=model,
        agent_model=agent_model,
        think=think,
        tools=tools or [],
    )
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
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    asyncio.run(main([AgentTool, ReadFileTool], model=args.model, agent_model=args.agent_model, think=args.think))
