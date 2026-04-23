import argparse

from .config import Configuration
from .state import set_config
from .tools import BaseTool, ReadFileTool
from .tools.agent import AgentTool


def main(
    model: str,
    agent_model: str,
    provider: str,
    tools: list[type[BaseTool]] | None = None,
    think: bool = True,
) -> None:
    config = Configuration(
        model=model,
        agent_model=agent_model,
        think=think,
        tools=tools or [],
        provider=provider,
    )
    set_config(config)
    backend = config.spawn_backend()
    app = config.spawn_terminal_ui(backend)
    app.run()


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--agent-model", default=None)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--provider", required=True)
    args = parser.parse_args()

    args.agent_model = args.agent_model or args.model

    main(
        tools=[AgentTool, ReadFileTool],
        model=args.model,
        agent_model=args.agent_model,
        think=args.think,
        provider=args.provider,
    )
