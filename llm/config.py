from .client import STYLE, TerminalUI, Style
from .core.backend import ChatBackend
from .tools import BaseTool
from .tools.agent import Agent
from pathlib import Path
SYSTEM_PROMPT = {
    "role": "system",
    "content": (Path(__file__).parent / "system_prompt.md").read_text()
}

AGENT_SYSTEM_PROMPT = {
    "role": "system",
    "content": (Path(__file__).parent / "agent_system_prompt.md").read_text()
}


class Configuration:
    def __init__(
        self,
        model: str,
        agent_model: str,
        think: bool,
        system_prompt: dict[str, str] | None = SYSTEM_PROMPT,
        agent_system_prompt: dict[str, str] | None = AGENT_SYSTEM_PROMPT,
        tools: list[type[BaseTool]] | None = None,
        style: Style = STYLE,
    ) -> None:
        self.model = model
        self.agent_model = agent_model

        self.think = think

        self.system_prompt = system_prompt
        self.agent_system_prompt = agent_system_prompt

        self.tools = tools or []
        self.style = style

    def spawn_backend(self) -> ChatBackend:
        return ChatBackend(
            self,
            model=self.model,
            think=self.think,
            system_prompt=self.system_prompt,
            tools=self.tools,
        )

    def spawn_terminal_ui(self) -> TerminalUI:
        return TerminalUI(self, style=self.style)

    def spawn_agent(self, prompt: str) -> "Agent":
        backend = ChatBackend(
            self,
            model=self.agent_model,
            think=self.think,
            system_prompt=self.agent_system_prompt,
            # No tools yet because it loops subagent
            # tools=self.tools,
        )
        return Agent(backend=backend, prompt=prompt)
