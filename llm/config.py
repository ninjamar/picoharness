from pathlib import Path

from .core.backend import ChatBackend
from .core.provider import *
from .tools import BaseTool
from .tools.agent import Agent
from .ui import ChatApp

SYSTEM_PROMPT = {"role": "system", "content": (Path(__file__).parent / "system_prompt.md").read_text()}

AGENT_SYSTEM_PROMPT = {"role": "system", "content": (Path(__file__).parent / "agent_system_prompt.md").read_text()}


class Configuration:
    def __init__(
        self,
        model: str,
        agent_model: str,
        think: bool,
        system_prompt: dict[str, str] | None = SYSTEM_PROMPT,
        agent_system_prompt: dict[str, str] | None = AGENT_SYSTEM_PROMPT,
        tools: list[type[BaseTool]] | None = None,
        provider: str = "ollama",
    ) -> None:
        self.model = model
        self.agent_model = agent_model

        self.think = think

        self.system_prompt = system_prompt
        self.agent_system_prompt = agent_system_prompt

        self.tools = tools or []

        self.provider = provider
        self.provider_instance = self._spawn_provider_instance()

    def _spawn_provider_instance(self) -> BaseProvider:
        if self.provider == "ollama":
            return OllamaProvider(tools=self.tools)
        else:
            return OpenAICompatibleProvider(f"http://{self.provider}/v1", tools=self.tools)

    def spawn_backend(self) -> ChatBackend:
        return ChatBackend(
            provider=self.provider_instance,
            model=self.model,
            think=self.think,
            system_prompt=self.system_prompt,
            tools=self.tools,
        )

    def spawn_terminal_ui(self, backend: ChatBackend) -> ChatApp:
        return ChatApp(backend=backend)

    def spawn_agent(self, prompt: str) -> "Agent":
        backend = ChatBackend(
            provider=self.provider_instance,
            model=self.agent_model,
            think=self.think,
            system_prompt=self.agent_system_prompt,
            tools=[],
        )
        return Agent(backend=backend, prompt=prompt)
