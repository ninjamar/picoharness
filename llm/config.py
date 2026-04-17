from .client import STYLE, Client, Style
from .core.backend import ChatBackend
from .tools import BaseTool
from .tools.agent import Agent


SYSTEM_PROMPT = {
    "role": "system",
    "content": "You have access to tools. Use available tools to complete the task. When tools are available and relevant, always use them. Be concise in responses.",
}

AGENT_SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a sub-agent. Complete the given task once. Use tools if needed. Your final response is returned directly to the caller — you will not run again.",
}

class Configuration:
    def __init__(
        self,
        model: str,
        agent_model: str,
        system_prompt: dict[str, str] | None = SYSTEM_PROMPT,
        agent_system_prompt: dict[str, str] | None = AGENT_SYSTEM_PROMPT,
        tools: list[type[BaseTool]] | None = None,
        style: Style = STYLE,
    ) -> None:
        self.model = model
        self.agent_model = agent_model
        self.system_prompt = system_prompt
        self.agent_system_prompt = agent_system_prompt

        self.tools = tools or []
        self.style = style

    def spawn_backend(self) -> ChatBackend:
        return ChatBackend(
            self,
            model=self.model,
            system_prompt=self.system_prompt,
            tools=self.tools,
        )

    def spawn_client(self) -> Client:
        return Client(self, style=self.style)

    def spawn_agent(self, prompt: str) -> "Agent":
        backend = ChatBackend(
            self,
            model=self.agent_model,
            system_prompt=self.agent_system_prompt,
            # No tools yet because it loops subagent
            #tools=self.tools,
        )
        return Agent(backend=backend, prompt=prompt)
