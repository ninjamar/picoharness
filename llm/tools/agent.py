from ..core.backend import ChatBackend
from ..events import ResponseEvent
from .base import BaseTool


class Agent:
    def __init__(self, backend: ChatBackend, prompt: str) -> None:
        self.backend = backend
        self.prompt = prompt

    async def get_output(self) -> str:
        response = ""
        async for event in self.backend.stream(self.prompt):
            if isinstance(event, ResponseEvent):
                response += event.text
        return response


class AgentTool(BaseTool):
    name = "agent"

    async def execute(self, prompt):
        """
        Spawn a sub-agent to complete a task or generate an output described in prompt

        Args:
            prompt: Query to ask sub-agent

        Returns:
            The output from the agent
        """
        return await self.config.spawn_agent(prompt).get_output()


if __name__ == "__main__":
    import pprint

    pprint.pprint(AgentTool.to_ollama())
