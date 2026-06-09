from backend.provider import BaseProvider


class Agent:
    """Internal agent for running simple text tasks (e.g., summarization) with a configured provider and system prompt."""

    def __init__(self, provider: BaseProvider, model: str, system_prompt: str) -> None:
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt

    @staticmethod
    def for_single_task(provider: BaseProvider, model: str) -> "Agent":
        """Factory method to create an agent configured for webpage summarization."""
        return Agent(
            provider=provider,
            model=model,
            system_prompt="Output exactly and only what the input requests. Nothing more.",
        )

    async def run(self, input: str) -> str:
        """Send a single-turn request to the provider and return the accumulated response."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input},
        ]

        response_text = ""
        async for part in self.provider.chat(model=self.model, messages=messages, think=False):
            if part.message.content:
                response_text += part.message.content

        return response_text
