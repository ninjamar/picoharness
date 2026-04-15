import ollama


class OllamaProvider:
    """Manages the Ollama AsyncClient connection."""

    def __init__(self):
        self.client = ollama.AsyncClient()

    async def chat(self, *, model, messages, stream, think, tools):
        """Call the Ollama chat API with the given parameters."""
        return await self.client.chat(
            model=model,
            messages=messages,
            stream=stream,
            think=think,
            tools=tools,
        )
