from ollama._utils import convert_function_to_tool


class BaseTool:
    """Base class for tools that can be called by the model."""

    name: str = ""

    def __init__(self, config):
        self.config = config

    @classmethod
    def to_ollama(cls) -> dict:
        """Return the Ollama-compatible tool definition."""
        tool = convert_function_to_tool(cls.execute)

        # Strip the self parameter
        del tool.function.parameters.properties["self"]
        tool.function.parameters.required.remove("self")

        if tool.function is not None:
            tool.function.name = cls.name  # preserve the explicit class-level name
        return tool.model_dump()

    async def execute(self, **kwargs) -> str:
        """Execute the tool and return a result string."""
        raise NotImplementedError
