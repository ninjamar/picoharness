from ollama._utils import convert_function_to_tool


class BaseTool:
    """Base class for tools that can be called by the model."""

    name: str = ""

    def __init__(self, config):
        self.config = config

    @classmethod
    def to_schema(cls) -> dict:
        """Convert a BaseTool class to Ollama/OpenAI-compatible JSON schema."""
        tool = convert_function_to_tool(cls.execute)
        del tool.function.parameters.properties["self"]
        tool.function.parameters.required.remove("self")
        tool.function.name = cls.name
        return tool.model_dump()

    async def execute(self, **kwargs) -> str:
        """Execute the tool and return a result string."""
        raise NotImplementedError
