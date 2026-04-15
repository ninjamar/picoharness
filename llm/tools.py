import asyncio
from pathlib import Path

from ollama._utils import convert_function_to_tool


class BaseTool:
    """Base class for tools that can be called by the model."""

    name: str = ""

    @classmethod
    def to_ollama(cls) -> dict:
        """Return the Ollama-compatible tool definition."""
        tool = convert_function_to_tool(cls.execute)
        tool.function.name = cls.name  # preserve the explicit class-level name
        return tool.model_dump()

    @classmethod
    async def execute(cls, **kwargs) -> str:
        """Execute the tool and return a result string."""
        raise NotImplementedError


class ReadFileTool(BaseTool):
    """Tool to read the contents of a file."""

    name = "read_file"

    @classmethod
    async def execute(cls, path: str, **kwargs) -> str:
        """Read the contents of a file on disk and return them as a string.

        Args:
            path: Absolute or relative path to the file to read.
        """
        try:
            return await asyncio.to_thread(Path(path).read_text)
        except OSError as e:
            return f"Error reading file: {e}"


class WeatherApiTool(BaseTool):
    name = "stub"

    @classmethod
    async def execute(cls) -> str:
        """Get the weather"""
        return "The weather is sunny today"
