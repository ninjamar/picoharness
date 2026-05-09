import asyncio
from pathlib import Path

from .base import BaseTool


class ReadFileTool(BaseTool):
    """Tool to read the contents of a file."""

    name = "read_file"

    output_format = "none"  # Don't print output since they are too large to show

    # IMPORTANT: Do not add any other parameters exept for what is needed as tool calls are constructed from the annotation
    async def execute(self, path) -> str:
        """Read the contents of a file on disk and return them as a string.

        Args:
            path: Absolute or relative path to the file to read.
        """
        try:
            return await asyncio.to_thread(Path(path).read_text)
        except OSError as e:
            return f"Error reading file: {e}"
