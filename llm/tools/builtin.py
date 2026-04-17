import asyncio
from pathlib import Path
from typing import Any

from .base import BaseTool


class ReadFileTool(BaseTool):
    """Tool to read the contents of a file."""

    name = "read_file"

    async def execute(self, path: str = "", **kwargs: Any) -> str:
        """Read the contents of a file on disk and return them as a string.

        Args:
            path: Absolute or relative path to the file to read.
        """
        try:
            return await asyncio.to_thread(Path(path).read_text)
        except OSError as e:
            return f"Error reading file: {e}"
