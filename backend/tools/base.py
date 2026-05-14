from __future__ import annotations

import inspect
import re
from collections import defaultdict
from typing import Any, Literal


def _parse_docstring(doc_string: str | None) -> dict[str, str]:
    # Taken from https://github.com/ollama/ollama-python/blob/main/ollama/_utils.py
    # LICENSE: MIT
    parsed_docstring = defaultdict(str)
    if not doc_string:
        return parsed_docstring

    key = str(hash(doc_string))
    for line in doc_string.splitlines():
        lowered_line = line.lower().strip()
        if lowered_line.startswith("args:"):
            key = "args"
        elif lowered_line.startswith(("returns:", "yields:", "raises:")):
            key = "_"
        else:
            parsed_docstring[key] += f"{line.strip()}\n"

    last_key = None
    for line in parsed_docstring["args"].splitlines():
        line = line.strip()
        if ":" in line:
            parts = re.split(r"(?:\(([^)]*)\)|:)\s*", line, maxsplit=1)
            arg_name = parts[0].strip()
            last_key = arg_name
            arg_description = parts[-1].strip()
            if len(parts) > 2 and parts[1]:
                arg_description = parts[-1].split(":", 1)[-1].strip()
            parsed_docstring[last_key] = arg_description
        elif last_key and line:
            parsed_docstring[last_key] += " " + line

    return parsed_docstring


class BaseTool:
    name: str = ""
    output_format: Literal["all", "truncate", "none"]

    def __init__(self) -> None:
        pass

    @classmethod
    def to_schema(cls) -> dict[str, Any]:

        # Adapted from https://github.com/ollama/ollama-python/blob/main/ollama/_utils.py

        doc = inspect.getdoc(cls.execute)  # find for method cls.execute
        parsed = _parse_docstring(doc)
        doc_key = str(hash(doc))

        sig = inspect.signature(cls.execute)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for name, param in sig.parameters.items():
            type_map = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object",
            }
            json_type = type_map.get(param.annotation, "string")
            if param.annotation is inspect.Parameter.empty:
                json_type = "string"

            if name != "self":
                properties[name] = {
                    "type": json_type,
                    "description": parsed.get(name, ""),
                }
                if param.default is inspect.Parameter.empty:
                    required.append(name)

        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": parsed.get(doc_key, "").strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    async def execute(self, *args, **kwargs) -> str:
        """
        IMPORTANT: Do not add any other parameters exept for what is needed as tool calls are constructed from the annotation
        For example, having kwargs in the annotation will pass it to the ai
        """
        raise NotImplementedError
