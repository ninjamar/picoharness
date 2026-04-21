from abc import ABC, abstractmethod
from typing import Any

from ollama._utils import convert_function_to_tool


class BaseTool(ABC):
    name: str = ""

    def __init__(self) -> None:
        pass

    @classmethod
    def to_schema(cls) -> dict[str, Any]:
        tool = convert_function_to_tool(cls.execute)

        # TODO: Remove asserts
        assert tool is not None
        assert tool.function is not None
        assert tool.function.parameters is not None
        assert tool.function.parameters.properties is not None
        assert tool.function.parameters.required is not None

        props = dict(tool.function.parameters.properties)
        del props["self"]
        tool.function.parameters.properties = props
        required_list = list(tool.function.parameters.required)
        required_list.remove("self")
        tool.function.parameters.required = required_list
        tool.function.name = cls.name
        return tool.model_dump()

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """
        IMPORTANT: Do not add any other parameters exept for what is needed as tool calls are constructed from the annotation
        For example, having kwargs in the annotation will pass it to the ai
        """
        raise NotImplementedError
