import datetime
import inspect

from jinja2 import Template

from backend.tools import BaseTool


def _get_tool_info(tool_classes: list[type[BaseTool]]) -> str:
    info = []
    for tool in tool_classes:
        sig = inspect.signature(tool.execute)

        params = dict(sig.parameters)
        params.pop("self", None)
        clean_sig = sig.replace(parameters=list(params.values()))

        docstring = inspect.getdoc(tool.execute) or ""
        name = tool.name

        indented_doc = "\n".join(f"    {line}" for line in docstring.splitlines())
        info.append(f"- {name}{clean_sig}\n{indented_doc}")
        # info.append(f"{name}{clean_sig}\n{docstring}")

    return "\n".join(info)


def format_system_prompt(prompt: str, tool_classes: list[type[BaseTool]]) -> str:
    # TODO: Use actual templating engine. pass info like date...
    template = Template(prompt)
    return template.render(tools=_get_tool_info(tool_classes), date=datetime.datetime.now().strftime("%B %d, %Y"))
    # return prompt.replace("{{tools}}", _get_tool_info(tool_classes))
