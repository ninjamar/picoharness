from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomlkit
from pydantic import Field, create_model

from backend.api import ALL_TOOLS
from frontend.schema import FIELDS, FieldDef


def _field_spec(f: FieldDef):
    """Return (type, default) tuple for pydantic.create_model."""
    if isinstance(f.default, list):
        return (f.type, Field(default_factory=lambda d=f.default: list(d)))
    return (f.type, f.default)


# Dynamically create AppConfig from schema
_appconfig_fields = {f.name: _field_spec(f) for f in FIELDS}
AppConfig = create_model("AppConfig", **_appconfig_fields)


def load_config(path: Path, preset: str | None = None) -> Any:
    """Load and validate TOML config file into AppConfig."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    if not data:
        raise SystemExit("Config file is empty")
    preset = preset or next(iter(data))
    if preset not in data:
        raise SystemExit(f"Preset '{preset}' not found. Available: {list(data.keys())}")
    return AppConfig.model_validate(data[preset])


def generate_config(path: Path) -> None:
    """Generate a sample TOML config file from schema."""

    DOCKER_TOOLS = {
        "search_web": "SearXNG",
        "search_and_read_web": "SearXNG",
        "read_webpage": "Jina Reader",
    }

    doc = tomlkit.document()
    table = tomlkit.table()
    for f in FIELDS:
        # Special handling for tools field: build multiline array with comments
        if f.name == "tools":
            table.add(tomlkit.comment(f.description))
            lines = ["["]
            for tool_cls in ALL_TOOLS:
                name = tool_cls.name
                if name in DOCKER_TOOLS:
                    lines.append(f'    # "{name}",  # Requires {DOCKER_TOOLS[name]} docker service')
                else:
                    lines.append(f'    "{name}",')
            lines.append("]")
            raw_array = "\n".join(lines)
            table.add(f.name, tomlkit.parse(f"x = {raw_array}\n")["x"])
            continue

        # Regular field handling
        table.add(tomlkit.comment(f.description))
        if (c := f.config_comment) is not None:
            table.add(tomlkit.comment(c))

        if f.commented_by_default:
            val = f.default if f.default is not None else ""
            if isinstance(val, str):
                val_str = f'"{val}"'
            else:
                val_str = str(val)
            table.add(tomlkit.comment(f"{f.name} = {val_str}"))
        elif f.default is None:
            table.add(tomlkit.comment(f"{f.name} = "))
        else:
            table.add(f.name, f.default)

    doc.add("base", table)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(doc))
    print(f"Config written to {path}")
