from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomlkit
from pydantic import Field, create_model

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
    doc = tomlkit.document()
    table = tomlkit.table()
    for f in FIELDS:
        table.add(tomlkit.comment(f.description))
        if f.default is None:
            table.add(tomlkit.comment(f"{f.name} = "))
        else:
            table.add(f.name, f.default)
    doc.add("base", table)
    path.write_text(tomlkit.dumps(doc))
    print(f"Config written to {path}")
