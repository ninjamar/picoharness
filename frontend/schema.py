from __future__ import annotations

from dataclasses import dataclass, field
from inspect import iscoroutinefunction
from typing import TYPE_CHECKING, Any

from backend.provider import OllamaProvider, OpenAICompatibleProvider

if TYPE_CHECKING:
    from frontend.app import ChatApp


@dataclass
class BaseMenu:
    get_current: Any  # Callable[["ChatApp"], Any]
    set_current: Any  # Callable[["ChatApp", Any], None]
    label: str | None = None


@dataclass
class ToggleMenu(BaseMenu):
    pass


@dataclass
class TextInputMenu(BaseMenu):
    nullable: bool = False


@dataclass
class DialogueMenu(BaseMenu):
    choices: Any = field(default_factory=list)


@dataclass
class MultiSelectMenu(BaseMenu):
    choices: Any = field(default_factory=list)


@dataclass
class FieldDef:
    name: str
    type: Any
    default: Any
    description: str
    menu: BaseMenu
    show: bool = True
    config_comment: str | None = None
    commented_by_default: bool = False


async def resolve_choices(choices: Any, app: ChatApp) -> list:
    """Resolve choices: list passthrough or call callable(app) - handles async and sync."""
    if iscoroutinefunction(choices):
        return list(await choices(app))
    elif callable(choices):
        return list(choices(app))
    return list(choices)


async def _fetch_models(app: ChatApp) -> list[str]:
    model_infos = await app.api.get_available_models()
    return [m.name for m in model_infos]


def _get_all_tools(app: ChatApp) -> list[str]:
    return app.api.get_all_tools()


def _set_provider(app: ChatApp, value: str) -> None:
    if value == "ollama":
        provider = OllamaProvider()
    else:
        provider = OpenAICompatibleProvider(base_url=f"http://{value}/v1", api_key=app.api.api_key or "")
    app.api.set_provider(provider)


# Sentinel label used in the provider DialogueMenu to trigger the custom URL flow
CUSTOM_PROVIDER_LABEL = "Custom..."

FIELDS: list[FieldDef] = [
    FieldDef(
        name="model",
        type=str,
        default="qwen3.5:4b",
        description="Ollama/OpenAI-compatible model identifier",
        menu=DialogueMenu(
            get_current=lambda f: f.api.model,
            set_current=lambda f, v: f.api.set_model(v),
            choices=_fetch_models,
        ),
        show=True,
    ),
    FieldDef(
        name="provider",
        type=str,
        default="ollama",
        description="Provider: 'ollama' or an OpenAI-compatible base URL",
        menu=DialogueMenu(
            get_current=lambda f: f.api.get_provider_type(),
            set_current=_set_provider,
            choices=lambda f: ["ollama", CUSTOM_PROVIDER_LABEL],
        ),
        show=False,
    ),
    FieldDef(
        name="think",
        type=bool,
        default=False,
        description="Enable chain-of-thought / extended thinking",
        menu=ToggleMenu(
            get_current=lambda f: f.api.think,
            set_current=lambda f, v: f.api.set_think(True if v == "on" else False),
        ),
        show=True,
    ),
    FieldDef(
        name="show_think",
        type=bool,
        default=True,
        description="Display thinking output in the terminal",
        menu=ToggleMenu(
            get_current=lambda f: f.show_think,
            set_current=lambda f, v: f.set_show_think(True if v == "on" else False),
        ),
        show=True,
    ),
    FieldDef(
        name="system_prompt_path",
        type=str | None,
        default=None,
        description="Path to a Markdown system prompt file",
        menu=TextInputMenu(
            get_current=lambda f: f.api.system_prompt_path,
            set_current=lambda f, v: f.api.set_system_prompt_path(v),
            nullable=True,
        ),
        show=False,
    ),
    FieldDef(
        name="tools",
        type=list,
        default=[],
        description="Tools available to the model",
        menu=MultiSelectMenu(
            get_current=lambda f: f.api.get_enabled_tools(),
            set_current=lambda f, v: f.api.set_enabled_tools(v),
            choices=_get_all_tools,
        ),
        show=True,
    ),
    FieldDef(
        name="context_length",
        type=int,
        default=4096,
        description="Context window length of the model (higher number increases RAM usage)",
        menu=TextInputMenu(
            get_current=lambda f: f.api.context_length, set_current=lambda f, v: f.api.set_context_length(int(v))
        ),
        show=False,
    ),
    FieldDef(
        name="searxng_url",
        type=str,
        default="http://localhost:4000",
        description="Base URL of the SearXNG search service",
        config_comment="Uncomment this value if the docker service for the search is running. Or use a public SearXNG instance (that supports a json api)",
        commented_by_default=True,
        menu=TextInputMenu(
            get_current=lambda f: f.api.searxng_url,
            set_current=lambda f, v: f.api.set_searxng_url(v),
        ),
        show=False,
    ),
    FieldDef(
        name="jina_reader_url",
        type=str,
        default="http://localhost:3001",
        description="Base URL of the Jina Reader service",
        config_comment="Uncomment this value if the docker service for the reader is running. Or use a public Jina Reader endpoint (https://r.jina.ai)",
        commented_by_default=True,
        menu=TextInputMenu(
            get_current=lambda f: f.api.jina_reader_url,
            set_current=lambda f, v: f.api.set_jina_reader_url(v),
        ),
        show=False,
    ),
    FieldDef(
        name="summarizer_model",
        type=str | None,
        default=None,
        description="Model for summarization tools (uses main model if unset). Recommended: qwen2.5:1.5b",
        config_comment="Fast local model for webpage summarization. Pull with: ollama pull qwen2.5:1.5b",
        commented_by_default=True,
        menu=TextInputMenu(
            get_current=lambda f: f.api.summarizer_model or "",
            set_current=lambda f, v: f.api.set_summarizer_model(v if v else None),
            nullable=True,
        ),
        show=False,
    ),
    FieldDef(
        name="api_key",
        type=str | None,
        default=None,
        description="API key for OpenAI-compatible provider (if required)",
        config_comment="Uncomment and set if using an authenticated OpenAI-compatible endpoint",
        commented_by_default=True,
        menu=TextInputMenu(
            get_current=lambda f: f.api.api_key or "",
            set_current=lambda f, v: f.api.set_api_key(v if v else None),
            nullable=True,
        ),
        show=False,
    ),
    FieldDef(
        name="session_save_location",
        type=str,
        default="~/.ph/sessions/",
        description="Directory where chat sessions are saved",
        menu=TextInputMenu(
            get_current=lambda f: str(f.api.session_manager.save_dir),
            set_current=lambda f, v: f.api.set_session_save_location(v),
        ),
        show=False,
    ),
]

FIELDS_SHOW_UI = [f for f in FIELDS if f.show]
