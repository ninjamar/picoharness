import argparse
import asyncio
import uuid
from collections.abc import AsyncGenerator

from prompt_toolkit import PromptSession
from rich.console import Console

from backend import (
    Backend,
    DoneEvent,
    Event,
    ResponseEvent,
    ThinkingEvent,
    ToolFinishEvent,
    ToolStartEvent,
    UserInputEvent,
)
from backend.backend import ALLOWED_TOOLS
from backend.provider.provider import OllamaProvider, OpenAICompatibleProvider

MAX_TOOL_OUTPUT = 500


def _fmt_tool_input(inp: dict | str) -> str:
    if not isinstance(inp, dict):
        return repr(inp)
    items = list(inp.items())
    if len(items) == 1:
        return repr(items[0][1])
    return ", ".join(f"{k}={v!r}" for k, v in items)


class ChatFrontend:
    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self._console = Console(highlight=False, markup=True)
        self._prompt = PromptSession()

    def run(self) -> None:
        """Sync entry point — creates event loop and runs the chat."""
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        async with self._backend:
            self._print_header()
            events_gen = self._backend.stream_events()
            while True:
                try:
                    user_text = await self._read_input()
                except (KeyboardInterrupt, EOFError):
                    self._console.print("\n[dim]Bye.[/dim]")
                    break
                if not user_text.strip():
                    continue
                input_id = str(uuid.uuid4())
                self._backend.feed(input_id, user_text)
                await self._collect_turn(input_id, events_gen)
                self._console.print()

    async def _read_input(self) -> str:
        return await self._prompt.prompt_async("> ")

    async def _collect_turn(self, input_id: str, events_gen: AsyncGenerator) -> None:
        last_was_thinking = False
        async for event in events_gen:
            if event.id != input_id:
                continue
            self._render_event(event, last_was_thinking)
            last_was_thinking = isinstance(event, ThinkingEvent)
            if isinstance(event, DoneEvent) and event.id == input_id:
                if event.error:
                    self._console.print(f"\n[red]Error: {event.error}[/red]")
                break

    def _render_event(self, event: Event, last_was_thinking: bool) -> None:
        match event:
            case UserInputEvent():
                pass
            case ThinkingEvent(text=text):
                self._console.print(text, end="", style="dim italic", markup=False)
            case ResponseEvent(text=text):
                if last_was_thinking:
                    self._console.print()
                self._console.print(text, end="", markup=False)
            case ToolStartEvent(tool_name=name, tool_input=inp):
                fmt = _fmt_tool_input(inp)
                self._console.print(f"\n⏺ {name}({fmt})", style="bold blue")
            case ToolFinishEvent(tool_name=name, tool_output=out, error=err):
                if err:
                    self._console.print(f"  Error: {err}", style="red", markup=False)
                else:
                    result = out.get("result", "") if out else ""
                    if len(result) > MAX_TOOL_OUTPUT:
                        result = result[:MAX_TOOL_OUTPUT] + "… [truncated]"
                    for line in result.splitlines():
                        self._console.print(f"  {line}", style="dim cyan", markup=False)
            case DoneEvent():
                pass

    def _print_header(self) -> None:
        self._console.rule("[bold]LocalAI Chat[/bold]")
        self._console.print("[dim]Ctrl+C or Ctrl+D to quit[/dim]\n")


def cli() -> None:
    parser = argparse.ArgumentParser(description="LocalAI TUI")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", required=True, help="'ollama' or 'host:port' for OpenAI-compatible")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    tools = ALLOWED_TOOLS

    if args.provider == "ollama":
        provider = OllamaProvider(tools=tools)
    else:
        provider = OpenAICompatibleProvider(base_url=f"http://{args.provider}/v1", tools=tools)

    backend = Backend(provider=provider, model=args.model, think=args.think, tools=tools)
    ChatFrontend(backend=backend).run()
