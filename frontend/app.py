import argparse
import asyncio
import uuid
from collections.abc import AsyncGenerator

from prompt_toolkit import PromptSession
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from backend import (Backend, DoneEvent, Event, ResponseEvent, ThinkingEvent,
                     ToolFinishEvent, ToolStartEvent, UserInputEvent)
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


def _build_live(thinking_buf: list[str], response_buf: list[str]):
    """Combine thinking and response buffers into a single renderable."""
    parts = []
    if thinking_buf:
        parts.append(Text("".join(thinking_buf), style="dim italic"))
    if response_buf:
        parts.append(Markdown("".join(response_buf)))
    return Group(*parts) if parts else Group()


class ChatFrontend:
    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self._console = Console(highlight=False, markup=True)
        self._prompt = PromptSession()
        self._live: Live | None = None

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
                except KeyboardInterrupt, EOFError:
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
        response_buf = []
        thinking_buf = []
        with Live(console=self._console, auto_refresh=False) as live:
            async for event in events_gen:
                if event.id != input_id:
                    # TODO: This doesn't handle a non-linear flow: other events are just dropped
                    # What should happen: events are always caught, then they are put in a queue.
                    # Then this function would operate on that queue
                    continue
                self._render_event(live, response_buf, thinking_buf, event)
                if isinstance(event, DoneEvent) and event.id == input_id:
                    # This should NOT be in _render_event as _collecting turn stops here.
                    # Only rendered in case of error
                    if event.error:
                        self._console.print(f"\n[red]Error: {event.error}[/red]")
                    break

    def _render_event(self, live: Live, response_buf: list, thinking_buf: list, event: Event) -> None:
        match event:
            case UserInputEvent():
                pass
            case ThinkingEvent(text=text):
                thinking_buf.append(text)
                live.update(_build_live(thinking_buf, response_buf))
                live.refresh()
            case ResponseEvent(text=text):
                response_buf.append(text)
                live.update(_build_live(thinking_buf, response_buf))
                live.refresh()
            case ToolStartEvent(tool_name=name, tool_input=inp):
                fmt = _fmt_tool_input(inp)
                self._console.print(f"\n⏺ {name}({fmt})", style="bold blue")
            case ToolFinishEvent(tool_name=name, tool_output=out, error=err):
                if err:
                    self._console.print(f"  Error: {err}", style="red", markup=False)
                else:
                    result = out.get("result", "") if out else ""
                    if output_format := out.get("output_format"):
                        match output_format:
                            case "all":
                                self._console.print(result, style="dim cyan", markup=False)
                            case "truncate":
                                if len(result) > MAX_TOOL_OUTPUT:
                                    result = result[:MAX_TOOL_OUTPUT] + "… [truncated]"
                                for line in result.splitlines():
                                    self._console.print(f"  {line}", style="dim cyan", markup=False)
                            case "none":
                                pass
                            case _:
                                raise RuntimeError("Encountered invalid output format (shouldn't happen)")

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
