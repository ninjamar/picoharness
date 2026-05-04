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


def _build_live(segments: list[tuple[str, list[str]]]):
    """Build live display from interleaved thinking/response/tool segments."""
    parts = []
    for kind, chunks in segments:
        match kind:
            case "thinking":
                text = "".join(chunks)
                parts.append(Text(text, style="dim italic"))
            case "response":
                text = "".join(chunks)
                parts.append(Markdown(text))
            case "tool_start":
                parts.append(Text(f"\n{chunks[0]}", style="bold blue"))
            case "tool_output":
                text = "\n".join(chunks)
                parts.append(Text(text, style="dim cyan"))
            case "tool_error":
                parts.append(Text(chunks[0], style="red"))
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
        segments: list[tuple[str, list[str]]] = []
        with Live(console=self._console, auto_refresh=False) as live:
            async for event in events_gen:
                if event.id != input_id:
                    # TODO: This doesn't handle a non-linear flow: other events are just dropped
                    # What should happen: events are always caught, then they are put in a queue.
                    # Then this function would operate on that queue
                    continue
                self._render_event(live, segments, event)
                if isinstance(event, DoneEvent) and event.id == input_id:
                    # This should NOT be in _render_event as _collecting turn stops here.
                    # Only rendered in case of error
                    if event.error:
                        self._console.print(f"\n[red]Error: {event.error}[/red]")
                    break

    def _render_event(self, live: Live, segments: list[tuple[str, list[str]]], event: Event) -> None:
        match event:
            case UserInputEvent():
                pass
            case ThinkingEvent(text=text) if text:
                if segments and segments[-1][0] == "thinking":
                    segments[-1][1].append(text)
                else:
                    segments.append(("thinking", [text]))
                live.update(_build_live(segments))
                live.refresh()
            case ResponseEvent(text=text) if text:
                if segments and segments[-1][0] == "response":
                    segments[-1][1].append(text)
                else:
                    segments.append(("response", [text]))
                live.update(_build_live(segments))
                live.refresh()
            case ToolStartEvent(tool_name=name, tool_input=inp):
                fmt = _fmt_tool_input(inp)
                segments.append(("tool_start", [f"⏺ {name}({fmt})"]))
                live.update(_build_live(segments))
                live.refresh()
            case ToolFinishEvent(tool_name=name, tool_output=out, error=err):
                if err:
                    segments.append(("tool_error", [f"  Error: {err}"]))
                else:
                    result = out.get("result", "") if out else ""
                    if output_format := out.get("output_format"):
                        match output_format:
                            case "all":
                                segments.append(("tool_output", [result]))
                            case "truncate":
                                if len(result) > MAX_TOOL_OUTPUT:
                                    result = result[:MAX_TOOL_OUTPUT] + "… [truncated]"
                                lines = [f"  {line}" for line in result.splitlines()]
                                segments.append(("tool_output", lines))
                            case "none":
                                pass
                            case _:
                                raise RuntimeError("Encountered invalid output format (shouldn't happen)")
                live.update(_build_live(segments))
                live.refresh()

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
