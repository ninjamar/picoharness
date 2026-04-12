import asyncio
from typing import Literal
import ollama
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style
from prompt_toolkit.patch_stdout import patch_stdout

MODEL = "lfm2.5-thinking:latest"

STYLE = Style.from_dict(
    {
        "prompt": "ansibrightgreen bold",
        "thinking": "ansidarkgray italic",
        "response": "ansiwhite",
    }
)


class LocalAI:
    def __init__(self, model: str = MODEL):
        self.model = model
        self.session = PromptSession()
        self.client = ollama.AsyncClient()
        self.messages = []

    async def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        think = ""
        response = ""
        prev_mode: Literal["think", "text"] | None = None

        async for part in await self.client.chat(model=self.model, messages=self.messages, stream=True, think=True):
            if data := part.message.thinking:
                if prev_mode is not None and prev_mode != "think":
                    print()
                think += data
                print_formatted_text(FormattedText([("class:thinking", data)]), end="", flush=True, style=STYLE)
                prev_mode = "think"

            if data := part.message.content:
                if prev_mode is not None and prev_mode != "text":
                    print()
                response += data
                print_formatted_text(FormattedText([("class:response", data)]), end="", flush=True, style=STYLE)
                prev_mode = "text"

        print()  # trailing newline
        self.messages.append({"role": "assistant", "content": response})

    async def run(self):
        try:
            await self.loop()
        except KeyboardInterrupt as e:
            pass

    async def loop(self):
        print(f"Running model {self.model}. Ensure the context window has been turned up for optimal usage")

        while True:
            with patch_stdout():
                user_input = await self.session.prompt_async(
                    FormattedText([("class:prompt", "> ")]),
                    style=STYLE,
                )

            user_input = user_input.strip()
            if not user_input:
                continue

            await self.chat(user_input)


if __name__ == "__main__":
    asyncio.run(LocalAI().run())
