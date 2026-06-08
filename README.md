# PicoHarness

A TUI chat interface that runs small local models as tool-calling agents. Models can search the web, read files, and browse Wikipedia — all running locally for full privacy with minimal overhead.

> Most agent harnesses target frontier models (Claude, GPT-4) and require 8b+ parameter models to function reliably. PicoHarness is built for small models — 1b to 4b parameters — so you can run a capable, no-cost AI assistant on consumer hardware. As a high school student, I've found that a model like `qwen3:4b` is better than the free version of ChatGPT, and I've successfully gotten it to help me with my math homework.

## Requirements

- **Python** ≥ 3.11
- **[uv](https://docs.astral.sh/uv/)**
- **Ollama** or any OpenAI-compatible provider (LM Studio, vLLM, mlx-lm, oMLX, etc.)
- For best results, use a modern terminal like iTerm2

## Quickstart (Ollama)

```bash
# 1. Install PicoHarness
uv tool install git+https://github.com/ninjamar/picoharness

# 2. Pull a model (browse models at https://ollama.com/library)
ollama pull qwen3:4b

# 3. Generate a config (writes to ~/.ph/config.toml by default)
ph init

# 4. Start the chat UI
ph chat
```

The default config targets Ollama with `qwen3:2b` and enables all tools that don't require external services.

> Using LM Studio or another OpenAI-compatible server? Set `provider` to `host:port` in your config instead (e.g., `localhost:1234`).

## Configuration

Generate an initial config:
```bash
ph init                    # writes ~/.ph/config.toml
ph init ~/myconfig.toml    # custom path
```

Launch the TUI:
```bash
ph chat                         # uses ~/.ph/config.toml, first preset
ph chat my_preset               # explicit preset
ph --config ~/myconfig.toml chat my_preset   # custom config
```

A **preset** is a named `[section]` in the TOML file — useful for keeping separate profiles per model or use case. You can define as many as you want in one file.

### Config Fields

```toml
[base]
model = "qwen3:2b"           # Model identifier
provider = "ollama"          # "ollama" or "host:port"
think = false                # Enable chain-of-thought (for reasoning models)
show_think = true            # Display thinking tokens in the UI
context_length = 4096        # Context window size (higher = more RAM)
tools = [                    # Tools to enable (omit field = enable all)
    "read_file",
    "search_wikipedia",
    "read_webpage",          # Requires jina_reader_url
    "search_web",            # Requires searxng_url
]

# Required only if using web tools:
# searxng_url = "http://localhost:4000"
# jina_reader_url = "http://localhost:3001"
```

Multiple presets in one file are supported:
```bash
ph --config config.toml --preset search_wikipedia
```

### Available Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read a local file by path |
| `search_wikipedia` | Search and summarize Wikipedia articles |
| `read_webpage` | Fetch and extract readable text from a URL (requires Jina Reader) |
| `search_web` | Run a web search query (requires SearXNG) |

## Docker Services (Optional)

Requires Docker. `search_web` and `read_webpage` depend on self-hosted services.

```bash
ph service start     # Start services in background
ph service stop      # Stop services
```

This starts:
- **SearXNG** on port 4000 — private, self-hosted meta search engine
- **Jina Reader** on port 3001 — extracts readable text from web pages

Then add to your config:
```toml
searxng_url = "http://localhost:4000"
jina_reader_url = "http://localhost:3001"
```

## Diagnostics

Run `ph doctor` to check that your configured provider and optional services are reachable:

```bash
ph doctor           # check default preset
ph doctor my_preset # check a specific preset
```

This reports pass/fail for: your LLM provider, SearXNG (if configured), and Jina Reader (if configured).
