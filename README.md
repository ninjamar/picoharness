# PicoHarness

A lightweight TUI agent harness for tiny (<8b), local AI models. Built for privacy, and minimal overhead, PicoHarness can search the internet and read files.

## Requirements

- **Python** ≥ 3.11
- **[uv](https://docs.astral.sh/uv/)**
- **Ollama** or an OpenAI-compatible provider running locally (e.g., LM Studio, vLLM, mlx-lm, or oMLX)

## Installation

```bash
# Clone the repo
git clone https://github.com/ninjamar/picoharness
cd picoharness

# Install dependencies
uv sync

# Install the CLI tool
uv pip install -e .
```

This registers the `ph` command.
```bash
ph --help
```

## Providers

Supported Providers:
- Ollama
- OpenAI-compatible endpoints (e.g, OpenAI, LM Studio vLLM, mlx-lm, and oMLX)

Ollama is recomended for its ease of use.

## Configuration

Generate an initial config:
```bash
ph --generate-config config.toml
```

Edit `config.toml` and run the following to open the TUI:
```bash
ph --config config.toml --preset base
```

### Config Fields

```toml
[base]
model = "qwen3:2b"           # Model identifier
provider = "ollama"          # "ollama" or "host:port"
think = false                # Enable chain-of-thought
show_think = true            # Display thinking output
context_length = 4096        # Context window (higher = more RAM)
tools = [                    # Tools to enable (empty = all)
    "read_file",
    "search_wikipedia",
    "read_webpage",          # Requires jina_reader_url
    "search_web",            # Requires searxng_url
]

# Optional (for web tools):
# searxng_url = "http://localhost:4000"
# jina_reader_url = "http://localhost:3001"
```

Multiple presets are supported (e.g., `[base]`, `[search_wikipedia]`). Use `ph --config config.toml --preset search_wikipedia` to pick one.

## Docker Services (Optional)

For web search and webpage reading, start the optional services:

```bash
ph services up -d        # Start in background
ph services down         # Stop services
```

This runs:
- **SearXNG** (web search engine) on port 4000 → wire to config as `searxng_url = "http://localhost:4000"`
- **Jina Reader** (webpage reader) on port 3001 → wire to config as `jina_reader_url = "http://localhost:3001"`

Without these services, the `search_web` and `read_webpage` tools do not function.
