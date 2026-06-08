import urllib.request
from pathlib import Path

from frontend.config_io import load_config


def _check(label: str, ok: bool) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def _http_reachable(url: str, timeout: int = 3) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def doctor_main(config_path: Path, preset: str | None) -> int:
    """Check that provider and optional services are reachable."""
    cfg = load_config(config_path, preset)
    print(f"Checking preset '{preset or 'default'}' from {config_path}\n")
    results = []

    if cfg.provider == "ollama":
        results.append(_check("Ollama (http://localhost:11434)", _http_reachable("http://localhost:11434/api/tags")))
    else:
        results.append(_check(f"Provider ({cfg.provider})", _http_reachable(f"http://{cfg.provider}/v1/models")))

    if cfg.searxng_url:
        results.append(_check(f"SearXNG ({cfg.searxng_url})", _http_reachable(cfg.searxng_url)))
    if cfg.jina_reader_url:
        results.append(_check(f"Jina Reader ({cfg.jina_reader_url})", _http_reachable(cfg.jina_reader_url)))

    print()
    print("All checks passed." if all(results) else "Some checks failed.")
    return 0 if all(results) else 1
