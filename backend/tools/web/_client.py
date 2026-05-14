"""Internal HTTP client functions for SearXNG and Jina Reader services."""

from urllib.parse import quote

import aiohttp

SEARXNG_BASE_URL = "http://localhost:4000"
JINA_READER_BASE_URL = "http://localhost:3001"


async def fetch_search_results(query: str, num_results: int) -> list[dict[str, str]]:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{SEARXNG_BASE_URL}/search",
            params={"q": query, "format": "json"},
        ) as response:
            response.raise_for_status()
            data = await response.json()
            results = data.get("results", [])[:num_results]

            # Dict of url, title
            return [{"title": r.get("title", ""), "url": r.get("url", "")} for r in results]


async def fetch_page(url: str, truncate: int | None = 2000) -> str:
    async with aiohttp.ClientSession() as session:
        url = quote(url, safe="")  # safe="" to quote slashes
        async with session.get(f"{JINA_READER_BASE_URL}/{url}") as response:
            response.raise_for_status()
            text = await response.text()

    if truncate is not None and len(text) > truncate:
        return text[:truncate] + "\n[truncated]"

    return text
