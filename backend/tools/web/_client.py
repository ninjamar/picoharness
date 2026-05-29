"""Internal HTTP client functions for SearXNG and Jina Reader services."""

from urllib.parse import quote

import aiohttp


async def fetch_search_results(query: str, num_results: int, base_url: str) -> list[dict[str, str]]:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{base_url}/search",
            params={"q": query, "format": "json"},
        ) as response:
            response.raise_for_status()
            data = await response.json()
            results = data.get("results", [])[:num_results]

            # Dict of url, title
            return [{"title": r.get("title", ""), "url": r.get("url", "")} for r in results]


async def fetch_page(url: str, base_url: str, truncate: int | None = 2000) -> str:
    async with aiohttp.ClientSession() as session:
        url = quote(url, safe="")  # safe="" to quote slashes
        async with session.get(f"{base_url}/{url}") as response:
            response.raise_for_status()
            text = await response.text()

    if truncate is not None and len(text) > truncate:
        return text[:truncate] + "\n[truncated]"

    return text
