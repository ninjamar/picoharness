import asyncio
from urllib.parse import quote

import aiohttp

from backend.tools.base import BaseTool

_timeout = aiohttp.ClientTimeout(total=30, connect=5)

session: aiohttp.ClientSession | None = None


def get_or_create_session() -> aiohttp.ClientSession:
    global session
    if session is None:
        session = aiohttp.ClientSession(timeout=_timeout)
    return session


async def close_session() -> None:
    global session
    if session is not None:
        await session.close()
        session = None


async def fetch_search_results(
    session: aiohttp.ClientSession, query: str, num_results: int, base_url: str
) -> list[dict[str, str]]:
    async with session.get(
        f"{base_url}/search",
        params={"q": query, "format": "json"},
    ) as response:
        response.raise_for_status()
        data = await response.json()
        results = data.get("results", [])[:num_results]

        # Dict of url, title
        return [{"title": r.get("title", ""), "url": r.get("url", "")} for r in results]


_JINA_HEADERS = {
    "x-cache-tolerance": "3600",
    "x-respond-timing": "html",
}


async def fetch_page(session: aiohttp.ClientSession, url: str, base_url: str, truncate: int | None = 2000) -> str:
    url = quote(url, safe="")  # safe="" to quote slashes
    async with session.get(f"{base_url}/{url}", headers=_JINA_HEADERS) as response:
        response.raise_for_status()
        text = await response.text()

    if truncate is not None and len(text) > truncate:
        return text[:truncate] + "\n[truncated]"

    return text


class ReadWebPage(BaseTool):
    """Tool to read a webpage as markdown"""

    name = "read_webpage"

    output_format = "none"

    def __init__(self, jina_reader_url, **kwargs: object) -> None:
        self.jina_reader_url = jina_reader_url
        self.session = get_or_create_session()

    async def _call(self, url: str) -> str:
        """
        Read the contents of a fully formed URL as markdown.

        Args:
            url: Fully formed url (includes protocol) of the webpage

        Returns:
            The contents of the webpage, as markdown
        """
        return await fetch_page(self.session, url, self.jina_reader_url)


class SearchWeb(BaseTool):
    """Tool to search the web using SearXNG"""

    name = "search_web"

    output_format = "all"

    def __init__(self, searxng_url: str, **kwargs: object) -> None:
        self.searxng_url = searxng_url
        self.session = get_or_create_session()

    async def _call(self, query: str, num_results: int = 5) -> str:
        """
        Search the web for a query and return top results.

        Args:
            query: The search query
            num_results: Number of results to return (default 5)

        Returns:
            A numbered list of results with format "N. Title: URL"
        """
        results = await fetch_search_results(self.session, query, num_results, self.searxng_url)
        if not results:
            return ""
        lines = [f"{i + 1}. {r['title']}: {r['url']}" for i, r in enumerate(results)]
        return "\n".join(lines)


class SearchAndReadWeb(BaseTool):
    """Tool to search the web and read the top results"""

    name = "search_and_read_web"

    output_format = "none"

    def __init__(
        self,
        searxng_url: str,
        jina_reader_url: str,
        **kwargs: object,
    ) -> None:
        self.searxng_url = searxng_url
        self.jina_reader_url = jina_reader_url
        self.session = get_or_create_session()

    async def _call(self, query: str, num_results: int = 3) -> str:
        """
        Search the web for a query and read the contents of top results.

        Args:
            query: The search query
            num_results: Number of results to fetch and read (default 3)

        Returns:
            Combined markdown content from all fetched pages, with each page under a heading
        """
        results = await fetch_search_results(self.session, query, num_results, self.searxng_url)

        async def _read_one(url: str) -> tuple[str, str]:
            try:
                content = await fetch_page(self.session, url, self.jina_reader_url, truncate=500)
                return url, content
            except Exception as exc:
                return url, f"Error reading page: {exc}"

        pages = await asyncio.gather(*[_read_one(r["url"]) for r in results])

        sections = [f"## {url}\n{content}" for url, content in pages]
        return "\n\n".join(sections)
