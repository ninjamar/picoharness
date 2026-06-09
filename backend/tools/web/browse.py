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

        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")} for r in results
        ]


_JINA_HEADERS = {
    "x-cache-tolerance": "3600",
    "x-respond-timing": "html",
}


async def fetch_page(session: aiohttp.ClientSession, url: str, base_url: str, truncate: int | None = 4000) -> str:
    url = quote(url, safe="")  # safe="" to quote slashes
    async with session.get(f"{base_url}/{url}", headers=_JINA_HEADERS) as response:
        response.raise_for_status()
        if truncate is None:
            return await response.text()
        # Stream and stop early — response.text() reads the full body then decodes
        # synchronously, which blocks the event loop for large pages.
        charset = response.charset or "utf-8"
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.content.iter_chunked(4096):
            chunks.append(chunk)
            total += len(chunk)
            if total >= truncate * 4:  # 4 bytes/char worst-case UTF-8
                break
    text = b"".join(chunks).decode(charset, errors="replace")
    if len(text) > truncate:
        return text[:truncate] + "\n[truncated]"
    return text


class ReadWebPage(BaseTool):
    """Tool to read a webpage as markdown"""

    name = "read_webpage"

    output_format = "none"

    def __init__(self, jina_reader_url, **kwargs: object) -> None:
        self.jina_reader_url = jina_reader_url
        self.session = get_or_create_session()

    async def _call(self, url: str, truncate: int = 4000) -> str:
        """
        Read the contents of a fully formed URL as markdown.

        Args:
            url: Fully formed url (includes protocol) of the webpage
            truncate: Maximum characters to return (default 4000). Higher values get more content but use more context.

        Returns:
            The contents of the webpage, as markdown
        """
        return await fetch_page(self.session, url, self.jina_reader_url, truncate=truncate)


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
            query: The search query.
            num_results: Number of results to return (default 5)

        Returns:
            A numbered list of results with titles, URLs, and snippets
        """
        results = await fetch_search_results(self.session, query, num_results, self.searxng_url)
        if not results:
            return ""
        lines = []
        for i, r in enumerate(results):
            lines.append(f"{i + 1}. {r['title']}")
            lines.append(f"   URL: {r['url']}")
            if r.get("content"):
                snippet = r["content"][:200]
                lines.append(f"   Snippet: {snippet}")
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
                content = await fetch_page(self.session, url, self.jina_reader_url, truncate=1000)
                return url, content
            except Exception as exc:
                return url, f"Error reading page: {exc}"

        pages = await asyncio.gather(*[_read_one(r["url"]) for r in results])

        sections = [f"## {url}\n{content}" for url, content in pages]
        return "\n\n".join(sections)


class SummarizeWebPage(BaseTool):
    """Tool to fetch and summarize a webpage"""

    name = "summarize_webpage"

    output_format = "all"

    def __init__(self, jina_reader_url, summarizer_agent=None, **kwargs: object) -> None:
        self.jina_reader_url = jina_reader_url
        self.summarizer_agent = summarizer_agent
        self.session = get_or_create_session()

    async def _call(self, url: str, query: str, summary_length: int = 200) -> str:
        """
        Fetch a webpage and summarize it focused on a specific query.

        Args:
            url: Fully formed URL of the webpage
            query: What to focus on when summarizing
            summary_length: Target length in words (default 200)

        Returns:
            A concise summary focused on the query
        """
        content = await fetch_page(self.session, url, self.jina_reader_url, truncate=8000)
        if not self.summarizer_agent:
            return content[:2000]
        prompt = f"Summarize in at most {summary_length} words, focusing only on: {query}\n\n{content}"
        return await self.summarizer_agent.run(prompt)


class SearchAndSummarizeWeb(BaseTool):
    """Tool to search the web and summarize the top results"""

    name = "search_and_summarize_web"

    output_format = "all"

    def __init__(
        self,
        searxng_url: str,
        jina_reader_url: str,
        summarizer_agent=None,
        **kwargs: object,
    ) -> None:
        self.searxng_url = searxng_url
        self.jina_reader_url = jina_reader_url
        self.summarizer_agent = summarizer_agent
        self.session = get_or_create_session()

    async def _call(self, query: str, num_results: int = 3, summary_length: int = 150) -> str:
        """
        Search the web for a query and summarize the top results.

        Args:
            query: The search query
            num_results: Number of results to fetch and summarize (default 3)
            summary_length: Target length per summary in words (default 150)

        Returns:
            Summaries of the top results, one per heading
        """
        results = await fetch_search_results(self.session, query, num_results, self.searxng_url)

        async def _fetch_and_summarize(url: str) -> tuple[str, str]:
            try:
                content = await fetch_page(self.session, url, self.jina_reader_url, truncate=8000)
                if not self.summarizer_agent:
                    summary = content[:1500]
                else:
                    prompt = f"Summarize in at most {summary_length} words: {content}"
                    summary = await self.summarizer_agent.run(prompt)
                return url, summary
            except Exception as exc:
                return url, f"Error: {exc}"

        summaries = await asyncio.gather(*[_fetch_and_summarize(r["url"]) for r in results])

        sections = [f"## {url}\n{summary}" for url, summary in summaries]
        return "\n\n".join(sections)
