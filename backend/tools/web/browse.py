import asyncio

from backend.tools.base import BaseTool

from ._client import fetch_page, fetch_search_results


class ReadWebPage(BaseTool):
    """Tool to read a webpage as markdown"""

    name = "read_webpage"

    output_format = "none"

    async def _call(self, url: str) -> str:
        """
        Read the contents of a fully formed URL as markdown.

        Args:
            url: Fully formed url (includes protocol) of the webpage

        Returns:
            The contents of the webpage, as markdown
        """
        return await fetch_page(url)


class SearchWeb(BaseTool):
    """Tool to search the web using SearXNG"""

    name = "search_web"

    output_format = "all"

    async def _call(self, query: str, num_results: int = 5) -> str:
        """
        Search the web for a query and return top results.

        Args:
            query: The search query
            num_results: Number of results to return (default 5)

        Returns:
            A numbered list of results with format "N. Title: URL"
        """
        results = await fetch_search_results(query, num_results)
        if not results:
            return ""
        lines = [f"{i + 1}. {r['title']}: {r['url']}" for i, r in enumerate(results)]
        return "\n".join(lines)


class SearchAndReadWeb(BaseTool):
    """Tool to search the web and read the top results"""

    name = "search_and_read_web"

    output_format = "none"

    async def _call(self, query: str, num_results: int = 3) -> str:
        """
        Search the web for a query and read the contents of top results.

        Args:
            query: The search query
            num_results: Number of results to fetch and read (default 3)

        Returns:
            Combined markdown content from all fetched pages, with each page under a heading
        """
        results = await fetch_search_results(query, num_results)

        async def _read_one(url: str) -> tuple[str, str]:
            try:
                content = await fetch_page(url, truncate=500)
                return url, content
            except Exception as exc:
                return url, f"Error reading page: {exc}"

        pages = await asyncio.gather(*[_read_one(r["url"]) for r in results])

        sections = [f"## {url}\n{content}" for url, content in pages]
        return "\n\n".join(sections)
