import re

import wikipediaapi

from backend.tools.base import BaseTool

_USER_AGENT = "localai/1.0"
# TODO: This always instantiates when imported
_wiki = wikipediaapi.AsyncWikipedia(user_agent=_USER_AGENT, language="en")


class SearchWikipedia(BaseTool):
    name = "search_wikipedia"
    output_format = "all"

    async def _call(self, query: str, limit: int = 5) -> str:
        """
        Search Wikipedia for articles matching a query.

        Args:
            query: The search query to find Wikipedia articles
            limit: Maximum number of results to return (default 5)

        Returns:
            A numbered list of article titles with short text snippets
            Format:
                Title: Text
        """
        results = await _wiki.search(query, limit=limit)
        if not results.pages:
            return "No Wikipedia results found."
        lines = []
        for i, (title, page) in enumerate(results.pages.items()):
            snippet = re.sub(r"<[^>]+>", "", page.search_meta.snippet) if page.search_meta else ""
            lines.append(f"{i + 1}. {title}: {snippet}")
        return "\n".join(lines)


class GetWikipediaPage(BaseTool):
    name = "get_wikipedia_page"
    output_format = "none"

    async def _call(self, title: str, max_length: int | None = 3000) -> str:
        """
        Fetch the summary of a Wikipedia page by its title.

        Args:
            title: The Wikipedia article title to retrieve (use exact title from search results)
            max_length: Maximum number of characters to return (default 3000). If it is None, then the entire
            article will be returned.

        Returns:
            The article's summary text, truncated to max_length characters
        """
        page = _wiki.page(title)
        if not await page.exists():
            return f"Wikipedia page not found: '{title}'. Try searching first with search_wikipedia."
        summary = await page.summary
        if max_length is None:
            return summary
        return summary[:max_length]
