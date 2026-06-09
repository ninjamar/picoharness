import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unittest.mock import AsyncMock, MagicMock, patch

from backend.tools.web.wikipedia import GetWikipediaPage, SearchWikipedia

# SearchWikipedia tests


async def test_search_wikipedia_returns_formatted_results():
    """SearchWikipedia should return results formatted as numbered list."""
    mock_page = MagicMock()
    mock_page.search_meta = MagicMock()
    mock_page.search_meta.snippet = "Example snippet about the topic"

    with patch("backend.tools.web.wikipedia._wiki.search", new_callable=AsyncMock) as mock_search:
        mock_result = MagicMock()
        mock_result.pages = {"Example Article": mock_page}
        mock_search.return_value = mock_result

        tool = SearchWikipedia()
        result = await tool.execute(query="test query")

        assert "1. Example Article:" in result
        assert "Example snippet about the topic" in result


async def test_search_wikipedia_no_results():
    """SearchWikipedia should return a message when no results are found."""
    with patch("backend.tools.web.wikipedia._wiki.search", new_callable=AsyncMock) as mock_search:
        mock_result = MagicMock()
        mock_result.pages = {}
        mock_search.return_value = mock_result

        tool = SearchWikipedia()
        result = await tool.execute(query="nonexistent query xyz")

        assert "No Wikipedia results found" in result


async def test_search_wikipedia_strips_html_tags():
    """SearchWikipedia should strip HTML tags from snippets."""
    mock_page = MagicMock()
    mock_page.search_meta = MagicMock()
    mock_page.search_meta.snippet = "This is <b>bold</b> text with <em>emphasis</em>"

    with patch("backend.tools.web.wikipedia._wiki.search", new_callable=AsyncMock) as mock_search:
        mock_result = MagicMock()
        mock_result.pages = {"Test Article": mock_page}
        mock_search.return_value = mock_result

        tool = SearchWikipedia()
        result = await tool.execute(query="test query")

        assert "<b>" not in result
        assert "<em>" not in result
        assert "This is bold text with emphasis" in result


async def test_search_wikipedia_handles_missing_snippet():
    """SearchWikipedia should handle articles without search_meta."""
    mock_page = MagicMock()
    mock_page.search_meta = None

    with patch("backend.tools.web.wikipedia._wiki.search", new_callable=AsyncMock) as mock_search:
        mock_result = MagicMock()
        mock_result.pages = {"Article Without Meta": mock_page}
        mock_search.return_value = mock_result

        tool = SearchWikipedia()
        result = await tool.execute(query="test query")

        assert "1. Article Without Meta:" in result


# GetWikipediaPage tests


async def test_get_wikipedia_page_returns_summary():
    """GetWikipediaPage should return the page summary."""

    async def mock_summary():
        return "This is the article summary content."

    mock_page = MagicMock()
    mock_page.exists = AsyncMock(return_value=True)
    mock_page.summary = mock_summary()

    with patch("backend.tools.web.wikipedia._wiki.page", return_value=mock_page):
        tool = GetWikipediaPage()
        result = await tool.execute(title="Test Article")

        assert "This is the article summary content." == result


async def test_get_wikipedia_page_not_found():
    """GetWikipediaPage should return error message when page doesn't exist."""
    mock_page = AsyncMock()
    mock_page.exists = AsyncMock(return_value=False)

    with patch("backend.tools.web.wikipedia._wiki.page", return_value=mock_page):
        tool = GetWikipediaPage()
        result = await tool.execute(title="Nonexistent Page")

        assert "Wikipedia page not found" in result
        assert "Nonexistent Page" in result


async def test_get_wikipedia_page_respects_max_length():
    """GetWikipediaPage should truncate content to max_length."""
    long_content = "x" * 5000

    async def mock_summary():
        return long_content

    mock_page = MagicMock()
    mock_page.exists = AsyncMock(return_value=True)
    mock_page.summary = mock_summary()

    with patch("backend.tools.web.wikipedia._wiki.page", return_value=mock_page):
        tool = GetWikipediaPage()
        result = await tool.execute(title="Test Article", max_length=1000)

        assert len(result) == 1000
        assert result == "x" * 1000
