import sys
from pathlib import Path
from unittest.mock import ANY

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from backend.tools.web.browse import ReadWebPage, SearchAndReadWeb, SearchWeb

# ReadWebPage tests


async def test_read_webpage_returns_content():
    """ReadWebPage should return the markdown content from the response."""
    expected_content = "# Example Page\n\nThis is example content."

    with patch("backend.tools.web.browse.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = expected_content
        tool = ReadWebPage(jina_reader_url="http://test-jina")
        result = await tool.execute(url="http://example.com")
        assert expected_content in result


async def test_read_webpage_propagates_http_error():
    """ReadWebPage should propagate HTTP errors from fetch_page."""
    with patch("backend.tools.web.browse.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=404, message="Not Found"
        )
        tool = ReadWebPage(jina_reader_url="http://test-jina")
        with pytest.raises(aiohttp.ClientResponseError):
            await tool.execute(url="http://example.com")


async def test_read_webpage_passes_truncate_arg():
    """ReadWebPage should pass truncate parameter to fetch_page."""
    with patch("backend.tools.web.browse.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = "content"
        tool = ReadWebPage(jina_reader_url="http://test-jina")
        await tool.execute(url="http://example.com", truncate=2000)

        mock_fetch.assert_called_once()
        assert mock_fetch.call_args.kwargs["truncate"] == 2000


# SearchWeb tests


async def test_search_web_returns_formatted_results():
    """SearchWeb should return results formatted as numbered list."""
    results = [
        {"title": "Example Result", "url": "http://example.com", "content": ""},
        {"title": "Another Result", "url": "http://example.org", "content": ""},
    ]

    with patch("backend.tools.web.browse.fetch_search_results", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = results
        tool = SearchWeb(searxng_url="http://test-searxng")
        result = await tool.execute(query="test query")

        assert "1. Example Result" in result
        assert "   URL: http://example.com" in result
        assert "2. Another Result" in result
        assert "   URL: http://example.org" in result


async def test_search_web_respects_num_results():
    """SearchWeb should only return the requested number of results."""
    results = [{"title": f"Result {i}", "url": f"http://example{i}.com", "content": ""} for i in range(5)]

    with patch("backend.tools.web.browse.fetch_search_results", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = results[:2]
        tool = SearchWeb(searxng_url="http://test-searxng")
        result = await tool.execute(query="test query", num_results=2)

        # Should only have results 1 and 2, not 3, 4, 5
        assert "1. Result 0" in result
        assert "2. Result 1" in result
        assert "3. Result" not in result


async def test_search_web_empty_results():
    """SearchWeb should return empty string when no results are found."""
    with patch("backend.tools.web.browse.fetch_search_results", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []
        tool = SearchWeb(searxng_url="http://test-searxng")
        result = await tool.execute(query="test query")

        assert result == ""


# SearchAndReadWeb tests


async def test_search_and_read_web_returns_combined_content():
    """SearchAndReadWeb should return content from all fetched pages."""
    with patch("backend.tools.web.browse.fetch_search_results", new_callable=AsyncMock) as mock_search:
        with patch("backend.tools.web.browse.fetch_page", new_callable=AsyncMock) as mock_read:
            mock_search.return_value = [
                {"title": "Page 1", "url": "http://example1.com", "content": ""},
                {"title": "Page 2", "url": "http://example2.com", "content": ""},
            ]
            mock_read.side_effect = ["Content from page 1", "Content from page 2"]

            tool = SearchAndReadWeb(searxng_url="http://test-searxng", jina_reader_url="http://test-jina")
            result = await tool.execute(query="test query")

            assert "Content from page 1" in result
            assert "Content from page 2" in result


async def test_search_and_read_web_includes_url_headings():
    """SearchAndReadWeb should include URLs as headings in the output."""
    with patch("backend.tools.web.browse.fetch_search_results", new_callable=AsyncMock) as mock_search:
        with patch("backend.tools.web.browse.fetch_page", new_callable=AsyncMock) as mock_read:
            mock_search.return_value = [
                {"title": "Page 1", "url": "http://example1.com", "content": ""},
                {"title": "Page 2", "url": "http://example2.com", "content": ""},
            ]
            mock_read.return_value = "Some content"

            tool = SearchAndReadWeb(searxng_url="http://test-searxng", jina_reader_url="http://test-jina")
            result = await tool.execute(query="test query")

            assert "http://example1.com" in result
            assert "http://example2.com" in result
            assert "##" in result  # heading format


async def test_search_and_read_web_handles_read_error_gracefully():
    """SearchAndReadWeb should include an error message for pages that fail to read."""
    with patch("backend.tools.web.browse.fetch_search_results", new_callable=AsyncMock) as mock_search:
        with patch("backend.tools.web.browse.fetch_page", new_callable=AsyncMock) as mock_read:
            mock_search.return_value = [
                {"title": "Page 1", "url": "http://example1.com", "content": ""},
                {"title": "Page 2", "url": "http://example2.com", "content": ""},
            ]
            mock_read.side_effect = [Exception("Network error"), "Content from page 2"]

            tool = SearchAndReadWeb(searxng_url="http://test-searxng", jina_reader_url="http://test-jina")
            result = await tool.execute(query="test query")

            assert "http://example1.com" in result
            assert "http://example2.com" in result
            assert "Error reading page" in result
            assert "Content from page 2" in result


async def test_search_and_read_web_respects_num_results():
    """SearchAndReadWeb should pass num_results to fetch_search_results."""
    with patch("backend.tools.web.browse.fetch_search_results", new_callable=AsyncMock) as mock_search:
        with patch("backend.tools.web.browse.fetch_page", new_callable=AsyncMock) as mock_read:
            mock_search.return_value = [{"title": "Page 1", "url": "http://example.com", "content": ""}]
            mock_read.return_value = "content"

            tool = SearchAndReadWeb(searxng_url="http://test-searxng", jina_reader_url="http://test-jina")
            await tool.execute(query="test query", num_results=2)

            mock_search.assert_called_once_with(ANY, "test query", 2, "http://test-searxng")


async def test_search_and_read_web_truncates_per_page():
    """SearchAndReadWeb should pass truncate=1000 to fetch_page."""
    with patch("backend.tools.web.browse.fetch_search_results", new_callable=AsyncMock) as mock_search:
        with patch("backend.tools.web.browse.fetch_page", new_callable=AsyncMock) as mock_read:
            mock_search.return_value = [
                {"title": "Page 1", "url": "http://example1.com", "content": ""},
                {"title": "Page 2", "url": "http://example2.com", "content": ""},
            ]
            mock_read.return_value = "Some content"

            tool = SearchAndReadWeb(searxng_url="http://test-searxng", jina_reader_url="http://test-jina")
            await tool.execute(query="test query")

            # Verify fetch_page was called with truncate=1000 for each URL
            assert mock_read.call_count == 2
            for call in mock_read.call_args_list:
                assert call.kwargs.get("truncate") == 1000
