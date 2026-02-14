"""DuckDuckGo search integration for debate agents."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ddgs import DDGS

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    href: str
    body: str


@dataclass
class SearchResponse:
    """Response from a search query."""

    query: str
    results: list[SearchResult]
    success: bool
    error: str | None = None


def search_web(
    query: str, max_results: int = 5, region: str = "wt-wt", safesearch: str = "moderate"
) -> SearchResponse:
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query string (can include site: operators for targeted search)
        max_results: Maximum number of results to return (default: 5)
        region: Region for search results (default: "wt-wt" for worldwide)
        safesearch: Safe search level ("on", "moderate", "off")

    Returns:
        SearchResponse containing results or error information

    Examples:
        >>> # General search
        >>> search_web("mRNA vaccine safety")

        >>> # Site-specific search for conspiracy sources
        >>> search_web("site:reddit.com/r/conspiracy vaccine microchips")

        >>> # Site-specific search for scientific sources
        >>> search_web("site:cdc.gov mRNA vaccine efficacy")
    """
    try:
        logger.info(f"Searching DuckDuckGo: {query}")

        with DDGS() as ddgs:
            raw_results = list(
                ddgs.text(query, region=region, safesearch=safesearch, max_results=max_results)
            )

        results = [
            SearchResult(title=r.get("title", ""), href=r.get("href", ""), body=r.get("body", ""))
            for r in raw_results
        ]

        logger.info(f"Found {len(results)} results for query: {query}")
        return SearchResponse(query=query, results=results, success=True)

    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        logger.error(error_msg)
        return SearchResponse(query=query, results=[], success=False, error=error_msg)


def format_search_results_for_prompt(response: SearchResponse, max_chars: int = 1000) -> str:
    """
    Format search results into a concise string for LLM context.

    Args:
        response: SearchResponse object
        max_chars: Maximum characters to include in output (default: 1000)

    Returns:
        Formatted string with search results
    """
    if not response.success:
        return f"[Search failed: {response.error}]"

    if not response.results:
        return "[No search results found]"

    lines = [f"Search results for: {response.query}\n"]
    char_count = len(lines[0])

    for i, result in enumerate(response.results, 1):
        # Truncate body if needed
        body = result.body[:200] if len(result.body) > 200 else result.body
        result_text = f"{i}. {result.title}\n   {body}\n   Source: {result.href}\n\n"

        if char_count + len(result_text) > max_chars:
            lines.append(f"[+{len(response.results) - i + 1} more results...]")
            break

        lines.append(result_text)
        char_count += len(result_text)

    return "".join(lines)
