"""
Web scraping and search helpers.
"""

import asyncio
import io
import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from config import SEARCH_USER_AGENT

logger = logging.getLogger("meeting-analyzer")


async def search_web(query: str, max_results: int = 5, delay: float = 0.0) -> list[dict[str, str]]:
    """Search DuckDuckGo via the duckduckgo-search library."""
    if delay > 0:
        await asyncio.sleep(delay)
    try:
        from duckduckgo_search import DDGS
        from duckduckgo_search.exceptions import DuckDuckGoSearchException
    except ImportError:
        logger.warning("duckduckgo-search not installed; web search unavailable")
        return []
    try:
        raw = await asyncio.to_thread(
            lambda: list(DDGS().text(query, max_results=max_results, backend="auto"))
        )
    except Exception as exc:
        logger.warning("DDG search failed for %r: %s", query[:80], exc)
        return []
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        url = (item.get("href") or "").strip()
        if not url or url in seen:
            continue
        netloc = urlparse(url).netloc.lower()
        if not netloc:
            continue
        seen.add(url)
        results.append({
            "title": (item.get("title") or url).strip(),
            "url": url,
            "domain": netloc,
            "query": query,
        })
    return results


def extract_web_text_sync(content: bytes, content_type: str, url: str) -> tuple[str, str]:
    lower_url = url.lower()
    if "application/pdf" in content_type or lower_url.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            parts = []
            for page in reader.pages[:25]:
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(text.strip())
            title = parts[0].splitlines()[0][:180] if parts else url
            return title, "\n\n".join(parts)[:20000]
        except Exception:
            return url, ""

    html_text = content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html_text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "svg", "noscript"]):
        tag.decompose()
    title = ""
    if soup.title and soup.title.string:
        title = " ".join(soup.title.string.split())
    blocks = []
    for el in soup.select("main h1, main h2, main h3, main p, main li, article h1, article h2, article h3, article p, article li"):
        text = " ".join(el.get_text(" ", strip=True).split())
        if text:
            blocks.append(text)
    if not blocks:
        for el in soup.select("h1, h2, h3, p, li"):
            text = " ".join(el.get_text(" ", strip=True).split())
            if text:
                blocks.append(text)
    return title or url, "\n\n".join(blocks)[:20000]


async def fetch_web_source(source: dict[str, str]) -> dict[str, Any] | None:
    headers = {"User-Agent": SEARCH_USER_AGENT}
    try:
        async with httpx.AsyncClient(timeout=30.0, headers=headers, follow_redirects=True) as client:
            resp = await client.get(source["url"])
            resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        title, text = await asyncio.to_thread(
            extract_web_text_sync,
            resp.content,
            content_type,
            source["url"],
        )
        if not text.strip():
            return None
        return {
            **source,
            "title": title or source["title"],
            "content": text[:12000],
        }
    except Exception as exc:
        logger.warning("Web fetch failed for %s: %s", source.get("url"), exc)
        return None


def extract_urls_from_text(text: str) -> list[str]:
    """Return unique http(s) URLs found in text, in order, stripping trailing punctuation."""
    found = re.findall(r'https?://[^\s\)\]>\"\']+', text or "")
    seen: set[str] = set()
    result = []
    for url in found:
        url = url.rstrip(".,;:")
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


async def collect_research_sources(
    queries: list[str],
    *,
    per_query_results: int,
    max_sources: int,
) -> list[dict[str, Any]]:
    search_batches = await asyncio.gather(
        *[search_web(q, max_results=per_query_results, delay=i * 0.3) for i, q in enumerate(queries)],
        return_exceptions=True,
    )
    ranked: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    domain_counts: dict[str, int] = {}
    for batch in search_batches:
        if isinstance(batch, Exception):
            continue
        for result in batch:
            if result["url"] in seen_urls:
                continue
            if domain_counts.get(result["domain"], 0) >= 2:
                continue
            seen_urls.add(result["url"])
            domain_counts[result["domain"]] = domain_counts.get(result["domain"], 0) + 1
            ranked.append(result)
            if len(ranked) >= max_sources:
                break
        if len(ranked) >= max_sources:
            break
    fetched = await asyncio.gather(*[fetch_web_source(item) for item in ranked], return_exceptions=True)
    sources = []
    for item in fetched:
        if isinstance(item, dict):
            sources.append(item)
    for idx, source in enumerate(sources, start=1):
        source["id"] = f"S{idx}"
    return sources
