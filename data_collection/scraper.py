"""
scraper.py
──────────
Upgrades FeedEntry stubs to full Article objects by scraping article pages
via newspaper3k. Handles retries, rate limits, 403s, and empty bodies.

Strategy:
  1. Try newspaper3k on the URL
  2. On failure → use RSS summary as body (degraded but usable)
  3. If summary also empty → caller triggers fallback_loader

Never raises. Returns None if the article is unrecoverable.
"""

from __future__ import annotations

import logging
import random
import re
import time
from typing import Any

import requests

from data_collection.rss_collector import FeedEntry
from data_collection.schema import Article, infer_topic, make_article_id

log = logging.getLogger(__name__)

# ── Lazy import newspaper3k (heavy; only loaded when needed) ─────────────────
_newspaper_available: bool | None = None


def _get_newspaper() -> Any | None:
    global _newspaper_available
    if _newspaper_available is None:
        try:
            import newspaper  # noqa: F401
            _newspaper_available = True
        except ImportError:
            log.warning("newspaper3k not installed — scraping disabled, RSS-only mode")
            _newspaper_available = False
    return _newspaper_available


# ── HTML cleanup ─────────────────────────────────────────────────────────────

_TAG_RE  = re.compile(r"<[^>]+>")
_WS_RE   = re.compile(r"\s{2,}")


def _strip_html(text: str) -> str:
    """Quick HTML strip for RSS summaries (not full articles — cleaner.py handles those)."""
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


# ── Core scrape function ──────────────────────────────────────────────────────

def scrape_article(
    entry: FeedEntry,
    outlet_name: str,
    region: str = "",
    timeout: int = 15,
    user_agent: str | None = None,
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> Article | None:
    """
    Attempt to scrape full article text from entry.url.

    Falls back to RSS summary if scraping fails.
    Returns None only if both the scrape AND the summary are unusable.

    Args:
        entry:        FeedEntry from rss_collector
        outlet_name:  outlet slug ('bbc', 'cnn', …)
        region:       outlet region string
        timeout:      HTTP timeout per attempt
        user_agent:   UA string for requests
        max_retries:  number of scrape attempts before summary fallback
        backoff_base: exponential backoff multiplier

    Returns:
        Article dataclass or None.
    """
    body = ""
    authors: list[str] = []

    if _get_newspaper():
        body, authors = _try_newspaper(
            entry.url,
            outlet_name,
            timeout,
            user_agent,
            max_retries,
            backoff_base,
        )

    # ── Fallback to RSS summary ──────────────────────────────────────────────
    if not body or len(body.strip()) < 50:
        summary_clean = _strip_html(entry.summary)
        if len(summary_clean) >= 50:
            log.debug(f"[{outlet_name}] Using RSS summary for {entry.url}")
            body = summary_clean
        else:
            log.debug(f"[{outlet_name}] Unrecoverable — skipping {entry.url}")
            return None

    # ── Build Article ────────────────────────────────────────────────────────
    topic = infer_topic(
        text=entry.title + " " + body,
        rss_category=entry.category,
    )

    article = Article(
        article_id=make_article_id(body, entry.url),
        source=outlet_name,
        title=entry.title,
        body=body,
        url=entry.url,
        date=entry.date,
        topic=topic,
        region=region,
        summary=_strip_html(entry.summary),
        authors=authors,
        fallback_source="",
    )

    valid, reason = article.is_valid()
    if not valid:
        log.debug(f"[{outlet_name}] Article failed validation ({reason}): {entry.url}")
        return None

    return article


def _try_newspaper(
    url: str,
    outlet_name: str,
    timeout: int,
    user_agent: str | None,
    max_retries: int,
    backoff_base: float,
) -> tuple[str, list[str]]:
    """
    Attempt newspaper3k download with retry + exponential backoff.

    Returns:
        (body_text, authors) — both may be empty strings / empty list on failure.
    """
    from newspaper import Article as NpArticle, Config as NpConfig  # type: ignore

    config = NpConfig()
    config.request_timeout = timeout
    config.fetch_images = False
    config.memoize_articles = False
    if user_agent:
        config.browser_user_agent = user_agent

    for attempt in range(1, max_retries + 1):
        try:
            art = NpArticle(url, config=config)
            art.download()
            art.parse()

            body    = (art.text or "").strip()
            authors = art.authors or []

            if body:
                return body, authors

            log.debug(f"[{outlet_name}] Empty body on attempt {attempt}: {url}")

        except Exception as e:
            status = _extract_status(e)
            if status in (403, 401, 429):
                log.debug(f"[{outlet_name}] HTTP {status} — stopping retries for {url}")
                break
            log.debug(f"[{outlet_name}] Attempt {attempt}/{max_retries} failed: {e}")

        if attempt < max_retries:
            sleep_s = backoff_base ** attempt + random.uniform(0, 0.5)
            time.sleep(sleep_s)

    return "", []


def _extract_status(exc: Exception) -> int | None:
    """Pull HTTP status code out of various newspaper3k exception types."""
    msg = str(exc)
    match = re.search(r"\b(4\d{2}|5\d{2})\b", msg)
    return int(match.group(1)) if match else None


# ── Batch scraper ─────────────────────────────────────────────────────────────

def scrape_outlet(
    outlet_cfg: dict[str, Any],
    entries: list[FeedEntry],
    max_articles: int = 500,
    timeout: int = 15,
    user_agent: str | None = None,
    rate_delay: float = 0.8,
    max_retries: int = 3,
) -> tuple[list[Article], int]:
    """
    Scrape a batch of FeedEntries for one outlet.

    Args:
        outlet_cfg:   one entry from outlets.json["outlets"]
        entries:      list of FeedEntry from rss_collector
        max_articles: cap per outlet
        timeout:      per-request timeout
        user_agent:   UA string
        rate_delay:   sleep between articles
        max_retries:  per-article retry budget

    Returns:
        (scraped_articles, failed_count)
        — failed_count tells the caller how many need fallback.
    """
    name   = outlet_cfg["name"]
    region = outlet_cfg.get("region", "")

    articles: list[Article] = []
    failed  = 0

    entries_to_process = entries[:max_articles]
    log.info(f"[{name}] Scraping {len(entries_to_process)} entries …")

    try:
        for i, entry in enumerate(entries_to_process, 1):
            art = scrape_article(
                entry=entry,
                outlet_name=name,
                region=region,
                timeout=timeout,
                user_agent=user_agent,
                max_retries=max_retries,
            )

            if art:
                articles.append(art)
            else:
                failed += 1

            # Progress log every 50 articles
            if i % 50 == 0:
                log.info(f"[{name}] Progress: {i}/{len(entries_to_process)} "
                         f"({len(articles)} ok, {failed} failed)")

            if rate_delay > 0 and i < len(entries_to_process):
                time.sleep(rate_delay)
    except KeyboardInterrupt:
        log.warning(f"[{name}] Scraping interrupted by user. Saving {len(articles)} articles collected so far...")
    except Exception as e:
        log.error(f"[{name}] Unexpected error during scraping: {e}")

    log.info(f"[{name}] Scrape complete/interrupted: {len(articles)} ok / {failed} failed")
    return articles, failed
