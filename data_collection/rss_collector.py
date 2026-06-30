"""
rss_collector.py
────────────────
Parses RSS/Atom feeds for each outlet and returns a list of
(url, title, summary, date, category) tuples — raw feed entries
before full-text scraping.

Keeps feedparser's quirks contained here so the scraper stays clean.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import requests

log = logging.getLogger(__name__)


@dataclass
class FeedEntry:
    url:      str
    title:    str
    summary:  str
    date:     str        # ISO 8601
    category: str        # raw RSS category tag


def _parse_date(entry: Any) -> str:
    """Best-effort date extraction from a feedparser entry."""
    # feedparser normalises to 'published_parsed' (time.struct_time) when possible
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        ts = getattr(entry, attr, None)
        if ts:
            try:
                dt = datetime(*ts[:6], tzinfo=timezone.utc)
                return dt.isoformat()
            except Exception:
                pass

    # Fallback: raw string fields
    for attr in ("published", "updated"):
        raw = getattr(entry, attr, None)
        if raw:
            try:
                return parsedate_to_datetime(raw).isoformat()
            except Exception:
                pass

    return datetime.now(timezone.utc).isoformat()


def _parse_category(entry: Any) -> str:
    """Extract the first category tag from a feedparser entry."""
    tags = getattr(entry, "tags", [])
    if tags:
        return getattr(tags[0], "term", "") or getattr(tags[0], "label", "")
    # Some feeds put category as a plain string
    return getattr(entry, "category", "")


def fetch_feed(
    outlet_name: str,
    feed_url: str,
    timeout: int = 15,
    user_agent: str | None = None,
) -> list[FeedEntry]:
    """
    Fetch and parse a single RSS/Atom feed URL.

    feedparser handles malformed XML gracefully — it never raises,
    it just returns an empty entries list and sets bozo=True.

    Args:
        outlet_name: slug for logging context
        feed_url:    full RSS URL
        timeout:     HTTP timeout in seconds
        user_agent:  optional UA override

    Returns:
        List of FeedEntry objects (may be empty on failure).
    """
    headers: dict[str, str] = {}
    if user_agent:
        headers["User-Agent"] = user_agent

    try:
        # feedparser can parse directly from URL but doesn't support
        # custom headers or timeout. We fetch raw bytes via requests instead.
        resp = requests.get(feed_url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except requests.exceptions.Timeout:
        log.warning(f"[{outlet_name}] Feed timeout: {feed_url}")
        return []
    except requests.exceptions.HTTPError as e:
        log.warning(f"[{outlet_name}] HTTP {e.response.status_code}: {feed_url}")
        return []
    except requests.exceptions.RequestException as e:
        log.warning(f"[{outlet_name}] Request error: {e}")
        return []
    except Exception as e:
        log.warning(f"[{outlet_name}] Unexpected feed error: {e}")
        return []

    if getattr(feed, "bozo", False):
        bozo_exc = getattr(feed, "bozo_exception", None)
        log.debug(f"[{outlet_name}] Malformed feed ({bozo_exc}); attempting anyway")

    entries: list[FeedEntry] = []
    for entry in feed.entries:
        url = getattr(entry, "link", "") or getattr(entry, "id", "")
        if not url:
            continue

        title   = getattr(entry, "title", "").strip()
        # summary may contain HTML — stripped later in cleaner.py
        summary = getattr(entry, "summary", "").strip()
        date    = _parse_date(entry)
        cat     = _parse_category(entry)

        entries.append(FeedEntry(
            url=url,
            title=title,
            summary=summary,
            date=date,
            category=cat,
        ))

    log.info(f"[{outlet_name}] Feed OK — {len(entries)} entries from {feed_url}")
    return entries


def collect_outlet_feeds(
    outlet_cfg: dict[str, Any],
    timeout: int = 15,
    user_agent: str | None = None,
    delay: float = 0.5,
) -> list[FeedEntry]:
    """
    Collect entries from ALL RSS feeds for a single outlet config,
    deduplicated by URL.

    Args:
        outlet_cfg:  one entry from outlets.json["outlets"]
        timeout:     per-request timeout
        user_agent:  UA string to use
        delay:       sleep between feed requests (courtesy delay)

    Returns:
        Deduplicated list of FeedEntry objects across all feeds.
    """
    name  = outlet_cfg["name"]
    feeds = outlet_cfg.get("rss_feeds", [])

    if not feeds:
        log.warning(f"[{name}] No RSS feeds configured")
        return []

    seen_urls: set[str] = set()
    all_entries: list[FeedEntry] = []

    for feed_url in feeds:
        entries = fetch_feed(name, feed_url, timeout=timeout, user_agent=user_agent)
        for e in entries:
            if e.url not in seen_urls:
                seen_urls.add(e.url)
                all_entries.append(e)
        if delay > 0:
            time.sleep(delay)

    log.info(f"[{name}] Total unique feed entries: {len(all_entries)}")
    return all_entries
