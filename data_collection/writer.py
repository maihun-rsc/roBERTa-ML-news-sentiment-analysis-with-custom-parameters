"""
writer.py
─────────
JSONL writer — one article per line, one file per outlet per run.
Also houses the top-level run_collection() pipeline that stitches
rss_collector → scraper → fallback_loader → deduplicator → writer.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from configs.env_config import Config
from data_collection.deduplicator import Deduplicator
from data_collection.fallback_loader import load_fallback
from data_collection.rss_collector import collect_outlet_feeds
from data_collection.schema import Article
from data_collection.scraper import scrape_outlet

log = logging.getLogger(__name__)


# ── JSONL writer ──────────────────────────────────────────────────────────────

class ArticleWriter:
    """
    Append-mode JSONL writer. Opens file lazily on first write.
    Call .close() explicitly or use as a context manager.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._f    = None
        self._count = 0

    def __enter__(self) -> "ArticleWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self._path, "a", encoding="utf-8")
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def write(self, article: Article) -> None:
        if self._f is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self._path, "a", encoding="utf-8")
        self._f.write(article.to_json() + "\n")
        self._count += 1

    def write_many(self, articles: list[Article]) -> None:
        for art in articles:
            self.write(art)

    def close(self) -> None:
        if self._f:
            self._f.flush()
            self._f.close()
            self._f = None

    @property
    def count(self) -> int:
        return self._count


# ── JSONL reader ──────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[Article]:
    """Load all Article objects from a JSONL file."""
    articles: list[Article] = []
    if not path.exists():
        return articles
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                articles.append(Article.from_json(line))
            except (json.JSONDecodeError, TypeError) as e:
                log.warning(f"[reader] Skipping malformed line {i} in {path}: {e}")
    return articles


def read_all_raw(raw_dir: Path) -> list[Article]:
    """Read every JSONL file in the raw directory into one list."""
    all_articles: list[Article] = []
    for jl in sorted(raw_dir.glob("*.jsonl")):
        loaded = read_jsonl(jl)
        log.info(f"[reader] {jl.name}: {len(loaded)} articles")
        all_articles.extend(loaded)
    return all_articles


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_collection(
    cfg: Config,
    outlets_cfg: list[dict[str, Any]],
    user_agents: list[str],
    seed: int = 42,
) -> dict[str, int]:
    """
    Full data collection pipeline for all outlets.

    For each outlet:
      1. Parse RSS feeds
      2. Scrape full articles
      3. If below target → trigger fallback dataset
      4. Deduplicate
      5. Write to JSONL

    Args:
        cfg:          Config from get_config()
        outlets_cfg:  list of outlet dicts from outlets.json
        user_agents:  rotating UA list from outlets.json
        seed:         random seed for fallback sampling

    Returns:
        Dict mapping outlet slug → article count written.
    """
    rng       = random.Random(seed)
    dedup     = Deduplicator(
        persist_path=cfg.log_dir / "seen_ids.txt"
    )
    run_ts    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    counts: dict[str, int] = {}

    for outlet_cfg in outlets_cfg:
        name = outlet_cfg["name"]
        log.info(f"\n{'─'*60}")
        log.info(f"  Processing outlet: {outlet_cfg['display']}")
        log.info(f"{'─'*60}")

        ua = rng.choice(user_agents)

        # ── Step 1: RSS ──────────────────────────────────────────────────────
        feed_entries = collect_outlet_feeds(
            outlet_cfg=outlet_cfg,
            timeout=cfg.request_timeout,
            user_agent=ua,
            delay=cfg.rate_limit_delay,
        )

        # ── Step 2: Scrape ───────────────────────────────────────────────────
        scraped, failed = [], 0
        if feed_entries:
            scraped, failed = scrape_outlet(
                outlet_cfg=outlet_cfg,
                entries=feed_entries,
                max_articles=cfg.max_articles_per_outlet,
                timeout=cfg.request_timeout,
                user_agent=ua,
                rate_delay=cfg.rate_limit_delay,
                max_retries=cfg.max_retries,
            )

        # ── Step 3: Fallback ─────────────────────────────────────────────────
        fallback_arts: list[Article] = []

        if len(scraped) == 0:
            shortfall = cfg.max_articles_per_outlet
            log.info(f"[{name}] Scraped 0 articles — triggering fallback for {shortfall} articles")
            fallback_arts = load_fallback(
                outlet_cfg=outlet_cfg,
                shortfall=shortfall,
                seed=seed,
            )

        all_articles = scraped + fallback_arts

        # ── Step 4: Dedup ────────────────────────────────────────────────────
        unique = dedup.filter(all_articles)
        log.info(
            f"[{name}] After dedup: {len(unique)}/{len(all_articles)} unique "
            f"({len(all_articles)-len(unique)} dupes removed)"
        )

        # ── Step 5: Write ────────────────────────────────────────────────────
        out_path = cfg.data_raw / f"{name}_{run_ts}.jsonl"
        writer = ArticleWriter(out_path)
        with writer:
            writer.write_many(unique)

        counts[name] = writer.count
        log.info(f"[{name}] Written {writer.count} articles → {out_path}")

    # Persist dedup state for resumable runs
    dedup.persist()
    log.info(f"\n[pipeline] Dedup stats: {dedup.stats}")
    log.info(f"[pipeline] Collection complete. Total articles: {sum(counts.values())}")
    log.info(f"[pipeline] Per outlet: {counts}")

    return counts
