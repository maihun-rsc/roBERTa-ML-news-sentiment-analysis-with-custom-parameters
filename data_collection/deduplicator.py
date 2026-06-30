"""
deduplicator.py
───────────────
SHA-256 content deduplication across the entire collection run.

Two articles are duplicates if they share the same article_id
(which is SHA-256 of body+url). Wire syndication means the same
story appears on multiple outlet pages — we keep the first instance
and discard the rest, preserving source attribution.

The Deduplicator is stateful — instantiate once per pipeline run
and pass it to every outlet collector.
"""

from __future__ import annotations

import logging
from pathlib import Path

from data_collection.schema import Article

log = logging.getLogger(__name__)


class Deduplicator:
    """
    Thread-unsafe but sufficient for a sequential pipeline.

    If you parallelise outlet collection, wrap calls to .is_duplicate()
    with a threading.Lock().
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        """
        Args:
            persist_path: optional path to a text file of already-seen IDs.
                          Useful for resuming an interrupted collection run.
        """
        self._seen: set[str] = set()

        if persist_path and persist_path.exists():
            loaded = 0
            with open(persist_path, encoding="utf-8") as f:
                for line in f:
                    aid = line.strip()
                    if aid:
                        self._seen.add(aid)
                        loaded += 1
            log.info(f"[dedup] Loaded {loaded} seen IDs from {persist_path}")

        self._persist_path = persist_path
        self._total_seen = 0
        self._total_dupes = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def is_duplicate(self, article: Article) -> bool:
        """
        Returns True if article.article_id has been seen before.
        Registers the ID if not a duplicate.
        """
        self._total_seen += 1
        if article.article_id in self._seen:
            self._total_dupes += 1
            log.debug(f"[dedup] Duplicate: {article.article_id} ({article.url[:60]})")
            return True
        self._seen.add(article.article_id)
        return False

    def filter(self, articles: list[Article]) -> list[Article]:
        """
        Filter a list in-place, returning only non-duplicates.

        Args:
            articles: list of Article objects

        Returns:
            Deduplicated list (preserves order).
        """
        before = len(articles)
        unique = [a for a in articles if not self.is_duplicate(a)]
        removed = before - len(unique)
        if removed:
            log.info(f"[dedup] Removed {removed} duplicates from batch of {before}")
        return unique

    def persist(self) -> None:
        """Write the current seen-set to disk for resumable runs."""
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "w", encoding="utf-8") as f:
            for aid in sorted(self._seen):
                f.write(aid + "\n")
        log.info(f"[dedup] Persisted {len(self._seen)} IDs to {self._persist_path}")

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_processed": self._total_seen,
            "total_duplicates": self._total_dupes,
            "unique_ids": len(self._seen),
        }
