"""
data_collection
───────────────
Module 1 — Data Collection Pipeline

Public API (import from here, not from submodules):

    from data_collection import run_collection, read_all_raw
    from data_collection import Article, make_article_id, infer_topic
    from data_collection import Deduplicator, ArticleWriter
"""

from data_collection.schema import Article, make_article_id, infer_topic
from data_collection.deduplicator import Deduplicator
from data_collection.writer import run_collection, read_all_raw, ArticleWriter

__all__ = [
    "Article",
    "make_article_id",
    "infer_topic",
    "Deduplicator",
    "ArticleWriter",
    "run_collection",
    "read_all_raw",
]
