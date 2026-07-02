"""
test_module1.py
───────────────
Smoke tests for Module 1 — Data Collection.
No network required. Tests all components with mocked/synthetic data.

Run:
    python test_module1.py
    python test_module1.py -v   # verbose
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make project root importable
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ════════════════════════════════════════════════════════════════
#  Test: schema.py
# ════════════════════════════════════════════════════════════════
class TestSchema(unittest.TestCase):

    def test_article_creation(self):
        from data_collection.schema import Article, make_article_id
        body = "This is a test article body with enough text to pass validation."
        art  = Article(
            article_id=make_article_id(body, "https://example.com/article"),
            source="bbc",
            title="Test Headline",
            body=body,
            url="https://example.com/article",
            date="2024-01-15T10:00:00+00:00",
            topic="politics",
        )
        self.assertEqual(art.source, "bbc")
        self.assertEqual(len(art.article_id), 16)

    def test_article_validation_passes(self):
        from data_collection.schema import Article, make_article_id
        body = "A" * 100
        art  = Article(
            article_id=make_article_id(body),
            source="cnn",
            title="Valid Title",
            body=body,
            url="https://cnn.com/test",
            date="2024-01-01T00:00:00+00:00",
            topic="economy",
        )
        valid, reason = art.is_valid()
        self.assertTrue(valid, f"Expected valid, got: {reason}")

    def test_article_validation_fails_empty_body(self):
        from data_collection.schema import Article, make_article_id
        art = Article(
            article_id="abc123",
            source="fox",
            title="Title",
            body="short",
            url="https://fox.com",
            date="2024-01-01T00:00:00+00:00",
            topic="politics",
        )
        valid, reason = art.is_valid()
        self.assertFalse(valid)
        self.assertIn("short", reason)

    def test_article_validation_fails_no_title(self):
        from data_collection.schema import Article
        art = Article(
            article_id="abc123",
            source="abc",
            title="",
            body="A" * 100,
            url="https://abc.com",
            date="2024-01-01T00:00:00+00:00",
            topic="politics",
        )
        valid, reason = art.is_valid()
        self.assertFalse(valid)
        self.assertIn("title", reason)

    def test_article_json_roundtrip(self):
        from data_collection.schema import Article, make_article_id
        body = "Roundtrip test body content that is long enough to be valid."
        art  = Article(
            article_id=make_article_id(body),
            source="wion",
            title="Roundtrip Test",
            body=body,
            url="https://wion.com/test",
            date="2024-06-01T12:00:00+00:00",
            topic="conflict",
            entities=["India", "UN"],
        )
        restored = Article.from_json(art.to_json())
        self.assertEqual(art.article_id, restored.article_id)
        self.assertEqual(art.entities,   restored.entities)
        self.assertEqual(art.topic,      restored.topic)

    def test_make_article_id_deterministic(self):
        from data_collection.schema import make_article_id
        body = "Deterministic test content"
        id1  = make_article_id(body, "https://url.com")
        id2  = make_article_id(body, "https://url.com")
        self.assertEqual(id1, id2)
        self.assertEqual(len(id1), 16)

    def test_make_article_id_different_bodies(self):
        from data_collection.schema import make_article_id
        id1 = make_article_id("Body A", "https://url.com")
        id2 = make_article_id("Body B", "https://url.com")
        self.assertNotEqual(id1, id2)

    def test_infer_topic_from_text(self):
        from data_collection.schema import infer_topic
        self.assertEqual(infer_topic("The election results are in, the vote count continues"),  "elections")
        self.assertEqual(infer_topic("GDP growth and inflation rates surprise analysts"),        "economy")
        self.assertEqual(infer_topic("Military strike kills troops near the border"),            "conflict")

    def test_infer_topic_from_rss_category(self):
        from data_collection.schema import infer_topic
        result = infer_topic("some text here", rss_category="politics")
        self.assertEqual(result, "politics")

    def test_infer_topic_fallback(self):
        from data_collection.schema import infer_topic
        result = infer_topic("The weather was pleasant and the flowers bloomed.")
        self.assertEqual(result, "general")


# ════════════════════════════════════════════════════════════════
#  Test: deduplicator.py
# ════════════════════════════════════════════════════════════════
class TestDeduplicator(unittest.TestCase):

    def _make_article(self, body: str, source: str = "bbc") -> "Article":
        from data_collection.schema import Article, make_article_id
        return Article(
            article_id=make_article_id(body, "https://example.com"),
            source=source,
            title="Title",
            body=body,
            url="https://example.com",
            date="2024-01-01T00:00:00+00:00",
            topic="politics",
        )

    def test_first_occurrence_not_duplicate(self):
        from data_collection.deduplicator import Deduplicator
        d   = Deduplicator()
        art = self._make_article("Unique body content here")
        self.assertFalse(d.is_duplicate(art))

    def test_second_occurrence_is_duplicate(self):
        from data_collection.deduplicator import Deduplicator
        d   = Deduplicator()
        art = self._make_article("Same body content duplicated")
        d.is_duplicate(art)
        self.assertTrue(d.is_duplicate(art))

    def test_filter_removes_duplicates(self):
        from data_collection.deduplicator import Deduplicator
        d  = Deduplicator()
        a1 = self._make_article("Article body one")
        a2 = self._make_article("Article body two")
        a3 = self._make_article("Article body one")  # duplicate of a1
        result = d.filter([a1, a2, a3])
        self.assertEqual(len(result), 2)
        ids = [r.article_id for r in result]
        self.assertNotIn(a3.article_id, ids[1:])  # a3 removed

    def test_cross_outlet_dedup(self):
        """Same body appearing on two outlets is a wire duplicate — deduplicate."""
        from data_collection.deduplicator import Deduplicator
        d   = Deduplicator()
        body = "Wire story that appeared on multiple outlets"
        a1  = self._make_article(body, source="bbc")
        a2  = self._make_article(body, source="cnn")
        result = d.filter([a1, a2])
        self.assertEqual(len(result), 1)

    def test_stats(self):
        from data_collection.deduplicator import Deduplicator
        d   = Deduplicator()
        a1  = self._make_article("Story one")
        a2  = self._make_article("Story one")   # duplicate
        d.filter([a1, a2])
        stats = d.stats
        self.assertEqual(stats["total_processed"],  2)
        self.assertEqual(stats["total_duplicates"], 1)
        self.assertEqual(stats["unique_ids"],       1)

    def test_persist_and_reload(self):
        from data_collection.deduplicator import Deduplicator
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "seen.txt"

            # First run
            d1  = Deduplicator(persist_path=persist_path)
            art = self._make_article("Persistent article body")
            d1.is_duplicate(art)
            d1.persist()

            # Second run — same article should now be a duplicate
            d2     = Deduplicator(persist_path=persist_path)
            art2   = self._make_article("Persistent article body")
            result = d2.is_duplicate(art2)
            self.assertTrue(result, "Article seen in previous run should be duplicate")


# ════════════════════════════════════════════════════════════════
#  Test: writer.py (JSONL I/O — no network)
# ════════════════════════════════════════════════════════════════
class TestWriter(unittest.TestCase):

    def _make_articles(self, n: int) -> list:
        from data_collection.schema import Article, make_article_id
        arts = []
        for i in range(n):
            body = f"Article body number {i} with enough content to be valid and pass schema checks."
            arts.append(Article(
                article_id=make_article_id(body, f"https://example.com/{i}"),
                source="bbc",
                title=f"Article {i}",
                body=body,
                url=f"https://example.com/{i}",
                date="2024-03-01T00:00:00+00:00",
                topic="politics",
            ))
        return arts

    def test_write_and_read_roundtrip(self):
        from data_collection.writer import ArticleWriter, read_jsonl
        arts = self._make_articles(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            with ArticleWriter(path) as w:
                w.write_many(arts)
            self.assertEqual(w.count, 5)

            restored = read_jsonl(path)
            self.assertEqual(len(restored), 5)
            for orig, rest in zip(arts, restored):
                self.assertEqual(orig.article_id, rest.article_id)
                self.assertEqual(orig.body,       rest.body)

    def test_append_mode(self):
        from data_collection.writer import ArticleWriter, read_jsonl
        arts = self._make_articles(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "append_test.jsonl"
            with ArticleWriter(path) as w:
                w.write_many(arts[:2])
            with ArticleWriter(path) as w:
                w.write_many(arts[2:])

            all_articles = read_jsonl(path)
            self.assertEqual(len(all_articles), 3)

    def test_read_all_raw(self):
        from data_collection.writer import ArticleWriter, read_all_raw
        arts = self._make_articles(6)
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            with ArticleWriter(raw_dir / "bbc_20240101.jsonl") as w:
                w.write_many(arts[:3])
            with ArticleWriter(raw_dir / "cnn_20240101.jsonl") as w:
                w.write_many(arts[3:])

            all_arts = read_all_raw(raw_dir)
            self.assertEqual(len(all_arts), 6)

    def test_malformed_jsonl_line_skipped(self):
        from data_collection.writer import read_jsonl
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            with open(path, "w") as f:
                f.write('{"article_id":"abc","source":"bbc","title":"T","body":"' + "B"*100 + '","url":"https://x.com","date":"2024-01-01T00:00:00+00:00","topic":"politics"}\n')
                f.write('not valid json at all\n')
                f.write('{"article_id":"def","source":"cnn","title":"T2","body":"' + "C"*100 + '","url":"https://y.com","date":"2024-01-01T00:00:00+00:00","topic":"economy"}\n')
            result = read_jsonl(path)
            self.assertEqual(len(result), 2)


# ════════════════════════════════════════════════════════════════
#  Test: rss_collector.py (mocked HTTP)
# ════════════════════════════════════════════════════════════════
class TestRSSCollector(unittest.TestCase):

    MOCK_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Test Feed</title>
        <item>
          <title>Test Article One</title>
          <link>https://example.com/article/1</link>
          <description>Summary of article one.</description>
          <pubDate>Mon, 15 Jan 2024 10:00:00 GMT</pubDate>
          <category>Politics</category>
        </item>
        <item>
          <title>Test Article Two</title>
          <link>https://example.com/article/2</link>
          <description>Summary of article two.</description>
          <pubDate>Mon, 15 Jan 2024 11:00:00 GMT</pubDate>
          <category>Economy</category>
        </item>
      </channel>
    </rss>"""

    def test_fetch_feed_returns_entries(self):
        from data_collection.rss_collector import fetch_feed

        mock_resp = MagicMock()
        mock_resp.content = self.MOCK_RSS
        mock_resp.raise_for_status = MagicMock()

        with patch("data_collection.rss_collector.requests.get", return_value=mock_resp):
            entries = fetch_feed("test_outlet", "https://fake-rss.com/feed.xml")

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].title, "Test Article One")
        self.assertEqual(entries[1].category, "Economy")
        self.assertIn("2024", entries[0].date)

    def test_fetch_feed_handles_404(self):
        from data_collection.rss_collector import fetch_feed
        import requests as req_lib

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req_lib.exceptions.HTTPError(
            response=MagicMock(status_code=404)
        )
        with patch("data_collection.rss_collector.requests.get", return_value=mock_resp):
            entries = fetch_feed("test_outlet", "https://bad-url.com/feed.xml")

        self.assertEqual(entries, [])

    def test_fetch_feed_handles_timeout(self):
        from data_collection.rss_collector import fetch_feed
        import requests as req_lib

        with patch(
            "data_collection.rss_collector.requests.get",
            side_effect=req_lib.exceptions.Timeout,
        ):
            entries = fetch_feed("test_outlet", "https://slow-server.com/feed.xml")

        self.assertEqual(entries, [])

    def test_collect_outlet_deduplicates_across_feeds(self):
        from data_collection.rss_collector import collect_outlet_feeds

        mock_resp = MagicMock()
        mock_resp.content = self.MOCK_RSS
        mock_resp.raise_for_status = MagicMock()

        outlet_cfg = {
            "name": "testoutlet",
            "rss_feeds": [
                "https://fake1.com/rss.xml",
                "https://fake2.com/rss.xml",   # same content
            ],
        }
        with patch("data_collection.rss_collector.requests.get", return_value=mock_resp):
            with patch("data_collection.rss_collector.time.sleep"):
                entries = collect_outlet_feeds(outlet_cfg, delay=0)

        # 2 unique URLs — not 4 (2 feeds × 2 articles)
        urls = [e.url for e in entries]
        self.assertEqual(len(set(urls)), len(entries))
        self.assertEqual(len(entries), 2)


# ════════════════════════════════════════════════════════════════
#  Test: scraper.py (mocked newspaper3k)
# ════════════════════════════════════════════════════════════════
class TestScraper(unittest.TestCase):

    def _make_entry(self, url: str = "https://bbc.com/news/test", summary: str = "") -> "FeedEntry":
        from data_collection.rss_collector import FeedEntry
        return FeedEntry(
            url=url,
            title="Test Headline BBC News",
            summary=summary or "Short summary of the article for testing purposes.",
            date="2024-03-15T10:00:00+00:00",
            category="Politics",
        )

    def test_scrape_uses_newspaper_when_available(self):
        from data_collection.scraper import scrape_article

        mock_np_article = MagicMock()
        mock_np_article.text    = "This is the full article body text. " * 10
        mock_np_article.authors = ["John Smith"]

        with patch("data_collection.scraper._newspaper_available", True):
            with patch("data_collection.scraper._try_newspaper",
                       return_value=(mock_np_article.text, mock_np_article.authors)):
                art = scrape_article(
                    self._make_entry(),
                    outlet_name="bbc",
                    region="UK",
                )

        self.assertIsNotNone(art)
        self.assertIn("full article", art.body)
        self.assertEqual(art.authors, ["John Smith"])
        self.assertEqual(art.source, "bbc")

    def test_scrape_falls_back_to_summary(self):
        from data_collection.scraper import scrape_article

        long_summary = "This is a long enough RSS summary that serves as the article body. " * 3

        with patch("data_collection.scraper._newspaper_available", True):
            with patch("data_collection.scraper._try_newspaper", return_value=("", [])):
                art = scrape_article(
                    self._make_entry(summary=long_summary),
                    outlet_name="cnn",
                )

        self.assertIsNotNone(art)
        # Body should come from summary
        self.assertGreater(len(art.body), 50)

    def test_scrape_returns_none_when_both_fail(self):
        from data_collection.scraper import scrape_article

        with patch("data_collection.scraper._newspaper_available", True):
            with patch("data_collection.scraper._try_newspaper", return_value=("", [])):
                art = scrape_article(
                    self._make_entry(summary="Too short"),
                    outlet_name="fox",
                )
        self.assertIsNone(art)

    def test_inferred_topic_from_category(self):
        from data_collection.scraper import scrape_article

        body = "Long enough article body for the politician's vote. " * 5

        with patch("data_collection.scraper._newspaper_available", True):
            with patch("data_collection.scraper._try_newspaper", return_value=(body, [])):
                art = scrape_article(
                    self._make_entry(),
                    outlet_name="bbc",
                )

        self.assertIsNotNone(art)
        self.assertIn(art.topic, ["politics", "elections", "general", "economy", "conflict", "climate", "crime"])


# ════════════════════════════════════════════════════════════════
#  Test: env_config.py
# ════════════════════════════════════════════════════════════════
class TestEnvConfig(unittest.TestCase):

    def test_get_config_returns_config(self):
        from configs.env_config import get_config, Config
        cfg = get_config()
        self.assertIsInstance(cfg, Config)
        self.assertIn(cfg.env, ("kaggle", "antigravity", "local"))
        self.assertIn(cfg.device, ("cuda", "mps", "cpu"))

    def test_config_creates_directories(self):
        from configs.env_config import Config
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(env="local", device="cpu", project_root=Path(tmpdir))
            self.assertTrue(cfg.data_raw.exists())
            self.assertTrue(cfg.data_processed.exists())
            self.assertTrue(cfg.data_fallback.exists())
            self.assertTrue(cfg.log_dir.exists())

    def test_batch_size_cpu(self):
        from configs.env_config import Config
        cfg = Config(env="local", device="cpu")
        self.assertEqual(cfg.batch_size, cfg.batch_size_cpu)

    def test_batch_size_gpu(self):
        from configs.env_config import Config
        cfg = Config(env="local", device="cuda")
        self.assertEqual(cfg.batch_size, cfg.batch_size_gpu)


# ════════════════════════════════════════════════════════════════
#  Run
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    verbosity = 2 if "-v" in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
