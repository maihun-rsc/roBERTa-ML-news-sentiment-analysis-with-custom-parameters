"""
test_module5.py
───────────────
Unit tests for Module 5 — Analysis.

Pure pandas/numpy for data logic; matplotlib/seaborn rendering is tested
by confirming files are written, not by pixel-comparing images.

Run:
    python test_module5.py
    python test_module5.py -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _make_article(source, topic, label, entities=None):
    from data_collection.schema import Article, make_article_id
    body = f"Test body for {source} {topic} {label} " + "x" * 60
    return Article(
        article_id=make_article_id(body, source + topic + label + str(id(entities))),
        source=source, title="Title", body=body, url=f"https://{source}.com/x",
        date="2024-01-01T00:00:00+00:00", topic=topic, label=label,
        entities=entities or [],
    )


# ════════════════════════════════════════════════════════════════
#  Test: cross_source.py
# ════════════════════════════════════════════════════════════════
class TestCrossSource(unittest.TestCase):

    def test_compute_framing_distribution_basic(self):
        from analysis.cross_source import compute_framing_distribution
        articles = [
            _make_article("bbc", "politics", "Critical"),
            _make_article("bbc", "politics", "Critical"),
            _make_article("bbc", "politics", "Supportive"),
        ]
        dist = compute_framing_distribution(articles)
        self.assertEqual(dist.total, 3)
        self.assertEqual(dist.counts["Critical"], 2)
        self.assertAlmostEqual(dist.percentages["Critical"], 200/3, places=2)

    def test_compute_framing_distribution_empty(self):
        from analysis.cross_source import compute_framing_distribution
        dist = compute_framing_distribution([])
        self.assertEqual(dist.total, 0)
        self.assertEqual(dist.percentages["Critical"], 0.0)

    def test_compute_framing_distribution_all_labels_present_even_if_zero(self):
        from analysis.cross_source import compute_framing_distribution, LABELS
        articles = [_make_article("bbc", "politics", "Critical")]
        dist = compute_framing_distribution(articles)
        for label in LABELS:
            self.assertIn(label, dist.counts)

    def test_compute_outlet_distribution_table_shape(self):
        from analysis.cross_source import compute_outlet_distribution_table, LABELS
        articles = [
            _make_article("bbc", "politics", "Critical"),
            _make_article("fox", "politics", "Critical"),
            _make_article("fox", "politics", "Critical"),
        ]
        df = compute_outlet_distribution_table(articles)
        self.assertEqual(set(df.index), {"bbc", "fox"})
        self.assertEqual(list(df.columns), LABELS)

    def test_compute_outlet_distribution_table_topic_filter(self):
        from analysis.cross_source import compute_outlet_distribution_table
        articles = [
            _make_article("bbc", "politics", "Critical"),
            _make_article("bbc", "economy", "Supportive"),
        ]
        df = compute_outlet_distribution_table(articles, topic_filter="politics")
        self.assertAlmostEqual(df.loc["bbc", "Critical"], 100.0)

    def test_compute_outlet_topic_cube_shape(self):
        from analysis.cross_source import compute_outlet_topic_cube
        articles = [
            _make_article("bbc", "politics", "Critical"),
            _make_article("bbc", "economy", "Supportive"),
            _make_article("fox", "politics", "Critical"),
        ]
        cube = compute_outlet_topic_cube(
            articles, outlets=["bbc", "fox"], topics=["politics", "economy"], label="Critical"
        )
        self.assertEqual(cube.shape, (2, 2))
        self.assertAlmostEqual(cube.loc["bbc", "politics"], 100.0)

    def test_compute_outlet_topic_cube_nan_for_missing_data(self):
        """Outlet+topic combos with zero articles should be NaN, not 0% —
        0% implies data was observed and the rate happened to be zero,
        which is a different claim than 'no data available'."""
        import math
        from analysis.cross_source import compute_outlet_topic_cube
        articles = [_make_article("bbc", "politics", "Critical")]
        cube = compute_outlet_topic_cube(
            articles, outlets=["bbc"], topics=["politics", "climate"], label="Critical"
        )
        self.assertTrue(math.isnan(cube.loc["bbc", "climate"]))

    def test_format_table_iv_is_string(self):
        from analysis.cross_source import compute_outlet_distribution_table, format_table_iv
        articles = [_make_article("bbc", "politics", "Critical")]
        df = compute_outlet_distribution_table(articles)
        table_str = format_table_iv(df)
        self.assertIsInstance(table_str, str)
        self.assertIn("bbc", table_str)

    def test_find_outlet_extremes(self):
        from analysis.cross_source import compute_outlet_distribution_table, find_outlet_extremes
        articles = (
            [_make_article("bbc", "politics", "Critical")] * 2 +
            [_make_article("bbc", "politics", "Supportive")] * 8 +
            [_make_article("fox", "politics", "Critical")] * 9 +
            [_make_article("fox", "politics", "Supportive")] * 1
        )
        df = compute_outlet_distribution_table(articles)
        extremes = find_outlet_extremes(df, "Critical")
        self.assertEqual(extremes["highest"][0], "fox")
        self.assertEqual(extremes["lowest"][0], "bbc")

    def test_find_outlet_extremes_invalid_label_raises(self):
        from analysis.cross_source import compute_outlet_distribution_table, find_outlet_extremes
        df = compute_outlet_distribution_table([_make_article("bbc", "politics", "Critical")])
        with self.assertRaises(ValueError):
            find_outlet_extremes(df, "NotARealLabel")

    def test_generate_narrative_summary_mentions_outlets(self):
        from analysis.cross_source import compute_outlet_distribution_table, generate_narrative_summary
        articles = (
            [_make_article("bbc", "politics", "Critical")] * 2 +
            [_make_article("bbc", "politics", "Supportive")] * 8 +
            [_make_article("fox", "politics", "Critical")] * 9 +
            [_make_article("fox", "politics", "Supportive")] * 1
        )
        df = compute_outlet_distribution_table(articles)
        summary = generate_narrative_summary(df, labels=["Critical"])
        self.assertIn("BBC", summary.upper())
        self.assertIn("FOX", summary.upper())

    def test_plot_outlet_topic_heatmap_saves_file(self):
        from analysis.cross_source import compute_outlet_topic_cube, plot_outlet_topic_heatmap
        articles = [
            _make_article("bbc", "politics", "Critical"),
            _make_article("fox", "politics", "Supportive"),
        ]
        cube = compute_outlet_topic_cube(articles, label="Critical")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "heatmap.png"
            plot_outlet_topic_heatmap(cube, label="Critical", save_path=path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_plot_all_label_heatmaps_generates_one_per_label(self):
        from analysis.cross_source import plot_all_label_heatmaps, LABELS
        articles = [
            _make_article("bbc", "politics", "Critical"),
            _make_article("fox", "politics", "Supportive"),
            _make_article("bbc", "economy", "Neutral-Reporting"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_all_label_heatmaps(articles, output_dir=Path(tmpdir))
            self.assertEqual(len(paths), len(LABELS))
            for p in paths:
                self.assertTrue(p.exists())


# ════════════════════════════════════════════════════════════════
#  Test: entity_profiler.py
# ════════════════════════════════════════════════════════════════
class TestEntityProfiler(unittest.TestCase):

    def test_build_entity_profiles_basic(self):
        from analysis.entity_profiler import build_entity_profiles
        articles = [
            _make_article("bbc", "politics", "Critical", entities=["Boris Johnson"]),
            _make_article("bbc", "politics", "Critical", entities=["Boris Johnson"]),
            _make_article("bbc", "politics", "Supportive", entities=["Boris Johnson"]),
        ]
        profiles = build_entity_profiles(articles, min_mentions=1)
        self.assertIn("Boris Johnson", profiles)
        self.assertEqual(profiles["Boris Johnson"].total_mentions, 3)
        self.assertEqual(profiles["Boris Johnson"].dominant_label, "Critical")

    def test_build_entity_profiles_respects_min_mentions(self):
        from analysis.entity_profiler import build_entity_profiles
        articles = [
            _make_article("bbc", "politics", "Critical", entities=["Rare Entity"]),
        ]
        profiles = build_entity_profiles(articles, min_mentions=3)
        self.assertNotIn("Rare Entity", profiles)

    def test_build_entity_profiles_tracks_outlets(self):
        from analysis.entity_profiler import build_entity_profiles
        articles = [
            _make_article("bbc", "politics", "Critical", entities=["Macron"]),
            _make_article("cnn", "politics", "Supportive", entities=["Macron"]),
            _make_article("fox", "politics", "Critical", entities=["Macron"]),
        ]
        profiles = build_entity_profiles(articles, min_mentions=1)
        self.assertEqual(profiles["Macron"].outlets_mentioning, {"bbc", "cnn", "fox"})

    def test_build_entity_profiles_skips_unlabeled_articles(self):
        from analysis.entity_profiler import build_entity_profiles
        from data_collection.schema import Article, make_article_id
        body = "x" * 100
        unlabeled = Article(
            article_id=make_article_id(body), source="bbc", title="T", body=body,
            url="https://bbc.com", date="2024-01-01T00:00:00+00:00",
            topic="politics", label="", entities=["Ghost Entity"],
        )
        profiles = build_entity_profiles([unlabeled], min_mentions=1)
        self.assertNotIn("Ghost Entity", profiles)

    def test_profiles_to_dataframe_sorted_by_mentions(self):
        from analysis.entity_profiler import build_entity_profiles, profiles_to_dataframe
        articles = (
            [_make_article("bbc", "politics", "Critical", entities=["Popular Entity"])] * 5 +
            [_make_article("bbc", "politics", "Supportive", entities=["Less Popular"])] * 2
        )
        profiles = build_entity_profiles(articles, min_mentions=1)
        df = profiles_to_dataframe(profiles)
        self.assertEqual(df.iloc[0]["entity"], "Popular Entity")

    def test_profiles_to_dataframe_empty(self):
        from analysis.entity_profiler import profiles_to_dataframe
        df = profiles_to_dataframe({})
        self.assertTrue(df.empty)

    def test_find_cross_outlet_divergent_entities(self):
        from analysis.entity_profiler import build_entity_profiles, find_cross_outlet_divergent_entities
        articles = (
            # Entity framed Critical 100% by Fox
            [_make_article("fox", "politics", "Critical", entities=["Divergent Person"])] * 5 +
            # Same entity framed Supportive 100% by BBC
            [_make_article("bbc", "politics", "Supportive", entities=["Divergent Person"])] * 5
        )
        profiles = build_entity_profiles(articles, min_mentions=1)
        divergent = find_cross_outlet_divergent_entities(
            profiles, min_outlets=2, min_divergence_pct=50.0, label="Critical"
        )
        self.assertEqual(len(divergent), 1)
        self.assertEqual(divergent[0][0], "Divergent Person")

    def test_find_cross_outlet_divergent_entities_requires_min_outlets(self):
        from analysis.entity_profiler import build_entity_profiles, find_cross_outlet_divergent_entities
        articles = [_make_article("bbc", "politics", "Critical", entities=["Single Outlet Entity"])] * 5
        profiles = build_entity_profiles(articles, min_mentions=1)
        divergent = find_cross_outlet_divergent_entities(profiles, min_outlets=2)
        self.assertEqual(len(divergent), 0)

    def test_get_top_entities_by_label(self):
        from analysis.entity_profiler import build_entity_profiles, get_top_entities_by_label
        articles = (
            [_make_article("bbc", "politics", "Critical", entities=["Always Critical"])] * 5 +
            [_make_article("bbc", "politics", "Supportive", entities=["Always Supportive"])] * 5
        )
        profiles = build_entity_profiles(articles, min_mentions=1)
        top = get_top_entities_by_label(profiles, label="Critical", top_k=1)
        self.assertEqual(top[0][0], "Always Critical")
        self.assertAlmostEqual(top[0][1], 100.0)

    def test_format_entity_profile_is_string(self):
        from analysis.entity_profiler import build_entity_profiles, format_entity_profile
        articles = [_make_article("bbc", "politics", "Critical", entities=["Test Entity"])]
        profiles = build_entity_profiles(articles, min_mentions=1)
        text = format_entity_profile(profiles["Test Entity"])
        self.assertIsInstance(text, str)
        self.assertIn("Test Entity", text)
        self.assertIn("bbc", text)


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # headless backend — no display needed for tests

    verbosity = 2 if "-v" in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
