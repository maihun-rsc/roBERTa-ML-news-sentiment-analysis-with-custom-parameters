"""
test_module4.py
───────────────
Unit tests for Module 4 — Evaluation.

Pure numpy/scipy/statsmodels — no network dependency, no GPU dependency.
Run:
    python test_module4.py
    python test_module4.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ════════════════════════════════════════════════════════════════
#  Test: metrics.py
# ════════════════════════════════════════════════════════════════
class TestMetrics(unittest.TestCase):

    def test_compute_metrics_perfect_predictions(self):
        from evaluation.metrics import compute_metrics
        y_true = ["Supportive", "Critical", "Neutral-Reporting", "Alarmist"]
        y_pred = ["Supportive", "Critical", "Neutral-Reporting", "Alarmist"]
        m = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m.macro_f1, 1.0)
        self.assertAlmostEqual(m.accuracy, 1.0)

    def test_compute_metrics_all_wrong(self):
        from evaluation.metrics import compute_metrics
        y_true = ["Supportive", "Supportive"]
        y_pred = ["Critical", "Critical"]
        m = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m.accuracy, 0.0)

    def test_compute_metrics_mismatched_lengths_raises(self):
        from evaluation.metrics import compute_metrics
        with self.assertRaises(ValueError):
            compute_metrics(["Supportive"], ["Critical", "Alarmist"])

    def test_compute_metrics_empty_raises(self):
        from evaluation.metrics import compute_metrics
        with self.assertRaises(ValueError):
            compute_metrics([], [])

    def test_compute_metrics_per_class_fields_present(self):
        from evaluation.metrics import compute_metrics, LABELS
        y_true = ["Supportive", "Critical", "Critical", "Neutral-Reporting"]
        y_pred = ["Supportive", "Critical", "Supportive", "Neutral-Reporting"]
        m = compute_metrics(y_true, y_pred)
        for label in LABELS:
            self.assertIn(label, m.per_class_precision)
            self.assertIn(label, m.per_class_recall)
            self.assertIn(label, m.per_class_f1)
            self.assertIn(label, m.per_class_support)

    def test_compute_metrics_confusion_matrix_shape(self):
        from evaluation.metrics import compute_metrics, LABELS
        y_true = ["Supportive", "Critical"]
        y_pred = ["Supportive", "Critical"]
        m = compute_metrics(y_true, y_pred)
        self.assertEqual(m.confusion.shape, (len(LABELS), len(LABELS)))

    def test_compute_metrics_confusion_matrix_diagonal_for_perfect(self):
        from evaluation.metrics import compute_metrics, LABELS
        y_true = LABELS.copy()
        y_pred = LABELS.copy()
        m = compute_metrics(y_true, y_pred)
        off_diag_sum = m.confusion.sum() - m.confusion.trace()
        self.assertEqual(off_diag_sum, 0)

    def test_format_metrics_table_is_string(self):
        from evaluation.metrics import compute_metrics, format_metrics_table
        m = compute_metrics(["Supportive"], ["Supportive"])
        table = format_metrics_table(m)
        self.assertIsInstance(table, str)
        self.assertIn("Macro-F1", table)

    def test_format_confusion_matrix_is_string(self):
        from evaluation.metrics import compute_metrics, format_confusion_matrix
        m = compute_metrics(["Supportive", "Critical"], ["Supportive", "Critical"])
        table = format_confusion_matrix(m)
        self.assertIsInstance(table, str)

    def test_compare_models_sorted_by_macro_f1(self):
        from evaluation.metrics import compute_metrics, compare_models
        good = compute_metrics(["Supportive", "Critical"], ["Supportive", "Critical"])
        bad  = compute_metrics(["Supportive", "Critical"], ["Critical", "Supportive"])
        table = compare_models({"bad_model": bad, "good_model": good})
        self.assertLess(table.index("good_model"), table.index("bad_model"))

    def test_compare_models_empty(self):
        from evaluation.metrics import compare_models
        result = compare_models({})
        self.assertIn("no results", result.lower())

    def test_macro_f1_delta_positive_for_improvement(self):
        from evaluation.metrics import compute_metrics, macro_f1_delta
        baseline = compute_metrics(["Supportive", "Critical"], ["Critical", "Supportive"])
        proposed = compute_metrics(["Supportive", "Critical"], ["Supportive", "Critical"])
        delta = macro_f1_delta(baseline, proposed)
        self.assertGreater(delta, 0)

    def test_macro_f1_delta_matches_paper_scale(self):
        """Sanity check the delta is on a percentage-point scale (e.g. -5.4),
        matching Table V's 'Delta vs. Proposed' column format. Uses all 4
        labels so macro-F1 isn't diluted by absent classes (macro averaging
        divides by the full label set when `labels=` is given explicitly)."""
        from evaluation.metrics import compute_metrics, macro_f1_delta, LABELS
        baseline = compute_metrics(LABELS, list(reversed(LABELS)))  # all wrong
        proposed = compute_metrics(LABELS, LABELS)                   # all correct
        delta = macro_f1_delta(baseline, proposed)
        self.assertAlmostEqual(delta, 100.0)


# ════════════════════════════════════════════════════════════════
#  Test: statistical_tests.py
# ════════════════════════════════════════════════════════════════
class TestStatisticalTests(unittest.TestCase):

    def test_mann_whitney_identical_distributions_not_significant(self):
        from evaluation.statistical_tests import mann_whitney_u_test
        import random
        random.seed(42)
        group_a = [random.gauss(0.3, 0.05) for _ in range(30)]
        group_b = [random.gauss(0.3, 0.05) for _ in range(30)]
        result = mann_whitney_u_test(group_a, group_b)
        self.assertFalse(result.significant)

    def test_mann_whitney_clearly_different_distributions_significant(self):
        from evaluation.statistical_tests import mann_whitney_u_test
        group_a = [0.1, 0.12, 0.11, 0.13, 0.09, 0.10, 0.14, 0.08, 0.12, 0.11]
        group_b = [0.8, 0.82, 0.79, 0.81, 0.83, 0.78, 0.80, 0.84, 0.79, 0.82]
        result = mann_whitney_u_test(group_a, group_b)
        self.assertTrue(result.significant)
        self.assertLess(result.p_value, 0.001)

    def test_mann_whitney_requires_min_samples(self):
        from evaluation.statistical_tests import mann_whitney_u_test
        with self.assertRaises(ValueError):
            mann_whitney_u_test([0.1], [0.2, 0.3])

    def test_mann_whitney_interpretation_contains_labels(self):
        from evaluation.statistical_tests import mann_whitney_u_test
        result = mann_whitney_u_test(
            [0.1, 0.2, 0.15], [0.8, 0.9, 0.85],
            label_a="BBC", label_b="Fox News",
        )
        self.assertIn("BBC", result.interpretation)
        self.assertIn("Fox News", result.interpretation)

    def test_kruskal_wallis_requires_three_groups(self):
        from evaluation.statistical_tests import kruskal_wallis_test
        with self.assertRaises(ValueError):
            kruskal_wallis_test({"a": [1, 2], "b": [3, 4]})

    def test_kruskal_wallis_detects_difference(self):
        from evaluation.statistical_tests import kruskal_wallis_test
        groups = {
            "bbc":  [0.1, 0.12, 0.11, 0.13],
            "cnn":  [0.15, 0.14, 0.16, 0.13],
            "fox":  [0.8, 0.82, 0.79, 0.81],
        }
        result = kruskal_wallis_test(groups)
        self.assertTrue(result.significant)

    def test_kruskal_wallis_no_difference(self):
        from evaluation.statistical_tests import kruskal_wallis_test
        import random
        random.seed(1)
        groups = {
            "a": [random.gauss(0.5, 0.05) for _ in range(20)],
            "b": [random.gauss(0.5, 0.05) for _ in range(20)],
            "c": [random.gauss(0.5, 0.05) for _ in range(20)],
        }
        result = kruskal_wallis_test(groups)
        self.assertFalse(result.significant)

    def test_one_way_anova_requires_three_groups(self):
        from evaluation.statistical_tests import one_way_anova
        with self.assertRaises(ValueError):
            one_way_anova({"a": [1, 2, 3], "b": [4, 5, 6]})

    def test_one_way_anova_detects_difference(self):
        from evaluation.statistical_tests import one_way_anova
        groups = {
            "a": [1.0, 1.1, 0.9, 1.05],
            "b": [1.0, 0.95, 1.05, 1.0],
            "c": [5.0, 5.1, 4.9, 5.05],
        }
        result = one_way_anova(groups)
        self.assertTrue(result.significant)

    def test_tukey_hsd_posthoc_runs(self):
        from evaluation.statistical_tests import tukey_hsd_posthoc
        groups = {
            "bbc":  [0.1, 0.12, 0.11, 0.13, 0.10],
            "cnn":  [0.15, 0.14, 0.16, 0.13, 0.15],
            "fox":  [0.8, 0.82, 0.79, 0.81, 0.80],
        }
        pairs = tukey_hsd_posthoc(groups)
        self.assertGreater(len(pairs), 0)
        bbc_fox = next(
            (p for p in pairs if {p.group_a, p.group_b} == {"bbc", "fox"}), None
        )
        self.assertIsNotNone(bbc_fox)
        self.assertTrue(bbc_fox.significant)

    def test_outlet_divergence_full_pipeline(self):
        from evaluation.statistical_tests import test_outlet_divergence
        outlet_rates = {
            "bbc":  [0.14, 0.15, 0.13, 0.16, 0.14],
            "cnn":  [0.17, 0.16, 0.18, 0.17, 0.16],
            "fox":  [0.22, 0.21, 0.23, 0.20, 0.22],
            "rt":   [0.25, 0.24, 0.26, 0.25, 0.27],
        }
        result = test_outlet_divergence(outlet_rates)
        self.assertIn("omnibus_kruskal", result)
        self.assertIn("omnibus_anova", result)
        self.assertIn("pairwise", result)

    def test_outlet_divergence_skips_pairwise_if_omnibus_not_significant(self):
        from evaluation.statistical_tests import test_outlet_divergence
        import random
        random.seed(7)
        outlet_rates = {
            "a": [random.gauss(0.3, 0.02) for _ in range(15)],
            "b": [random.gauss(0.3, 0.02) for _ in range(15)],
            "c": [random.gauss(0.3, 0.02) for _ in range(15)],
        }
        result = test_outlet_divergence(outlet_rates)
        if not result["omnibus_kruskal"].significant:
            self.assertEqual(len(result["pairwise"]), 0)


# ════════════════════════════════════════════════════════════════
#  Test: kappa.py
# ════════════════════════════════════════════════════════════════
class TestKappa(unittest.TestCase):

    def test_fleiss_kappa_perfect_agreement(self):
        from evaluation.kappa import fleiss_kappa
        annotations = [
            ["Critical", "Critical", "Critical"],
            ["Supportive", "Supportive", "Supportive"],
            ["Alarmist", "Alarmist", "Alarmist"],
        ]
        result = fleiss_kappa(annotations)
        self.assertAlmostEqual(result.kappa, 1.0, places=4)
        self.assertEqual(result.interpretation, "almost perfect agreement")

    def test_fleiss_kappa_substantial_agreement_matches_paper(self):
        from evaluation.kappa import fleiss_kappa
        annotations = (
            [["Critical", "Critical", "Critical"]] * 20 +
            [["Supportive", "Supportive", "Supportive"]] * 20 +
            [["Neutral-Reporting", "Neutral-Reporting", "Neutral-Reporting"]] * 20 +
            [["Alarmist", "Alarmist", "Alarmist"]] * 15 +
            [["Critical", "Critical", "Alarmist"]] * 8 +
            [["Supportive", "Neutral-Reporting", "Supportive"]] * 7
        )
        result = fleiss_kappa(annotations)
        self.assertGreater(result.kappa, 0.6)
        self.assertIn(result.interpretation, ["substantial agreement", "almost perfect agreement"])

    def test_fleiss_kappa_requires_consistent_rater_count(self):
        from evaluation.kappa import fleiss_kappa
        annotations = [
            ["Critical", "Critical", "Critical"],
            ["Supportive", "Supportive"],
        ]
        with self.assertRaises(ValueError):
            fleiss_kappa(annotations)

    def test_fleiss_kappa_rejects_unknown_category(self):
        from evaluation.kappa import fleiss_kappa
        annotations = [["Critical", "Critical", "NotARealLabel"]]
        with self.assertRaises(ValueError):
            fleiss_kappa(annotations, categories=["Critical", "Supportive"])

    def test_fleiss_kappa_empty_input_raises(self):
        from evaluation.kappa import fleiss_kappa
        with self.assertRaises(ValueError):
            fleiss_kappa([])

    def test_cohens_kappa_perfect_agreement(self):
        from evaluation.kappa import cohens_kappa
        rater_a = ["Critical", "Supportive", "Alarmist", "Neutral-Reporting"]
        rater_b = ["Critical", "Supportive", "Alarmist", "Neutral-Reporting"]
        result = cohens_kappa(rater_a, rater_b)
        self.assertAlmostEqual(result.kappa, 1.0, places=4)

    def test_cohens_kappa_no_agreement_beyond_chance(self):
        from evaluation.kappa import cohens_kappa
        rater_a = ["Critical", "Critical", "Critical", "Critical"]
        rater_b = ["Supportive", "Supportive", "Supportive", "Supportive"]
        result = cohens_kappa(rater_a, rater_b)
        self.assertLessEqual(result.kappa, 0.0)

    def test_cohens_kappa_mismatched_lengths_raises(self):
        from evaluation.kappa import cohens_kappa
        with self.assertRaises(ValueError):
            cohens_kappa(["Critical"], ["Critical", "Supportive"])

    def test_majority_vote_clear_majority(self):
        from evaluation.kappa import majority_vote
        label, is_tie = majority_vote(["Critical", "Critical", "Supportive"])
        self.assertEqual(label, "Critical")
        self.assertFalse(is_tie)

    def test_majority_vote_three_way_disagreement(self):
        from evaluation.kappa import majority_vote
        label, is_tie = majority_vote(["Critical", "Supportive", "Alarmist"])
        self.assertIsNone(label)
        self.assertTrue(is_tie)

    def test_majority_vote_unanimous(self):
        from evaluation.kappa import majority_vote
        label, is_tie = majority_vote(["Neutral-Reporting"] * 3)
        self.assertEqual(label, "Neutral-Reporting")
        self.assertFalse(is_tie)

    def test_compute_corpus_agreement_full_report(self):
        from evaluation.kappa import compute_corpus_agreement
        annotations = (
            [["Critical", "Critical", "Critical"]] * 48 +
            [["Supportive", "Supportive", "Critical"]] * 1 +
            [["Critical", "Supportive", "Alarmist"]] * 1
        )
        report = compute_corpus_agreement(annotations)
        self.assertIn("fleiss_kappa", report)
        self.assertEqual(report["n_items"], 50)
        self.assertEqual(report["n_disagreements"], 1)
        self.assertAlmostEqual(report["disagreement_rate"], 1/50)

    def test_compute_corpus_agreement_disagreement_rate_under_paper_threshold(self):
        from evaluation.kappa import compute_corpus_agreement
        annotations = (
            [["Critical", "Critical", "Critical"]] * 96 +
            [["Critical", "Supportive", "Alarmist"]] * 4
        )
        report = compute_corpus_agreement(annotations)
        self.assertLessEqual(report["disagreement_rate"], 0.04)


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    verbosity = 2 if "-v" in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
