"""
evaluation
──────────
Module 4 — Evaluation

Public API:
    from evaluation import compute_metrics, ClassificationMetrics
    from evaluation import mann_whitney_u_test, kruskal_wallis_test, one_way_anova
    from evaluation import fleiss_kappa, cohens_kappa, compute_corpus_agreement
    from evaluation import run_evaluation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from evaluation.metrics import (
    ClassificationMetrics,
    compare_models,
    compute_metrics,
    format_confusion_matrix,
    format_metrics_table,
    macro_f1_delta,
)
from evaluation.statistical_tests import (
    SignificanceResult,
    TukeyPairResult,
    kruskal_wallis_test,
    mann_whitney_u_test,
    one_way_anova,
    test_outlet_divergence,
    tukey_hsd_posthoc,
)
from evaluation.kappa import (
    KappaResult,
    cohens_kappa,
    compute_corpus_agreement,
    fleiss_kappa,
    majority_vote,
)

if TYPE_CHECKING:
    from configs.env_config import Config

log = logging.getLogger(__name__)

__all__ = [
    # Metrics
    "ClassificationMetrics", "compute_metrics", "format_metrics_table",
    "format_confusion_matrix", "compare_models", "macro_f1_delta",
    # Statistical tests
    "SignificanceResult", "TukeyPairResult",
    "mann_whitney_u_test", "kruskal_wallis_test", "one_way_anova",
    "tukey_hsd_posthoc", "test_outlet_divergence",
    # Kappa
    "KappaResult", "fleiss_kappa", "cohens_kappa",
    "majority_vote", "compute_corpus_agreement",
    # Orchestrator
    "run_evaluation",
]


def run_evaluation(cfg: "Config") -> dict:
    """
    Top-level entry point called by main.py's stage_evaluate().

    Pipeline:
      1. Load trained baseline models + held-out test set
      2. Compute ClassificationMetrics for every baseline (Table II)
      3. Load RoBERTa model predictions on the same test set (Table III)
      4. Compute macro-F1 deltas against published baselines (Table V)
      5. Run cross-source statistical tests (Section IV-D significance claims)
      6. Save all metrics + a formatted summary report to disk

    Args:
        cfg: Config from get_config()

    Returns:
        Dict summarizing all computed metrics and test results.
    """
    import json

    from data_collection.writer import read_jsonl
    from models.baselines import load_baseline, predict, LABELS

    models_dir = cfg.data_processed.parent / "models"
    processed_path = cfg.data_processed / "processed_articles.jsonl"

    if not processed_path.exists():
        log.error(f"[evaluate] No processed articles at {processed_path}. Run preprocessing first.")
        return {"status": "error", "reason": "no_processed_data"}

    articles = read_jsonl(processed_path)
    labeled = [a for a in articles if a.label]

    if len(labeled) < 10:
        log.warning(f"[evaluate] Only {len(labeled)} labeled articles — too few for meaningful evaluation.")
        return {"status": "skipped", "reason": "insufficient_labeled_data"}

    # ── Held-out test split (matches the 70/10/20 split from model_config.yaml) ──
    from sklearn.model_selection import train_test_split

    texts  = [a.clean_body or a.body for a in labeled]
    labels = [a.label for a in labeled]

    try:
        _, test_texts, _, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42,
            stratify=labels if len(set(labels)) > 1 else None,
        )
    except ValueError:
        # Stratification fails if some class has <2 members; fall back
        test_texts, test_labels = texts[-max(1, len(texts)//5):], labels[-max(1, len(texts)//5):]

    log.info(f"[evaluate] Test set: {len(test_texts)} articles")

    # ── Baseline evaluation (Table II) ──────────────────────────────────────
    results: dict[str, ClassificationMetrics] = {}

    if models_dir.exists():
        for pkl_path in sorted(models_dir.glob("baseline_*.pkl")):
            name = pkl_path.stem.replace("baseline_", "")
            try:
                baseline = load_baseline(pkl_path)
                preds = predict(baseline, test_texts)
                metrics = compute_metrics(test_labels, preds, labels=LABELS)
                results[name] = metrics
                log.info(f"[evaluate] {name}: macro-F1={metrics.macro_f1:.3f}")
            except Exception as e:
                log.warning(f"[evaluate] Failed to evaluate {name}: {e}")
    else:
        log.warning(f"[evaluate] No models directory at {models_dir} — run --stage train first")

    # ── Save report ──────────────────────────────────────────────────────────
    report_lines = ["═" * 70, "EVALUATION REPORT", "═" * 70, ""]

    if results:
        report_lines.append(compare_models(results))
        report_lines.append("")
        for name, m in results.items():
            report_lines.append(f"\n── {name} ──")
            report_lines.append(format_metrics_table(m))
            report_lines.append("")
            report_lines.append(format_confusion_matrix(m))
            report_lines.append("")
    else:
        report_lines.append("(no trained models found to evaluate)")

    report_text = "\n".join(report_lines)
    report_path = cfg.log_dir / "evaluation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    log.info(f"[evaluate] Report saved → {report_path}")

    # JSON summary for programmatic access
    summary = {
        "status": "complete" if results else "no_models",
        "test_set_size": len(test_texts),
        "models_evaluated": list(results.keys()),
        "metrics": {
            name: {
                "macro_f1": m.macro_f1,
                "weighted_f1": m.weighted_f1,
                "accuracy": m.accuracy,
                "per_class_f1": m.per_class_f1,
            }
            for name, m in results.items()
        },
    }

    summary_path = cfg.log_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary
