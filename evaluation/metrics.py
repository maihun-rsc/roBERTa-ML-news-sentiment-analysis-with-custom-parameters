"""
metrics.py
──────────
Core classification metrics per context.md Module 4 spec:
    "Macro-F1, confusion matrix, per-class precision and recall"

This reproduces exactly the metrics reported in the paper's Tables II, III,
and IV — same metric definitions, so results computed here are directly
comparable to the paper's published numbers (macro-F1 = 0.814 for the
proposed entity-aware RoBERTa, 0.668 for the strongest classical baseline).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

log = logging.getLogger(__name__)

LABELS = ["Supportive", "Critical", "Neutral-Reporting", "Alarmist"]


@dataclass
class ClassificationMetrics:
    """Full metric bundle for one model's predictions on one dataset split."""
    macro_f1:       float
    weighted_f1:    float
    accuracy:       float
    per_class_precision: dict[str, float] = field(default_factory=dict)
    per_class_recall:    dict[str, float] = field(default_factory=dict)
    per_class_f1:        dict[str, float] = field(default_factory=dict)
    per_class_support:   dict[str, int]   = field(default_factory=dict)
    confusion:      np.ndarray = field(default_factory=lambda: np.zeros((4, 4), dtype=int))
    labels_used:    list[str] = field(default_factory=lambda: list(LABELS))


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> ClassificationMetrics:
    """
    Compute the full metric bundle for a set of predictions.

    Args:
        y_true:  gold framing labels
        y_pred:  predicted framing labels
        labels:  label set to use for confusion matrix ordering
                 (default: the canonical 4-class LABELS order, so matrices
                 are always comparable across different model runs even if
                 one run happens to never predict a given class)

    Returns:
        ClassificationMetrics with everything needed for Tables II-IV.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    if not y_true:
        raise ValueError("Cannot compute metrics on empty input")

    labels = labels or LABELS

    macro_f1    = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    accuracy    = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return ClassificationMetrics(
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        accuracy=accuracy,
        per_class_precision={l: float(p) for l, p in zip(labels, precision)},
        per_class_recall={l: float(r) for l, r in zip(labels, recall)},
        per_class_f1={l: float(f) for l, f in zip(labels, f1)},
        per_class_support={l: int(s) for l, s in zip(labels, support)},
        confusion=cm,
        labels_used=labels,
    )


def format_metrics_table(metrics: ClassificationMetrics) -> str:
    """
    Render a metrics bundle as a readable text table — matches the
    column layout of the paper's Table III (per-class F1 + macro-F1).

    Args:
        metrics: ClassificationMetrics from compute_metrics

    Returns:
        Formatted string table.
    """
    lines = []
    lines.append(f"{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("-" * 62)
    for label in metrics.labels_used:
        lines.append(
            f"{label:<20} "
            f"{metrics.per_class_precision[label]:>10.3f} "
            f"{metrics.per_class_recall[label]:>10.3f} "
            f"{metrics.per_class_f1[label]:>10.3f} "
            f"{metrics.per_class_support[label]:>10d}"
        )
    lines.append("-" * 62)
    lines.append(f"{'Macro-F1':<20} {'':<10} {'':<10} {metrics.macro_f1:>10.3f}")
    lines.append(f"{'Weighted-F1':<20} {'':<10} {'':<10} {metrics.weighted_f1:>10.3f}")
    lines.append(f"{'Accuracy':<20} {'':<10} {'':<10} {metrics.accuracy:>10.3f}")
    return "\n".join(lines)


def format_confusion_matrix(metrics: ClassificationMetrics) -> str:
    """
    Render the confusion matrix as a readable text grid, rows=true,
    columns=predicted (standard convention).

    Returns:
        Formatted string table.
    """
    labels = metrics.labels_used
    cm = metrics.confusion

    # Truncate long label names for column headers to keep the table narrow
    short = [l[:4] for l in labels]

    lines = []
    header = f"{'true\\pred':<20}" + "".join(f"{s:>8}" for s in short)
    lines.append(header)
    lines.append("-" * len(header))
    for i, label in enumerate(labels):
        row = f"{label:<20}" + "".join(f"{cm[i,j]:>8d}" for j in range(len(labels)))
        lines.append(row)
    return "\n".join(lines)


def compare_models(
    results: dict[str, ClassificationMetrics],
) -> str:
    """
    Side-by-side comparison table — reproduces the structure of Table III
    (multiple models, per-class F1 columns, macro-F1 column).

    Args:
        results: {model_name: ClassificationMetrics}

    Returns:
        Formatted comparison table, models sorted by macro-F1 descending.
    """
    if not results:
        return "(no results to compare)"

    sample = next(iter(results.values()))
    labels = sample.labels_used

    lines = []
    header = f"{'Model':<35}" + "".join(f"{l[:8]:>10}" for l in labels) + f"{'Macro-F1':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    sorted_results = sorted(results.items(), key=lambda kv: -kv[1].macro_f1)
    for name, m in sorted_results:
        row = f"{name:<35}"
        for label in labels:
            row += f"{m.per_class_f1.get(label, 0.0):>10.3f}"
        row += f"{m.macro_f1:>10.3f}"
        lines.append(row)

    return "\n".join(lines)


def macro_f1_delta(
    baseline: ClassificationMetrics,
    proposed: ClassificationMetrics,
) -> float:
    """
    Compute the absolute macro-F1 point improvement, matching the paper's
    "Delta vs. Proposed" column in Table V.

    Args:
        baseline: ClassificationMetrics for the baseline model
        proposed: ClassificationMetrics for the proposed model

    Returns:
        Percentage point difference (proposed - baseline) * 100.
    """
    return (proposed.macro_f1 - baseline.macro_f1) * 100
