"""
kappa.py
────────
Inter-annotator agreement metrics per context.md Module 4 spec:
    "Custom Fleiss' Kappa — no library implements it cleanly for our
     annotation format; written from scratch."

Reproduces the paper's reported Fleiss' Kappa of 0.71 (substantial
agreement) for the 3-annotator, 4-label framing scheme described in
Section III-B (Dataset Description) and III-C (Annotation Protocol).

Both Fleiss' (N annotators) and Cohen's (exactly 2 annotators) are
implemented since the annotation protocol allows adjudication by a
single lead annotator on 3-way disagreements (mentioned in context.md
— "items with three-way disagreement... were adjudicated by the lead
annotator"), which is effectively a 2-annotator agreement check between
the lead and the majority vote.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# Standard interpretation bands (Landis & Koch, 1977) — the conventional
# reference scale cited whenever a kappa value is reported in NLP papers.
_INTERPRETATION_BANDS: list[tuple[float, float, str]] = [
    (-1.00,  0.00, "poor agreement"),
    ( 0.00,  0.20, "slight agreement"),
    ( 0.20,  0.40, "fair agreement"),
    ( 0.40,  0.60, "moderate agreement"),
    ( 0.60,  0.80, "substantial agreement"),
    ( 0.80,  1.00, "almost perfect agreement"),
]


def _interpret_kappa(kappa: float) -> str:
    for low, high, label in _INTERPRETATION_BANDS:
        if low <= kappa <= high:
            return label
    return "out of range"


@dataclass
class KappaResult:
    kappa: float
    interpretation: str
    n_items: int
    n_raters: int
    n_categories: int


# ── Fleiss' Kappa ─────────────────────────────────────────────────────────────

def fleiss_kappa(annotations: list[list[str]], categories: list[str] | None = None) -> KappaResult:
    """
    Fleiss' Kappa for N raters (N can vary per item, though the paper's
    protocol uses a constant N=3).

    Implementation follows Fleiss (1971) directly:

        P_bar      = mean observed agreement per item
        P_bar_e    = expected agreement by chance, from category marginals
        kappa      = (P_bar - P_bar_e) / (1 - P_bar_e)

    Args:
        annotations: list of items, each item a list of category labels
                     (one per annotator). e.g.
                     [["Critical", "Critical", "Alarmist"],   # item 1, 3 annotators
                      ["Supportive", "Supportive", "Supportive"]]  # item 2
        categories:  the full label set (default: inferred from the data,
                     but explicitly passing LABELS from models/baselines.py
                     is recommended so categories never seen in a given
                     sample don't silently change the denominator)

    Returns:
        KappaResult with the kappa value and Landis & Koch interpretation.

    Raises:
        ValueError: if items have inconsistent numbers of raters, or if
                    there are fewer than 2 items or 2 raters.
    """
    if not annotations:
        raise ValueError("Cannot compute Fleiss' Kappa on empty input")

    n_raters_per_item = [len(item) for item in annotations]
    if len(set(n_raters_per_item)) > 1:
        raise ValueError(
            f"Fleiss' Kappa requires the SAME number of raters per item; "
            f"got varying counts: {set(n_raters_per_item)}. "
            "If raters vary per item, use a generalized variant or pad "
            "with adjudicated labels per context.md's annotation protocol."
        )

    n_items  = len(annotations)
    n_raters = n_raters_per_item[0]

    if n_items < 2:
        raise ValueError(f"Need at least 2 items, got {n_items}")
    if n_raters < 2:
        raise ValueError(f"Need at least 2 raters, got {n_raters}")

    if categories is None:
        categories = sorted({label for item in annotations for label in item})
    n_categories = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Build the n_items x n_categories count matrix:
    # n_ij = number of raters who assigned category j to item i
    n_matrix = np.zeros((n_items, n_categories), dtype=int)
    for i, item in enumerate(annotations):
        for label in item:
            if label not in cat_to_idx:
                raise ValueError(
                    f"Label '{label}' not in categories {categories}. "
                    "Pass the full category list explicitly if some "
                    "categories may not appear in every batch."
                )
            n_matrix[i, cat_to_idx[label]] += 1

    # P_i: per-item agreement (proportion of agreeing rater PAIRS)
    # P_i = [ (sum_j n_ij^2) - n_raters ] / [ n_raters * (n_raters - 1) ]
    sum_sq = (n_matrix ** 2).sum(axis=1)
    P_i = (sum_sq - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()

    # p_j: overall proportion of all assignments that went to category j
    p_j = n_matrix.sum(axis=0) / (n_items * n_raters)
    P_bar_e = (p_j ** 2).sum()

    if P_bar_e == 1.0:
        # Degenerate case: every rater assigned every item to the same
        # single category — kappa is mathematically undefined (0/0).
        # By convention, perfect agreement with zero chance-corrected
        # variance is reported as kappa=1.0 if P_bar is also 1.0,
        # else 0.0 (no information to correct for).
        kappa = 1.0 if P_bar == 1.0 else 0.0
    else:
        kappa = (P_bar - P_bar_e) / (1 - P_bar_e)

    return KappaResult(
        kappa=float(kappa),
        interpretation=_interpret_kappa(kappa),
        n_items=n_items,
        n_raters=n_raters,
        n_categories=n_categories,
    )


# ── Cohen's Kappa (2 raters) ─────────────────────────────────────────────────

def cohens_kappa(
    rater_a: list[str],
    rater_b: list[str],
    categories: list[str] | None = None,
) -> KappaResult:
    """
    Cohen's Kappa for exactly 2 raters — used for the lead-annotator-vs-
    majority-vote adjudication check on 3-way disagreement items per
    context.md's annotation protocol, and as a simpler diagnostic when
    only 2 of the 3 annotators are being compared pairwise.

        kappa = (P_o - P_e) / (1 - P_e)

    where P_o is observed agreement and P_e is chance agreement computed
    from each rater's marginal category distribution.

    Args:
        rater_a, rater_b: parallel lists of labels, same length, same
                          item order
        categories:        full label set (default: inferred)

    Returns:
        KappaResult.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(f"Length mismatch: rater_a={len(rater_a)}, rater_b={len(rater_b)}")
    if len(rater_a) < 2:
        raise ValueError(f"Need at least 2 items, got {len(rater_a)}")

    if categories is None:
        categories = sorted(set(rater_a) | set(rater_b))
    n_categories = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    n_items = len(rater_a)

    # Confusion-style matrix between the two raters
    matrix = np.zeros((n_categories, n_categories), dtype=int)
    for a, b in zip(rater_a, rater_b):
        matrix[cat_to_idx[a], cat_to_idx[b]] += 1

    P_o = np.trace(matrix) / n_items

    row_marginals = matrix.sum(axis=1) / n_items
    col_marginals = matrix.sum(axis=0) / n_items
    P_e = (row_marginals * col_marginals).sum()

    if P_e == 1.0:
        kappa = 1.0 if P_o == 1.0 else 0.0
    else:
        kappa = (P_o - P_e) / (1 - P_e)

    return KappaResult(
        kappa=float(kappa),
        interpretation=_interpret_kappa(kappa),
        n_items=n_items,
        n_raters=2,
        n_categories=n_categories,
    )


# ── Majority vote + disagreement detection ───────────────────────────────────

def majority_vote(item_labels: list[str]) -> tuple[str | None, bool]:
    """
    Compute the majority-vote label for one annotated item, and flag
    whether it was a genuine 3-way disagreement (no label has a strict
    majority) — exactly the condition context.md says triggers lead-
    annotator adjudication ("fewer than 4% of the corpus" per the paper).

    Args:
        item_labels: labels from all annotators for one item

    Returns:
        (majority_label, is_three_way_tie)
        majority_label is None if there's a genuine 3-way tie with no
        single most-common label (e.g. 3 annotators, 3 different labels).
    """
    from collections import Counter
    counts = Counter(item_labels)
    most_common = counts.most_common()

    top_count = most_common[0][1]
    tied_at_top = [label for label, c in most_common if c == top_count]

    if len(tied_at_top) > 1:
        # No strict majority — e.g. 3 annotators all disagreed, or a 2-2 tie
        return None, True

    return most_common[0][0], False


def compute_corpus_agreement(
    all_annotations: list[list[str]],
    categories: list[str] | None = None,
) -> dict[str, object]:
    """
    Full corpus-level agreement report, matching the paper's claim:
    "Fleiss' Kappa of κ = 0.71 ... items with three-way disagreement
    (less than 4% of the corpus) were adjudicated by the lead annotator."

    Args:
        all_annotations: one list of rater labels per article in the corpus
        categories:       full label set

    Returns:
        {
            'fleiss_kappa': KappaResult,
            'n_items': int,
            'n_disagreements': int,
            'disagreement_rate': float,   # should be < 0.04 per the paper
            'majority_labels': list[str | None],
        }
    """
    kappa_result = fleiss_kappa(all_annotations, categories=categories)

    majority_labels: list[str | None] = []
    n_disagreements = 0
    for item in all_annotations:
        label, is_disagreement = majority_vote(item)
        majority_labels.append(label)
        if is_disagreement:
            n_disagreements += 1

    disagreement_rate = n_disagreements / len(all_annotations)

    log.info(
        f"[kappa] Fleiss' κ={kappa_result.kappa:.3f} ({kappa_result.interpretation}), "
        f"disagreement rate={disagreement_rate:.1%} "
        f"({n_disagreements}/{len(all_annotations)} items)"
    )

    if disagreement_rate > 0.04:
        log.warning(
            f"[kappa] Disagreement rate {disagreement_rate:.1%} exceeds the "
            f"paper's reported <4% threshold — consider reviewing annotation "
            f"guidelines for ambiguous label boundaries."
        )

    return {
        "fleiss_kappa": kappa_result,
        "n_items": len(all_annotations),
        "n_disagreements": n_disagreements,
        "disagreement_rate": disagreement_rate,
        "majority_labels": majority_labels,
    }
