"""
statistical_tests.py
─────────────────────
Statistical significance tests per context.md Module 4 spec:
    "scipy.stats — Mann-Whitney U, Kruskal-Wallis H-test for outlet
     divergence significance."
    "statsmodels — ANOVA, post-hoc Tukey HSD."

These reproduce exactly the significance claims in the paper's Section
IV-D (Cross-Source Framing Divergence): the Mann-Whitney U tests between
BBC/Fox News (p<0.001), ANI/RT (p<0.01), and WION/CNN (p<0.05).

Used for outlet-vs-outlet pairwise comparisons (Mann-Whitney U, 2 groups)
and outlet-vs-many comparisons (Kruskal-Wallis / ANOVA, N groups), feeding
directly into Module 5's cross-source heatmaps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

log = logging.getLogger(__name__)


@dataclass
class SignificanceResult:
    """Result of a single statistical test, with interpretation baked in."""
    test_name:    str
    statistic:    float
    p_value:      float
    significant:  bool          # at alpha=0.05 by default
    alpha:        float = 0.05
    interpretation: str = ""
    group_sizes:  list[int] = field(default_factory=list)


def _significance_stars(p: float) -> str:
    """Standard significance notation matching the paper's tables."""
    if p < 0.001:
        return "p < 0.001 (***)"
    elif p < 0.01:
        return "p < 0.01 (**)"
    elif p < 0.05:
        return "p < 0.05 (*)"
    else:
        return f"p = {p:.3f} (n.s.)"


# ── Mann-Whitney U: two-group comparison ─────────────────────────────────────

def mann_whitney_u_test(
    group_a: list[float],
    group_b: list[float],
    label_a: str = "Group A",
    label_b: str = "Group B",
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> SignificanceResult:
    """
    Mann-Whitney U test — non-parametric test for whether two independent
    samples come from the same distribution. Used for outlet-pair framing
    distribution comparisons (e.g. "is BBC's Critical-framing rate
    distribution different from Fox News's?").

    Why non-parametric: framing rates per article cluster are not
    guaranteed to be normally distributed, and sample sizes per outlet
    per topic can be small — Mann-Whitney U doesn't assume normality,
    unlike a t-test.

    Args:
        group_a, group_b: numeric samples (e.g. per-article Critical-
                           framing indicator, or per-cluster framing rate)
        label_a, label_b:  outlet/group names for the result's interpretation
        alpha:             significance threshold
        alternative:       'two-sided' | 'less' | 'greater'

    Returns:
        SignificanceResult with U statistic, p-value, and interpretation.
    """
    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError(
            f"Mann-Whitney U requires ≥2 samples per group; "
            f"got {len(group_a)} and {len(group_b)}"
        )

    statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative=alternative)

    significant = p_value < alpha
    direction = "differs significantly from" if significant else "does not differ significantly from"
    interpretation = (
        f"{label_a} (n={len(group_a)}) {direction} {label_b} (n={len(group_b)}) "
        f"[{_significance_stars(p_value)}]"
    )

    return SignificanceResult(
        test_name="Mann-Whitney U",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=significant,
        alpha=alpha,
        interpretation=interpretation,
        group_sizes=[len(group_a), len(group_b)],
    )


# ── Kruskal-Wallis: N-group comparison ───────────────────────────────────────

def kruskal_wallis_test(
    groups: dict[str, list[float]],
    alpha: float = 0.05,
) -> SignificanceResult:
    """
    Kruskal-Wallis H-test — non-parametric one-way ANOVA. Tests whether
    AT LEAST ONE of N groups differs from the others. Used as the
    omnibus test across all 9 outlets before drilling into pairwise
    Mann-Whitney comparisons (standard practice: omnibus test first,
    pairwise post-hoc only if omnibus is significant, to control
    family-wise error rate).

    Args:
        groups: {outlet_name: [framing_rate_per_cluster, ...]}
        alpha:  significance threshold

    Returns:
        SignificanceResult — if significant, proceed to pairwise tests;
        if not, the apparent outlet differences in Table IV are within
        the range expected by chance.
    """
    if len(groups) < 3:
        raise ValueError(
            f"Kruskal-Wallis is for 3+ groups; got {len(groups)}. "
            "Use mann_whitney_u_test for 2-group comparisons."
        )

    samples = list(groups.values())
    for name, sample in groups.items():
        if len(sample) < 2:
            raise ValueError(f"Group '{name}' has only {len(sample)} sample(s); need ≥2")

    statistic, p_value = stats.kruskal(*samples)
    significant = p_value < alpha

    interpretation = (
        f"Across {len(groups)} groups ({', '.join(groups.keys())}), "
        f"{'at least one group differs significantly' if significant else 'no significant difference detected'} "
        f"[{_significance_stars(p_value)}]"
    )

    return SignificanceResult(
        test_name="Kruskal-Wallis H",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=significant,
        alpha=alpha,
        interpretation=interpretation,
        group_sizes=[len(s) for s in samples],
    )


# ── One-way ANOVA + Tukey HSD post-hoc ───────────────────────────────────────

def one_way_anova(
    groups: dict[str, list[float]],
    alpha: float = 0.05,
) -> SignificanceResult:
    """
    Parametric one-way ANOVA — alternative omnibus test to Kruskal-Wallis,
    used when framing-rate distributions are closer to normal (e.g. when
    aggregated at the topic-cluster level rather than per-article, where
    the central limit theorem applies more comfortably).

    Args:
        groups: {outlet_name: [framing_rate_per_cluster, ...]}
        alpha:  significance threshold

    Returns:
        SignificanceResult with the F-statistic.
    """
    if len(groups) < 3:
        raise ValueError(f"ANOVA is for 3+ groups; got {len(groups)}")

    samples = list(groups.values())
    statistic, p_value = stats.f_oneway(*samples)
    significant = p_value < alpha

    interpretation = (
        f"One-way ANOVA across {len(groups)} groups: "
        f"{'significant between-group variance' if significant else 'no significant between-group variance'} "
        f"[{_significance_stars(p_value)}]"
    )

    return SignificanceResult(
        test_name="One-way ANOVA",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=significant,
        alpha=alpha,
        interpretation=interpretation,
        group_sizes=[len(s) for s in samples],
    )


@dataclass
class TukeyPairResult:
    """One pairwise comparison from a Tukey HSD post-hoc test."""
    group_a: str
    group_b: str
    mean_diff: float
    p_adj: float
    significant: bool
    ci_lower: float
    ci_upper: float


def tukey_hsd_posthoc(
    groups: dict[str, list[float]],
    alpha: float = 0.05,
) -> list[TukeyPairResult]:
    """
    Tukey's Honestly Significant Difference post-hoc test — run AFTER a
    significant one-way ANOVA to identify WHICH specific pairs of outlets
    differ, while controlling the family-wise error rate across all
    pairwise comparisons (unlike running many independent Mann-Whitney
    tests, which inflates false-positive rate).

    Args:
        groups: {outlet_name: [framing_rate_per_cluster, ...]}
        alpha:  family-wise significance threshold

    Returns:
        List of TukeyPairResult, one per pairwise comparison.
    """
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    all_values: list[float] = []
    all_labels: list[str]   = []
    for name, values in groups.items():
        all_values.extend(values)
        all_labels.extend([name] * len(values))

    result = pairwise_tukeyhsd(
        endog=np.array(all_values),
        groups=np.array(all_labels),
        alpha=alpha,
    )

    pairs: list[TukeyPairResult] = []
    for row in result.summary().data[1:]:  # skip header row
        group1, group2, meandiff, p_adj, lower, upper, reject = row
        pairs.append(TukeyPairResult(
            group_a=str(group1), group_b=str(group2),
            mean_diff=float(meandiff), p_adj=float(p_adj),
            significant=bool(reject),
            ci_lower=float(lower), ci_upper=float(upper),
        ))

    return pairs


# ── Convenience: full outlet divergence pipeline ─────────────────────────────

def test_outlet_divergence(
    outlet_framing_rates: dict[str, list[float]],
    alpha: float = 0.05,
    run_posthoc: bool = True,
) -> dict[str, object]:
    """
    End-to-end pipeline matching the paper's Section IV-D methodology:
      1. Kruskal-Wallis omnibus test across all outlets
      2. If significant, pairwise Mann-Whitney U for every outlet pair
      3. (Optional) ANOVA + Tukey HSD as the parametric cross-check

    Args:
        outlet_framing_rates: {outlet_name: [per-cluster framing rate, ...]}
                              e.g. the Critical-framing percentage computed
                              per event-cluster, per outlet
        alpha:                significance threshold
        run_posthoc:          whether to compute Tukey HSD pairs too

    Returns:
        {
            'omnibus_kruskal': SignificanceResult,
            'omnibus_anova':   SignificanceResult,
            'pairwise':        {(outlet_a, outlet_b): SignificanceResult, ...},
            'tukey':           [TukeyPairResult, ...] or None,
        }
    """
    omnibus_kw = kruskal_wallis_test(outlet_framing_rates, alpha=alpha)
    omnibus_anova = one_way_anova(outlet_framing_rates, alpha=alpha)

    log.info(f"[stats] Omnibus Kruskal-Wallis: {omnibus_kw.interpretation}")
    log.info(f"[stats] Omnibus ANOVA: {omnibus_anova.interpretation}")

    pairwise: dict[tuple[str, str], SignificanceResult] = {}

    if omnibus_kw.significant:
        outlets = list(outlet_framing_rates.keys())
        for i in range(len(outlets)):
            for j in range(i + 1, len(outlets)):
                a, b = outlets[i], outlets[j]
                try:
                    result = mann_whitney_u_test(
                        outlet_framing_rates[a], outlet_framing_rates[b],
                        label_a=a, label_b=b, alpha=alpha,
                    )
                    pairwise[(a, b)] = result
                    if result.significant:
                        log.info(f"[stats] {a} vs {b}: {result.interpretation}")
                except ValueError as e:
                    log.warning(f"[stats] Skipping {a} vs {b}: {e}")
    else:
        log.info("[stats] Omnibus test not significant — skipping pairwise comparisons "
                 "(per standard practice, controls false-positive rate)")

    tukey_pairs = None
    if run_posthoc and omnibus_anova.significant:
        try:
            tukey_pairs = tukey_hsd_posthoc(outlet_framing_rates, alpha=alpha)
        except Exception as e:
            log.warning(f"[stats] Tukey HSD failed: {e}")

    return {
        "omnibus_kruskal": omnibus_kw,
        "omnibus_anova": omnibus_anova,
        "pairwise": pairwise,
        "tukey": tukey_pairs,
    }
