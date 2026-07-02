"""
cross_source.py
────────────────
Cross-source framing comparison per context.md Module 5 spec:
    "Cross-source framing comparison; entity framing profiler;
     outlet-by-topic divergence heatmap generator using seaborn and matplotlib."

This reproduces Table IV of the paper (Cross-Source Framing Distribution —
Political Topics) and generalizes it across all topic categories, then
renders the heatmaps that visualize outlet-by-topic framing divergence.

Feeds directly from Module 4's statistical_tests.py output — the heatmap
cells show framing RATES, while statistical significance comes from
Module 4's Mann-Whitney/Kruskal-Wallis results on the same underlying data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from data_collection.schema import Article

log = logging.getLogger(__name__)

LABELS = ["Supportive", "Critical", "Neutral-Reporting", "Alarmist"]
TOPICS = ["politics", "economy", "conflict", "climate", "crime", "elections"]


@dataclass
class FramingDistribution:
    """
    Per-outlet (or per-outlet-per-topic) framing label distribution,
    matching the structure of the paper's Table IV columns.
    """
    group_key:  str                         # e.g. "bbc" or "bbc|politics"
    counts:     dict[str, int] = field(default_factory=dict)   # label -> count
    total:      int = 0
    percentages: dict[str, float] = field(default_factory=dict)  # label -> pct


def compute_framing_distribution(
    articles: list["Article"],
    labels: list[str] = LABELS,
) -> FramingDistribution:
    """
    Compute the framing label distribution for one group of articles
    (typically all articles from one outlet, optionally filtered to one
    topic — see compute_outlet_topic_table for the full cross-tabulation).

    Args:
        articles: list of Article objects, all already labeled
        labels:   the canonical label set (ensures consistent ordering
                  and zero-counts for labels that happen not to appear)

    Returns:
        FramingDistribution with counts and percentages.
    """
    counts = {label: 0 for label in labels}
    for art in articles:
        if art.label in counts:
            counts[art.label] += 1

    total = sum(counts.values())
    percentages = {
        label: (100.0 * count / total if total > 0 else 0.0)
        for label, count in counts.items()
    }

    return FramingDistribution(
        group_key="", counts=counts, total=total, percentages=percentages,
    )


def compute_outlet_distribution_table(
    articles: list["Article"],
    outlets: list[str] | None = None,
    labels: list[str] = LABELS,
    topic_filter: str | None = None,
) -> pd.DataFrame:
    """
    Build the exact table structure of the paper's Table IV:
    rows = outlets, columns = framing labels, cells = percentages.

    Args:
        articles:     all labeled articles (across all outlets)
        outlets:      outlet slugs to include, in row order (default:
                      infer from data, sorted alphabetically)
        labels:       framing label columns, in order
        topic_filter: if given, restrict to articles of this topic only
                      (e.g. "politics" reproduces Table IV exactly)

    Returns:
        DataFrame, index=outlet, columns=labels, values=percentage (0-100).
    """
    if topic_filter:
        articles = [a for a in articles if a.topic == topic_filter]

    if outlets is None:
        outlets = sorted({a.source for a in articles})

    rows: dict[str, dict[str, float]] = {}
    for outlet in outlets:
        outlet_articles = [a for a in articles if a.source == outlet]
        dist = compute_framing_distribution(outlet_articles, labels=labels)
        rows[outlet] = dist.percentages

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(columns=labels)  # enforce consistent column order
    return df


def compute_outlet_topic_cube(
    articles: list["Article"],
    outlets: list[str] | None = None,
    topics: list[str] = TOPICS,
    label: str = "Critical",
) -> pd.DataFrame:
    """
    Build an outlet × topic matrix for ONE framing label's rate —
    this is the data structure the heatmap actually visualizes
    (e.g. "what % of entity mentions are Critical, broken down by
    outlet AND topic simultaneously").

    Args:
        articles: all labeled articles
        outlets:  outlet slugs (rows)
        topics:   topic slugs (columns)
        label:    which framing label's rate to compute per cell

    Returns:
        DataFrame, index=outlet, columns=topic, values=percentage (0-100)
        of articles in that outlet+topic cell carrying the given label.
    """
    if outlets is None:
        outlets = sorted({a.source for a in articles})

    matrix: dict[str, dict[str, float]] = {}
    for outlet in outlets:
        row: dict[str, float] = {}
        for topic in topics:
            subset = [a for a in articles if a.source == outlet and a.topic == topic]
            if not subset:
                row[topic] = np.nan   # no data — render as blank in heatmap, not 0%
                continue
            n_label = sum(1 for a in subset if a.label == label)
            row[topic] = 100.0 * n_label / len(subset)
        matrix[outlet] = row

    df = pd.DataFrame.from_dict(matrix, orient="index")
    df = df.reindex(columns=topics)
    return df


# ── Heatmap rendering ─────────────────────────────────────────────────────────

def plot_outlet_topic_heatmap(
    cube: pd.DataFrame,
    label: str,
    save_path: Path | None = None,
    title: str | None = None,
    cmap: str = "RdYlGn_r",
) -> None:
    """
    Render the outlet × topic heatmap matching context.md's spec:
    "outlet-by-topic divergence heatmap generator using seaborn and matplotlib."

    Args:
        cube:      DataFrame from compute_outlet_topic_cube
        label:     which framing label this heatmap shows (for the title)
        save_path: if given, save PNG to this path instead of/in addition
                   to display
        title:     custom title (default: auto-generated)
        cmap:      colormap — RdYlGn_r means red=high rate, green=low rate,
                   appropriate for "Critical" and "Alarmist" (high = notable);
                   use a different cmap if plotting "Supportive" or
                   "Neutral-Reporting" where high might not mean "concerning"
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        cube,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": f"{label} framing rate (%)"},
        ax=ax,
    )

    ax.set_title(title or f"Outlet × Topic — {label} Framing Rate (%)", fontsize=13)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Outlet")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"[analysis] Saved heatmap → {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_all_label_heatmaps(
    articles: list["Article"],
    output_dir: Path,
    outlets: list[str] | None = None,
    topics: list[str] = TOPICS,
    labels: list[str] = LABELS,
) -> list[Path]:
    """
    Generate one heatmap per framing label — the full Module 5 deliverable
    set (4 heatmaps: Supportive, Critical, Neutral-Reporting, Alarmist,
    each showing outlet × topic rates).

    Args:
        articles:   all labeled articles
        output_dir: directory to save PNGs into
        outlets:    outlet slugs
        topics:     topic slugs
        labels:     framing labels to generate one heatmap per

    Returns:
        List of saved file paths.
    """
    # Labels where a HIGH rate is the "notable" direction (red=high makes sense)
    _alarm_direction = {"Critical", "Alarmist"}

    saved_paths: list[Path] = []
    for label in labels:
        cube = compute_outlet_topic_cube(articles, outlets=outlets, topics=topics, label=label)
        cmap = "RdYlGn_r" if label in _alarm_direction else "YlGnBu"
        path = output_dir / f"heatmap_{label.lower().replace('-', '_')}.png"
        plot_outlet_topic_heatmap(cube, label=label, save_path=path, cmap=cmap)
        saved_paths.append(path)

    return saved_paths


# ── Divergence summary table (Table IV reproduction) ─────────────────────────

def format_table_iv(df: pd.DataFrame) -> str:
    """
    Render the outlet distribution table as readable text, matching the
    layout of the paper's Table IV exactly (outlet rows, label columns,
    percentages to 1 decimal place).

    Args:
        df: DataFrame from compute_outlet_distribution_table

    Returns:
        Formatted string table.
    """
    lines = []
    col_width = 16
    header = f"{'Outlet':<14}" + "".join(f"{c[:14]:>{col_width}}" for c in df.columns)
    lines.append(header)
    lines.append("-" * len(header))
    for outlet, row in df.iterrows():
        line = f"{outlet:<14}" + "".join(f"{v:>{col_width}.1f}" for v in row)
        lines.append(line)
    return "\n".join(lines)


def find_outlet_extremes(df: pd.DataFrame, label: str) -> dict[str, tuple[str, float]]:
    """
    Identify which outlet has the highest/lowest rate for a given label —
    matches the paper's narrative pattern ("RT exhibits the highest
    Supportive framing rate (25.3%)... Fox News shows the highest Critical
    framing rate (41.3%)...").

    Args:
        df:    DataFrame from compute_outlet_distribution_table
        label: which column to find extremes for

    Returns:
        {'highest': (outlet_name, value), 'lowest': (outlet_name, value)}
    """
    if label not in df.columns:
        raise ValueError(f"Label '{label}' not in columns: {list(df.columns)}")

    col = df[label]
    highest_outlet = col.idxmax()
    lowest_outlet  = col.idxmin()

    return {
        "highest": (highest_outlet, float(col[highest_outlet])),
        "lowest":  (lowest_outlet, float(col[lowest_outlet])),
    }


def generate_narrative_summary(df: pd.DataFrame, labels: list[str] = LABELS) -> str:
    """
    Auto-generate the kind of narrative sentences the paper's Section IV-D
    discussion contains, e.g. "RT exhibits the highest Supportive framing
    rate (25.3%)..." — useful as a first draft for the paper's discussion
    section, to be reviewed and refined by a human author, not used verbatim.

    Args:
        df:     DataFrame from compute_outlet_distribution_table
        labels: which labels to generate sentences for

    Returns:
        Multi-line string, one observation per label.
    """
    lines = []
    for label in labels:
        extremes = find_outlet_extremes(df, label)
        high_outlet, high_val = extremes["highest"]
        low_outlet, low_val   = extremes["lowest"]
        lines.append(
            f"{high_outlet.upper()} shows the highest {label} framing rate "
            f"({high_val:.1f}%), while {low_outlet.upper()} shows the lowest "
            f"({low_val:.1f}%)."
        )
    return "\n".join(lines)
