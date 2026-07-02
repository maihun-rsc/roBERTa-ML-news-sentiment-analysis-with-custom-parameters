"""
analysis
────────
Module 5 — Analysis

Public API:
    from analysis import compute_outlet_distribution_table, plot_all_label_heatmaps
    from analysis import build_entity_profiles, find_cross_outlet_divergent_entities
    from analysis import run_analysis
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from analysis.cross_source import (
    FramingDistribution,
    compute_framing_distribution,
    compute_outlet_distribution_table,
    compute_outlet_topic_cube,
    find_outlet_extremes,
    format_table_iv,
    generate_narrative_summary,
    plot_all_label_heatmaps,
    plot_outlet_topic_heatmap,
)
from analysis.entity_profiler import (
    EntityProfile,
    build_entity_profiles,
    find_cross_outlet_divergent_entities,
    format_entity_profile,
    get_top_entities_by_label,
    profiles_to_dataframe,
)

if TYPE_CHECKING:
    from configs.env_config import Config

log = logging.getLogger(__name__)

__all__ = [
    # cross_source
    "FramingDistribution", "compute_framing_distribution",
    "compute_outlet_distribution_table", "compute_outlet_topic_cube",
    "plot_outlet_topic_heatmap", "plot_all_label_heatmaps",
    "format_table_iv", "find_outlet_extremes", "generate_narrative_summary",
    # entity_profiler
    "EntityProfile", "build_entity_profiles", "profiles_to_dataframe",
    "find_cross_outlet_divergent_entities", "get_top_entities_by_label",
    "format_entity_profile",
    # orchestrator
    "run_analysis",
]


def run_analysis(cfg: "Config") -> dict:
    """
    Top-level entry point called by main.py's stage_analyse().

    Loads labeled articles, produces:
      1. The Table IV-style outlet × label distribution table
      2. Four outlet × topic heatmaps (one per framing label)
      3. Entity framing profiles + cross-outlet divergence report
      4. A narrative summary draft for the paper's discussion section

    Args:
        cfg: Config from get_config()

    Returns:
        Dict summarizing what was generated and where it was saved.
    """
    from data_collection.writer import read_jsonl

    processed_path = cfg.data_processed / "processed_articles.jsonl"
    if not processed_path.exists():
        log.error(f"[analyse] No processed articles at {processed_path}")
        return {"status": "error", "reason": "no_processed_data"}

    articles = read_jsonl(processed_path)
    labeled  = [a for a in articles if a.label]

    if len(labeled) < 10:
        log.warning(f"[analyse] Only {len(labeled)} labeled articles — insufficient for analysis")
        return {"status": "skipped", "reason": "insufficient_labeled_data"}

    log.info(f"[analyse] Running cross-source + entity analysis on {len(labeled)} labeled articles")

    output_dir = cfg.data_processed.parent / "analysis_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Table IV reproduction ────────────────────────────────────────────────
    table_iv = compute_outlet_distribution_table(labeled, topic_filter="politics")
    log.info("\n[analyse] Table IV — Cross-Source Framing Distribution (Politics)\n" +
             format_table_iv(table_iv))

    narrative = generate_narrative_summary(table_iv)
    log.info(f"\n[analyse] Draft narrative summary:\n{narrative}")

    table_iv_path = output_dir / "table_iv_distribution.csv"
    table_iv.to_csv(table_iv_path)

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    heatmap_paths = plot_all_label_heatmaps(labeled, output_dir=output_dir)

    # ── Entity profiles ───────────────────────────────────────────────────────
    profiles = build_entity_profiles(labeled, min_mentions=3)
    profiles_df = profiles_to_dataframe(profiles)
    profiles_path = output_dir / "entity_profiles.csv"
    profiles_df.to_csv(profiles_path, index=False)

    divergent = find_cross_outlet_divergent_entities(profiles, label="Critical")
    log.info(f"[analyse] Found {len(divergent)} entities with significant cross-outlet "
             f"divergence in Critical framing")

    return {
        "status": "complete",
        "n_articles": len(labeled),
        "n_entities_profiled": len(profiles),
        "n_divergent_entities": len(divergent),
        "table_iv_path": str(table_iv_path),
        "heatmap_paths": [str(p) for p in heatmap_paths],
        "entity_profiles_path": str(profiles_path),
        "narrative_draft": narrative,
    }
