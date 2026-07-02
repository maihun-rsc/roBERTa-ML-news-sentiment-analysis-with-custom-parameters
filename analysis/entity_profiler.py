"""
entity_profiler.py
───────────────────
Entity framing profiler per context.md Module 5 spec:
    "entity framing profiler"

Where cross_source.py asks "how does outlet X frame topic Y?", this
module asks "how is entity E framed, and does that framing differ across
the outlets that mention E?" — the per-entity equivalent of Table IV.

This directly operationalizes the entity-centric design philosophy of
the whole project: the labels are about ENTITIES, not documents, so the
analysis layer needs an entity-first view, not just an outlet-first one.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from data_collection.schema import Article

log = logging.getLogger(__name__)

LABELS = ["Supportive", "Critical", "Neutral-Reporting", "Alarmist"]


@dataclass
class EntityProfile:
    """
    Full framing profile for one named entity across the corpus.

    Note on data model: the current Article schema (Module 1/2) stores
    `label` at the DOCUMENT level and `entities`/`entity_spans` as the
    list of entities mentioned, without a per-entity label mapping. This
    profiler therefore treats "this entity was mentioned in an article
    labeled X" as a proxy for "this entity was framed X in that article"
    — exact when the entity is the article's sole/primary subject (the
    common case for the top-ranked entity per get_primary_entities),
    approximate otherwise. A future schema revision could add an explicit
    {entity: label} dict per article if multi-entity per-entity labels
    are annotated directly.
    """
    entity_text: str
    total_mentions: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    label_percentages: dict[str, float] = field(default_factory=dict)
    outlets_mentioning: set[str] = field(default_factory=set)
    per_outlet_label_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    dominant_label: str = ""


def build_entity_profiles(
    articles: list["Article"],
    min_mentions: int = 3,
    labels: list[str] = LABELS,
) -> dict[str, EntityProfile]:
    """
    Build a framing profile for every entity mentioned at least
    `min_mentions` times across the (labeled) corpus.

    Args:
        articles:     labeled Article objects with .entities populated
                      (from Module 2's preprocessing)
        min_mentions: minimum mention count to include an entity (avoids
                      noisy single-mention profiles dominating the output)
        labels:       canonical label set for consistent percentage columns

    Returns:
        {entity_text: EntityProfile}, only entities meeting min_mentions.
    """
    raw_counts: dict[str, dict[str, int]] = defaultdict(lambda: {l: 0 for l in labels})
    outlet_sets: dict[str, set[str]] = defaultdict(set)
    per_outlet: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {l: 0 for l in labels})
    )

    for art in articles:
        if not art.label or art.label not in labels:
            continue
        for entity_text in art.entities:
            raw_counts[entity_text][art.label] += 1
            outlet_sets[entity_text].add(art.source)
            per_outlet[entity_text][art.source][art.label] += 1

    profiles: dict[str, EntityProfile] = {}
    for entity_text, counts in raw_counts.items():
        total = sum(counts.values())
        if total < min_mentions:
            continue

        percentages = {l: (100.0 * c / total if total > 0 else 0.0) for l, c in counts.items()}
        dominant = max(counts, key=lambda l: counts[l])

        profiles[entity_text] = EntityProfile(
            entity_text=entity_text,
            total_mentions=total,
            label_counts=dict(counts),
            label_percentages=percentages,
            outlets_mentioning=outlet_sets[entity_text],
            per_outlet_label_counts={o: dict(c) for o, c in per_outlet[entity_text].items()},
            dominant_label=dominant,
        )

    log.info(
        f"[analysis] Built {len(profiles)} entity profiles "
        f"(min_mentions={min_mentions}, from {len(raw_counts)} total entities seen)"
    )
    return profiles


def profiles_to_dataframe(profiles: dict[str, EntityProfile], labels: list[str] = LABELS) -> pd.DataFrame:
    """
    Convert profile dict into a flat DataFrame for export / further analysis
    — one row per entity, columns for mention count, per-label percentages,
    dominant label, and outlet count.

    Args:
        profiles: output of build_entity_profiles
        labels:   column order for percentage fields

    Returns:
        DataFrame sorted by total_mentions descending.
    """
    rows = []
    for entity_text, profile in profiles.items():
        row = {
            "entity": entity_text,
            "total_mentions": profile.total_mentions,
            "n_outlets": len(profile.outlets_mentioning),
            "dominant_label": profile.dominant_label,
        }
        for label in labels:
            row[f"pct_{label}"] = profile.label_percentages.get(label, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("total_mentions", ascending=False).reset_index(drop=True)
    return df


def find_cross_outlet_divergent_entities(
    profiles: dict[str, EntityProfile],
    min_outlets: int = 2,
    min_divergence_pct: float = 20.0,
    label: str = "Critical",
) -> list[tuple[str, dict[str, float]]]:
    """
    Find entities whose framing for a given label varies substantially
    across the outlets that mention them — these are the most interesting
    cases for the paper's discussion ("Entity X is framed Critically by
    Outlet A in 60% of mentions but only 10% by Outlet B").

    Args:
        profiles:            output of build_entity_profiles
        min_outlets:         entity must be mentioned by at least this
                              many DIFFERENT outlets to be eligible
                              (single-outlet entities can't show divergence)
        min_divergence_pct:  minimum spread (max - min) in per-outlet
                              label percentage to be flagged
        label:                which framing label to check divergence for

    Returns:
        List of (entity_text, {outlet: label_pct}) tuples, sorted by
        divergence magnitude descending.
    """
    results: list[tuple[str, dict[str, float], float]] = []

    for entity_text, profile in profiles.items():
        if len(profile.outlets_mentioning) < min_outlets:
            continue

        outlet_pcts: dict[str, float] = {}
        for outlet, label_counts in profile.per_outlet_label_counts.items():
            outlet_total = sum(label_counts.values())
            if outlet_total == 0:
                continue
            outlet_pcts[outlet] = 100.0 * label_counts.get(label, 0) / outlet_total

        if len(outlet_pcts) < min_outlets:
            continue

        divergence = max(outlet_pcts.values()) - min(outlet_pcts.values())
        if divergence >= min_divergence_pct:
            results.append((entity_text, outlet_pcts, divergence))

    results.sort(key=lambda x: -x[2])
    return [(text, pcts) for text, pcts, _div in results]


def get_top_entities_by_label(
    profiles: dict[str, EntityProfile],
    label: str,
    top_k: int = 10,
    min_mentions: int = 3,
) -> list[tuple[str, float, int]]:
    """
    Rank entities by their percentage-framed-as-{label}, useful for
    "which entities are most consistently framed Critically across the
    whole corpus?" type queries.

    Args:
        profiles:     output of build_entity_profiles
        label:        which framing label to rank by
        top_k:        how many entities to return
        min_mentions: re-filter (in case caller wants a stricter threshold
                      than was used when building the profiles)

    Returns:
        List of (entity_text, label_percentage, total_mentions), sorted
        descending by percentage, ties broken by mention count.
    """
    candidates = [
        (text, p.label_percentages.get(label, 0.0), p.total_mentions)
        for text, p in profiles.items()
        if p.total_mentions >= min_mentions
    ]
    candidates.sort(key=lambda x: (-x[1], -x[2]))
    return candidates[:top_k]


def format_entity_profile(profile: EntityProfile, labels: list[str] = LABELS) -> str:
    """Render one entity's profile as readable text."""
    lines = [
        f"Entity: {profile.entity_text}",
        f"  Total mentions: {profile.total_mentions}",
        f"  Outlets: {', '.join(sorted(profile.outlets_mentioning))}",
        f"  Dominant framing: {profile.dominant_label}",
        "  Distribution:",
    ]
    for label in labels:
        pct = profile.label_percentages.get(label, 0.0)
        count = profile.label_counts.get(label, 0)
        lines.append(f"    {label:<20} {pct:>6.1f}%  (n={count})")
    return "\n".join(lines)
