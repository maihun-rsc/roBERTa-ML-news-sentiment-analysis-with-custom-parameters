"""
schema.py
─────────
Article dataclass — the single source of truth for what a record looks like
across every module. If you change a field here, it breaks everywhere
intentionally, so you remember to update the consuming code too.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── Core dataclass ───────────────────────────────────────────────────────────

@dataclass
class Article:
    """
    Canonical record for a single news article.

    Fields populated at collection time are marked [C].
    Fields populated at preprocessing time are marked [P].
    Fields populated at annotation time are marked [A].
    """

    # ── [C] Core fields — must be present after collection ──────────────────
    article_id:  str          # SHA-256(body) first 16 hex chars
    source:      str          # outlet slug: 'bbc', 'cnn', etc.
    title:       str          # headline / article title
    body:        str          # full article text (cleaned at collection time)
    url:         str          # canonical URL
    date:        str          # ISO 8601: "2024-03-15T10:30:00+00:00"
    topic:       str          # inferred from RSS category or keyword match
    region:      str = ""     # source region: 'UK', 'USA', 'India', etc.

    # ── [P] Preprocessing fields — empty string / empty list until Module 2 ─
    entities:    list[str] = field(default_factory=list)
    # Each entity: {"text": "Boris Johnson", "label": "PERSON", "start": 12, "end": 24}
    entity_spans: list[dict[str, Any]] = field(default_factory=list)
    tokens:      list[str] = field(default_factory=list)
    clean_body:  str = ""     # HTML-stripped, unicode-normalised body

    # ── [C/P] Optional fields ────────────────────────────────────────────────
    summary:     str = ""     # RSS summary (shorter than body; kept for reference)
    authors:     list[str] = field(default_factory=list)
    transcript:  str = ""     # ASR transcript if broadcast source
    language:    str = "en"

    # ── [A] Annotation fields — empty until annotation step ─────────────────
    label:       str = ""     # Supportive | Critical | Neutral-Reporting | Alarmist
    confidence:  float = 0.0  # inter-annotator confidence [0, 1]

    # ── Metadata ─────────────────────────────────────────────────────────────
    collection_ts: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    fallback_source: str = ""  # 'mind' | 'ccnews' | 'semeval' | '' (scraped)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Article":
        # Only pass keys that exist in the dataclass to avoid TypeError
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, line: str) -> "Article":
        return cls.from_dict(json.loads(line))

    def is_valid(self) -> tuple[bool, str]:
        """
        Validate required fields.

        Returns:
            (True, '') if valid, (False, reason) if not.
        """
        if not self.title or not self.title.strip():
            return False, "empty title"
        if not self.body or len(self.body.strip()) < 50:
            return False, f"body too short ({len(self.body.strip())} chars)"
        if not self.source:
            return False, "missing source"
        if not self.url:
            return False, "missing url"
        return True, ""


# ── ID generation ────────────────────────────────────────────────────────────

def make_article_id(body: str, url: str = "") -> str:
    """
    Deterministic ID from content hash.
    Two articles with the same body get the same ID → deduplication.

    Args:
        body: article body text
        url:  URL as secondary disambiguation input

    Returns:
        16-character hex string
    """
    content = (body.strip().lower() + url).encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


# ── Topic inference ──────────────────────────────────────────────────────────

TOPIC_KEYWORDS: dict[str, list[str]] = {
    # NOTE: "election" / "vote" deliberately excluded from politics —
    # they belong to the more specific 'elections' category below.
    # Overlap between categories causes incorrect tie-breaking in infer_topic().
    "politics":  ["parliament", "minister", "government", "president",
                  "senate", "policy", "party", "legislation", "congress",
                  "diplomat", "cabinet", "lawmaker"],
    "economy":   ["economy", "gdp", "inflation", "trade", "market", "bank",
                  "stock", "budget", "recession", "growth", "unemployment", "fiscal"],
    "conflict":  ["war", "attack", "military", "strike", "troops", "ceasefire",
                  "battle", "killed", "missile", "army", "bombing", "invasion"],
    "climate":   ["climate", "emissions", "carbon", "temperature", "drought",
                  "flood", "renewable", "cop", "environment", "fossil", "warming"],
    "crime":     ["crime", "murder", "arrest", "police", "court", "verdict",
                  "sentence", "fraud", "corruption", "terrorism", "shooting"],
    "elections": ["election", "ballot", "candidate", "campaign", "poll",
                  "vote", "constituency", "referendum", "polling", "voter"],
}


def infer_topic(text: str, rss_category: str = "") -> str:
    """
    Infer topic from RSS category tag or keyword frequency.

    Tie-breaking: when multiple topics score equally on keyword frequency,
    the topic whose keyword list is matched by the LONGEST keyword wins —
    longer keyword matches are more specific and less likely to be
    coincidental substring hits (e.g. "vote" inside a longer word).

    Args:
        text:         title + body text to search
        rss_category: category string from the RSS entry (may be empty)

    Returns:
        topic slug or 'general' if nothing matches
    """
    # First, try the RSS category tag — exact, deliberate signal from the source
    cat = rss_category.lower().strip()
    if cat:
        # Direct match: category string IS a topic name (e.g. category="Politics")
        for topic in TOPIC_KEYWORDS:
            if topic == cat or topic.rstrip("s") == cat or cat in topic:
                return topic
        # Keyword match: category string contains a topic keyword
        # (e.g. category="World Politics & Government")
        for topic, kws in TOPIC_KEYWORDS.items():
            if any(kw in cat for kw in kws):
                return topic

    # Fall back to keyword frequency in text
    text_lower = text.lower()
    scores: dict[str, int] = {}
    best_match_len: dict[str, int] = {}

    for topic, kws in TOPIC_KEYWORDS.items():
        count = 0
        longest = 0
        for kw in kws:
            hits = text_lower.count(kw)
            if hits:
                count += hits
                longest = max(longest, len(kw))
        scores[topic] = count
        best_match_len[topic] = longest

    top_score = max(scores.values())
    if top_score == 0:
        return "general"

    # Among topics tied at top_score, prefer the one with the longest
    # matched keyword (more specific signal).
    candidates = [t for t, s in scores.items() if s == top_score]
    best = max(candidates, key=lambda t: best_match_len[t])
    return best
