"""
cleaner.py
──────────
Text cleaning pipeline applied BEFORE NER. This is deliberately
conservative — see context.md's critical rule:

    "Do NOT blindly remove stopwords in news sentiment work.
     Words like 'not', 'no', 'never', 'against', 'despite' matter a lot."

So this module does NOT do stopword removal, lemmatization, or
lowercasing of the body text. It only removes things that are
genuinely noise: HTML tags, boilerplate, excess whitespace, and
typographic variants that would otherwise fragment tokenization.
"""

from __future__ import annotations

import logging
import re
import unicodedata

log = logging.getLogger(__name__)


# ── Boilerplate patterns ──────────────────────────────────────────────────────
# Common navigation / ad / subscription junk that survives newspaper3k's
# extraction on some outlets. These are conservative — better to under-remove
# than to accidentally strip real content.

_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?im)^\s*(advertisement|sponsored content|read more|related articles?)\s*:?\s*$"),
    re.compile(r"(?im)^\s*subscribe (to|now|for).{0,60}$"),
    re.compile(r"(?im)^\s*sign up for our newsletter.{0,80}$"),
    re.compile(r"(?im)^\s*follow us on (twitter|facebook|instagram).{0,60}$"),
    re.compile(r"(?im)^\s*share this (article|story).{0,40}$"),
    re.compile(r"(?im)^\s*\d+ (comments?|shares?)\s*$"),
    re.compile(r"(?im)^\s*copyright © \d{4}.{0,80}$"),
    re.compile(r"(?im)^\s*all rights reserved\.?\s*$"),
    re.compile(r"(?im)^\s*click here to.{0,80}$"),
    re.compile(r"(?im)^\s*this (article|story) (was|first) (originally )?(appeared|published).{0,80}$"),
]

_HTML_TAG_RE     = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE  = re.compile(r"&[a-zA-Z]+;|&#\d+;")
_MULTI_WS_RE     = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_URL_RE          = re.compile(r"https?://\S+")

# Typographic variants → ASCII equivalents.
# We map smart quotes/dashes to ASCII, NOT to remove meaning but to prevent
# tokenizer fragmentation (spaCy handles unicode fine, but downstream
# regex-based feature extractors in Module 3 expect ASCII punctuation).
_TYPOGRAPHIC_MAP: dict[str, str] = {
    "\u2018": "'", "\u2019": "'",      # single smart quotes
    "\u201c": '"', "\u201d": '"',      # double smart quotes
    "\u2013": "-", "\u2014": "-",      # en-dash, em-dash
    "\u2026": "...",                    # ellipsis
    "\u00a0": " ",                      # non-breaking space
    "\u200b": "",                       # zero-width space
    "\ufeff": "",                       # BOM
}


def strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities. Conservative — does not parse DOM."""
    text = _HTML_TAG_RE.sub(" ", text)
    text = _HTML_ENTITY_RE.sub(" ", text)
    return text


def normalize_unicode(text: str) -> str:
    """
    Normalize typographic variants to ASCII equivalents and apply NFKC
    normalization for consistent character representation.

    Deliberately does NOT strip accented characters from proper nouns
    (e.g. "Erdoğan" stays "Erdoğan") — only punctuation-level normalization.
    """
    for variant, ascii_eq in _TYPOGRAPHIC_MAP.items():
        text = text.replace(variant, ascii_eq)
    text = unicodedata.normalize("NFKC", text)
    return text


def remove_boilerplate(text: str) -> str:
    """Strip lines matching known boilerplate patterns (ads, share prompts, etc.)."""
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    return text


def remove_urls(text: str) -> str:
    """Strip raw URLs (these add no semantic content for framing analysis)."""
    return _URL_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs and excessive blank lines."""
    text = _MULTI_WS_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def deduplicate_paragraphs(text: str) -> str:
    """
    Remove exact-duplicate paragraphs that survive from RSS+scrape merging
    (e.g. summary repeated verbatim as the first paragraph of the body).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    seen: set[str] = set()
    unique: list[str] = []
    for p in paragraphs:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return "\n\n".join(unique)


def clean_text(
    text: str,
    strip_boilerplate: bool = True,
    strip_urls: bool = True,
    dedupe_paragraphs: bool = True,
) -> str:
    """
    Full cleaning pipeline, applied in order:
      1. HTML strip
      2. Unicode/typographic normalization
      3. Boilerplate removal
      4. URL removal
      5. Paragraph deduplication
      6. Whitespace collapse

    Args:
        text:               raw article body (post-scrape, pre-NER)
        strip_boilerplate:  toggle boilerplate pattern removal
        strip_urls:         toggle URL stripping
        dedupe_paragraphs:  toggle duplicate paragraph removal

    Returns:
        Cleaned text, ready for spaCy NER pipeline.

    Note:
        Does NOT lowercase, does NOT remove stopwords, does NOT lemmatize.
        Negation words (not/never/without) and modality words
        (allegedly/reportedly) are semantically load-bearing in framing
        analysis and must survive to the NER/attention stage intact.
    """
    if not text:
        return ""

    text = strip_html(text)
    text = normalize_unicode(text)

    if strip_boilerplate:
        text = remove_boilerplate(text)
    if strip_urls:
        text = remove_urls(text)
    if dedupe_paragraphs:
        text = deduplicate_paragraphs(text)

    text = collapse_whitespace(text)
    return text


def clean_title(title: str) -> str:
    """
    Lighter cleaning for headlines — no boilerplate/paragraph logic needed,
    just HTML strip and typographic normalization.
    """
    if not title:
        return ""
    title = strip_html(title)
    title = normalize_unicode(title)
    title = collapse_whitespace(title)
    return title
