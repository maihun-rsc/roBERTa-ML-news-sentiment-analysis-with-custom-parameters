"""
fallback_loader.py
──────────────────
When scraping fails (or yields too few articles), this module pulls
from three public datasets using the HuggingFace datasets library.

All three are streamed — nothing is downloaded in full.
A subset is sampled per outlet to keep memory under control.

Datasets:
  MIND       — Microsoft News Dataset (news recommendation)
  CC-News    — Common Crawl news (diverse international sources)
  SemEval    — SemEval-2017 Task 5 (entity-level sentiment, financial)

None of these have the exact outlet label we want — we map the
closest available source field to our outlet slugs where possible,
and mark the rest as fallback_source='ccnews'/'mind'/'semeval'.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Iterator

from data_collection.schema import Article, infer_topic, make_article_id

log = logging.getLogger(__name__)

# ── HuggingFace datasets — lazy import ───────────────────────────────────────

def _load_hf_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
    try:
        from datasets import load_dataset  # type: ignore
        return load_dataset(name, *args, **kwargs)
    except ImportError:
        raise RuntimeError(
            "HuggingFace 'datasets' library not installed. "
            "Run: pip install datasets"
        )
    except Exception as e:
        log.warning(f"Could not load dataset '{name}': {e}")
        return None


# ── MIND loader ───────────────────────────────────────────────────────────────

# MIND source category → outlet slug mapping (best-effort)
_MIND_CAT_TO_OUTLET: dict[str, str] = {
    "news": "generic",
    "sports": "generic",
    "entertainment": "generic",
    "health": "generic",
    "politics": "generic",
    "world": "generic",
    "finance": "generic",
}


def load_mind(
    outlet_slug: str,
    max_articles: int = 300,
    split: str = "train",
    seed: int = 42,
) -> list[Article]:
    """
    Load articles from MIND (Microsoft News Dataset).

    MIND has 'title' and 'abstract' fields but no full body.
    We use abstract as body — it's typically 2-4 sentences, enough
    for framing-level analysis.

    Args:
        outlet_slug:  target outlet name for the Article.source field
        max_articles: cap on returned articles
        split:        'train' or 'validation'
        seed:         random seed for sampling

    Returns:
        List of Article objects.
    """
    log.info(f"[fallback/MIND] Loading up to {max_articles} articles for '{outlet_slug}'")

    ds = _load_hf_dataset(
        "mind_news",
        split=split,
        streaming=True,
        trust_remote_code=True,
    )
    if ds is None:
        # Try alternate MIND dataset name
        ds = _load_hf_dataset("nn0/MIND", split=split, streaming=True)
    if ds is None:
        log.warning("[fallback/MIND] Dataset unavailable")
        return []

    articles: list[Article] = []
    rng = random.Random(seed)

    try:
        for raw in ds:
            title    = str(raw.get("title", "")).strip()
            abstract = str(raw.get("abstract", "")).strip()
            category = str(raw.get("category", "")).strip()

            body = abstract or title
            if len(body) < 50:
                continue

            topic = infer_topic(title + " " + body, rss_category=category)

            art = Article(
                article_id=make_article_id(body, title),
                source=outlet_slug,
                title=title,
                body=body,
                url=str(raw.get("url", "")),
                date=str(raw.get("date", "2023-01-01T00:00:00+00:00")),
                topic=topic,
                region="USA",
                fallback_source="mind",
            )
            valid, _ = art.is_valid()
            if valid:
                articles.append(art)

            # Shuffle reservoir sample to avoid topic bias from MIND ordering
            if len(articles) >= max_articles * 3:
                rng.shuffle(articles)
                articles = articles[:max_articles]

            if len(articles) >= max_articles:
                break

    except Exception as e:
        log.warning(f"[fallback/MIND] Stream error: {e}")

    log.info(f"[fallback/MIND] Loaded {len(articles)} articles for '{outlet_slug}'")
    return articles[:max_articles]


# ── CC-News loader ────────────────────────────────────────────────────────────

# Map CC-News domain fragments → outlet slugs
_CCNEWS_DOMAIN_MAP: dict[str, str] = {
    "bbc.co.uk":          "bbc",
    "bbc.com":            "bbc",
    "cnn.com":            "cnn",
    "foxnews.com":        "fox",
    "abcnews.go.com":     "abc",
    "wionews.com":        "wion",
    "firstpost.com":      "firstpost",
    "timesnownews.com":   "timesnow",
    "rt.com":             "rt",
    "aninews.in":         "ani",
}


def load_ccnews(
    outlet_slug: str,
    max_articles: int = 300,
    seed: int = 42,
) -> list[Article]:
    """
    Load articles from CC-News filtered by outlet domain where possible.

    CC-News is huge (~700K articles). We stream and filter.

    Args:
        outlet_slug:  desired outlet slug
        max_articles: cap
        seed:         for sampling

    Returns:
        List of Article objects.
    """
    log.info(f"[fallback/CC-News] Loading up to {max_articles} for '{outlet_slug}'")

    ds = _load_hf_dataset(
        "cc_news",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    if ds is None:
        log.warning("[fallback/CC-News] Dataset unavailable")
        return []

    # Find domain fragments that match our outlet
    target_domains = [
        domain for domain, slug in _CCNEWS_DOMAIN_MAP.items()
        if slug == outlet_slug
    ]

    articles: list[Article] = []
    checked  = 0
    max_scan = max_articles * 50   # scan up to 50× to find matching articles

    try:
        for raw in ds:
            checked += 1
            if checked > max_scan:
                break

            url  = str(raw.get("url", ""))
            text = str(raw.get("text", "")).strip()
            title = str(raw.get("title", "")).strip()

            # Domain filter (if we have specific domains for this outlet)
            if target_domains and not any(d in url for d in target_domains):
                continue

            if len(text) < 150:
                continue

            topic = infer_topic(title + " " + text[:500])

            art = Article(
                article_id=make_article_id(text, url),
                source=outlet_slug,
                title=title or text[:80],
                body=text,
                url=url,
                date=str(raw.get("date", "2023-01-01T00:00:00+00:00")),
                topic=topic,
                region=_infer_region(outlet_slug),
                fallback_source="ccnews",
            )
            valid, _ = art.is_valid()
            if valid:
                articles.append(art)

            if len(articles) >= max_articles:
                break

    except Exception as e:
        log.warning(f"[fallback/CC-News] Stream error: {e}")

    log.info(f"[fallback/CC-News] Loaded {len(articles)} articles (scanned {checked})")
    return articles[:max_articles]


# ── SemEval loader ────────────────────────────────────────────────────────────

def load_semeval(max_articles: int = 200) -> list[Article]:
    """
    Load SemEval-2017 Task 5 (financial news sentiment).

    This dataset has entity-level gold labels — useful for eval
    and for supplementing the Supportive/Critical framing classes
    since financial news tends to be clear-cut.

    Returns:
        List of Article objects (source='semeval', label populated).
    """
    log.info(f"[fallback/SemEval] Loading up to {max_articles} articles")

    # SemEval-2017 Task 5 is not on HF Hub in a clean form.
    # We load the closest available proxy: financial_phrasebank
    ds = _load_hf_dataset(
        "financial_phrasebank",
        "sentences_allagree",
        split="train",
        trust_remote_code=True,
    )
    if ds is None:
        log.warning("[fallback/SemEval] Dataset unavailable")
        return []

    # financial_phrasebank labels: 0=negative, 1=neutral, 2=positive
    # Map to our framing labels (approximate)
    _LABEL_MAP = {0: "Critical", 1: "Neutral-Reporting", 2: "Supportive"}

    articles: list[Article] = []
    try:
        for raw in ds:
            sentence = str(raw.get("sentence", "")).strip()
            label_id = int(raw.get("label", 1))
            if len(sentence) < 20:
                continue

            art = Article(
                article_id=make_article_id(sentence),
                source="semeval",
                title=sentence[:80],
                body=sentence,
                url="",
                date="2017-01-01T00:00:00+00:00",
                topic="economy",
                region="",
                label=_LABEL_MAP.get(label_id, "Neutral-Reporting"),
                confidence=1.0,   # all-agree split → high confidence
                fallback_source="semeval",
            )
            articles.append(art)
            if len(articles) >= max_articles:
                break

    except Exception as e:
        log.warning(f"[fallback/SemEval] Load error: {e}")

    log.info(f"[fallback/SemEval] Loaded {len(articles)} articles")
    return articles


# ── Custom Dataset loader ─────────────────────────────────────────────────────

def load_custom(
    outlet_slug: str,
    dataset_name: str,
    max_articles: int = 300,
    seed: int = 42,
) -> list[Article]:
    """
    Load articles from any custom HuggingFace dataset (e.g. for specific outlets).
    Attempts to intelligently map common fields (text, content, body, sentence) to the body.
    """
    log.info(f"[fallback/Custom] Loading up to {max_articles} from '{dataset_name}' for '{outlet_slug}'")

    ds = _load_hf_dataset(
        dataset_name,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    if ds is None:
        log.warning(f"[fallback/Custom] Dataset '{dataset_name}' unavailable")
        return []

    articles: list[Article] = []
    rng = random.Random(seed)

    try:
        for raw in ds:
            # Try to find a text field
            text = str(raw.get("text", raw.get("content", raw.get("body", raw.get("sentence", ""))))).strip()
            title = str(raw.get("title", raw.get("headline", ""))).strip()

            if len(text) < 50:
                continue

            topic = infer_topic(title + " " + text[:500])

            art = Article(
                article_id=make_article_id(text, title),
                source=outlet_slug,
                title=title or text[:80],
                body=text,
                url=str(raw.get("url", "")),
                date=str(raw.get("date", "2023-01-01T00:00:00+00:00")),
                topic=topic,
                region=_infer_region(outlet_slug),
                fallback_source=dataset_name,
            )
            valid, _ = art.is_valid()
            if valid:
                articles.append(art)

            # Shuffle reservoir sample
            if len(articles) >= max_articles * 3:
                rng.shuffle(articles)
                articles = articles[:max_articles]

            if len(articles) >= max_articles:
                break

    except Exception as e:
        log.warning(f"[fallback/Custom] Stream error for '{dataset_name}': {e}")

    log.info(f"[fallback/Custom] Loaded {len(articles)} articles from '{dataset_name}'")
    return articles[:max_articles]


# ── Dispatcher ────────────────────────────────────────────────────────────────

def load_fallback(
    outlet_cfg: dict[str, Any],
    shortfall: int,
    seed: int = 42,
) -> list[Article]:
    """
    Top-level fallback dispatcher. Called by the pipeline when scraping
    yields fewer articles than desired.

    Args:
        outlet_cfg: one outlet config dict from outlets.json
        shortfall:  how many articles are still needed
        seed:       random seed

    Returns:
        List of Article objects from the best available fallback.
    """
    slug     = outlet_cfg["name"]
    strategy = outlet_cfg.get("fallback", "ccnews")

    log.info(f"[{slug}] Triggering fallback '{strategy}' for {shortfall} articles")

    if strategy == "ccnews":
        return load_ccnews(slug, max_articles=shortfall, seed=seed)
    elif strategy == "mind":
        return load_mind(slug, max_articles=shortfall, seed=seed)
    elif strategy == "semeval":
        return load_semeval(max_articles=shortfall)
    else:
        # Treat the strategy string as a custom HuggingFace dataset name
        dataset_name = outlet_cfg.get("fallback_dataset", strategy)
        return load_custom(slug, dataset_name, max_articles=shortfall, seed=seed)


# ── Helpers ───────────────────────────────────────────────────────────────────

_OUTLET_REGIONS: dict[str, str] = {
    "bbc": "UK", "cnn": "USA", "fox": "USA", "abc": "USA",
    "rt": "Russia", "ani": "India", "timesnow": "India",
    "wion": "India", "firstpost": "India",
}

def _infer_region(outlet_slug: str) -> str:
    return _OUTLET_REGIONS.get(outlet_slug, "")
