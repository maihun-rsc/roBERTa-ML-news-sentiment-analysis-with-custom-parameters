"""
__init__.py (preprocessing module)
────────────────────────────────────
Public API + the run_preprocessing() orchestrator that main.py calls
via stage_preprocess().

Pipeline order per article:
    1. clean_text()           — strip HTML, normalize unicode, remove boilerplate
    2. process_document()     — spaCy NER + POS + dependency parse
    3. get_primary_entities() — rank entities, pick framing targets
    4. compute_proximity_scores() — per-entity attention weighting input

Output: Article objects with .clean_body, .entities, .entity_spans,
.tokens populated — ready for Module 3's RoBERTa fine-tuner.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from preprocessing.asr_cleaner import (
    CleanedTranscript,
    TranscriptSegment,
    clean_transcript,
    whisper_result_to_segments,
)
from preprocessing.cleaner import clean_text, clean_title
from preprocessing.ner_pipeline import (
    EntitySpan,
    ProcessedDoc,
    batch_process,
    get_primary_entities,
    load_ner_model,
    process_document,
)
from preprocessing.proximity_scorer import (
    compute_all_entity_proximities,
    compute_proximity_scores,
    get_entity_context_window,
)

if TYPE_CHECKING:
    from configs.env_config import Config
    from data_collection.schema import Article

log = logging.getLogger(__name__)

__all__ = [
    "clean_text", "clean_title",
    "EntitySpan", "ProcessedDoc", "load_ner_model", "process_document",
    "batch_process", "get_primary_entities",
    "compute_proximity_scores", "compute_all_entity_proximities",
    "get_entity_context_window",
    "TranscriptSegment", "CleanedTranscript", "clean_transcript",
    "whisper_result_to_segments",
    "run_preprocessing", "preprocess_article",
]


def preprocess_article(nlp, article: "Article") -> "Article":
    """
    Run the full preprocessing pipeline on a single Article in-place
    (mutates and returns the same object for convenient chaining).

    Args:
        nlp:     loaded spaCy model (from load_ner_model)
        article: Article with .body populated from Module 1

    Returns:
        The same Article, now with .clean_body, .entities, .entity_spans,
        .tokens populated.
    """
    article.title      = clean_title(article.title)
    article.clean_body = clean_text(article.body)

    if not article.clean_body:
        log.debug(f"[preprocess] Empty clean_body for {article.article_id} — skipping NER")
        return article

    processed = process_document(nlp, article.clean_body)

    article.tokens = processed.tokens
    article.entities = get_primary_entities(processed, top_k=10)
    article.entity_spans = [
        {
            "text": e.text, "label": e.label,
            "start": e.start, "end": e.end,
            "sent_idx": e.sent_idx, "token_idx": e.token_idx,
        }
        for e in processed.entities
    ]

    return article


def run_preprocessing(
    articles: list["Article"],
    cfg: "Config",
    batch_size: int | None = None,
) -> list["Article"]:
    """
    Top-level entry point called by main.py's stage_preprocess().

    Loads the NER model once, processes all articles via spaCy's
    nlp.pipe() batching (much faster than per-article calls), and
    writes the result to cfg.data_processed.

    Args:
        articles:   list of Article objects from Module 1 (data_collection)
        cfg:        Config from get_config()
        batch_size: override spaCy batch size (defaults to cfg.batch_size)

    Returns:
        List of processed Article objects.
    """
    from data_collection.writer import ArticleWriter

    if not articles:
        log.warning("[preprocess] No articles to process")
        return []

    prefer_trf = cfg.device in ("cuda", "mps")
    nlp, model_name = load_ner_model(prefer_transformer=prefer_trf, device=cfg.device)
    log.info(f"[preprocess] Using model: {model_name} (device={cfg.device})")

    effective_batch = batch_size or cfg.batch_size
    bodies = [clean_text(a.body) for a in articles]
    titles = [clean_title(a.title) for a in articles]

    log.info(f"[preprocess] Processing {len(articles)} articles (batch_size={effective_batch}) …")
    processed_docs = batch_process(nlp, bodies, batch_size=effective_batch)

    for article, title, clean_body, doc in zip(articles, titles, bodies, processed_docs):
        article.title = title
        article.clean_body = clean_body
        article.tokens = doc.tokens
        article.entities = get_primary_entities(doc, top_k=10)
        article.entity_spans = [
            {
                "text": e.text, "label": e.label,
                "start": e.start, "end": e.end,
                "sent_idx": e.sent_idx, "token_idx": e.token_idx,
            }
            for e in doc.entities
        ]

    out_path = cfg.data_processed / "processed_articles.jsonl"
    with ArticleWriter(out_path) as w:
        w.write_many(articles)

    n_with_entities = sum(1 for a in articles if a.entities)
    log.info(f"[preprocess] ✓ Complete. {len(articles)} articles processed, "
             f"{n_with_entities} with ≥1 entity → {out_path}")

    return articles
