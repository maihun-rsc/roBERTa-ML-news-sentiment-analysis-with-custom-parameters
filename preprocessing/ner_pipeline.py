"""
ner_pipeline.py
───────────────
Named Entity Recognition + POS tagging + dependency parsing, all in
one spaCy pass (spaCy pipes these together — running them separately
would re-tokenize and waste compute).

Model selection:
    en_core_web_trf  — transformer-backed, 89.9 F1 on OntoNotes 5.0.
                        Used in production (Antigravity GPU / Kaggle GPU).
    en_core_web_sm   — CPU-fast fallback for development / low-resource runs.
                        Lower NER accuracy but adequate for pipeline testing.

The model is selected automatically based on cfg.device — GPU available
means we load the transformer model and move it to GPU; CPU-only falls
back to the small model. This mirrors the same dual-environment pattern
as configs/env_config.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Entity types we keep — others (CARDINAL, ORDINAL, PERCENT, MONEY, DATE,
# TIME, QUANTITY, LANGUAGE) are dropped at this stage. They're not framing
# targets; a framing label attaches to PERSON/ORG/GPE/EVENT/NORP entities.
KEPT_ENTITY_TYPES: frozenset[str] = frozenset({
    "PERSON", "ORG", "GPE", "EVENT", "NORP", "FAC", "LOC",
})

_MODEL_TRF = "en_core_web_trf"
_MODEL_SM  = "en_core_web_sm"


@dataclass
class EntitySpan:
    """One named entity mention within a document."""
    text:  str
    label: str          # PERSON | ORG | GPE | EVENT | NORP | FAC | LOC
    start: int           # character offset, start (inclusive)
    end:   int           # character offset, end (exclusive)
    sent_idx: int = 0    # which sentence (0-indexed) this entity falls in
    token_idx: int = 0   # token index of the entity's head token


@dataclass
class ProcessedDoc:
    """
    Full output of the NER pipeline for one article body.

    This is what gets attached back onto the Article dataclass
    (entities, entity_spans, tokens fields) before Module 3 consumes it.
    """
    tokens:        list[str]               = field(default_factory=list)
    pos_tags:      list[str]               = field(default_factory=list)
    dep_labels:    list[str]               = field(default_factory=list)
    head_indices:  list[int]               = field(default_factory=list)
    sent_boundaries: list[tuple[int, int]] = field(default_factory=list)  # (start_tok, end_tok) per sentence
    entities:      list[EntitySpan]        = field(default_factory=list)
    entity_freq:   dict[str, int]          = field(default_factory=dict)  # entity text -> mention count


# ── Model loading (lazy, cached) ─────────────────────────────────────────────

_loaded_models: dict[str, Any] = {}


def load_ner_model(prefer_transformer: bool = True, device: str = "cpu") -> tuple[Any, str]:
    """
    Load the appropriate spaCy model, falling back gracefully if the
    transformer model isn't installed.

    Args:
        prefer_transformer: try en_core_web_trf first if True
        device: 'cuda' | 'mps' | 'cpu' — only affects trf model GPU placement

    Returns:
        (nlp, model_name) — model_name is whichever model actually loaded.

    Raises:
        RuntimeError: if neither model is installed. The error message
                      includes the exact pip/spacy download commands needed.
    """
    import spacy

    candidates = [_MODEL_TRF, _MODEL_SM] if prefer_transformer else [_MODEL_SM, _MODEL_TRF]

    for model_name in candidates:
        if model_name in _loaded_models:
            return _loaded_models[model_name], model_name

        try:
            nlp = spacy.load(model_name)
            if model_name == _MODEL_TRF and device == "cuda":
                try:
                    spacy.require_gpu()
                    log.info(f"[ner] {model_name} loaded on GPU")
                except Exception:
                    log.warning(f"[ner] GPU requested but unavailable for {model_name} — using CPU")
            _loaded_models[model_name] = nlp
            log.info(f"[ner] Loaded model: {model_name}")
            return nlp, model_name
        except OSError:
            log.warning(f"[ner] Model '{model_name}' not found, trying next candidate")
            continue

    raise RuntimeError(
        "No spaCy English model installed. Run ONE of:\n"
        f"  python -m spacy download {_MODEL_TRF}   # production, GPU recommended\n"
        f"  python -m spacy download {_MODEL_SM}    # development, CPU-friendly\n"
    )


# ── Core processing ───────────────────────────────────────────────────────────

def process_document(nlp: Any, text: str) -> ProcessedDoc:
    """
    Run the full spaCy pipeline (tokenize, POS, dep parse, NER) on one
    document and extract everything Module 3's entity-aware attention
    and proximity scorer need.

    Args:
        nlp:  loaded spaCy Language object (from load_ner_model)
        text: cleaned article body (output of cleaner.clean_text)

    Returns:
        ProcessedDoc with tokens, tags, entities, and sentence boundaries.
    """
    if not text or not text.strip():
        return ProcessedDoc()

    doc = nlp(text)

    tokens, pos_tags, dep_labels, head_indices = [], [], [], []
    for tok in doc:
        tokens.append(tok.text)
        pos_tags.append(tok.pos_)
        dep_labels.append(tok.dep_)
        head_indices.append(tok.head.i)

    # Sentence boundaries as (start_token_idx, end_token_idx) pairs
    sent_boundaries: list[tuple[int, int]] = []
    for sent in doc.sents:
        sent_boundaries.append((sent.start, sent.end))

    # Map each token index to its sentence index for entity tagging
    tok_to_sent: dict[int, int] = {}
    for sent_idx, (start, end) in enumerate(sent_boundaries):
        for tok_i in range(start, end):
            tok_to_sent[tok_i] = sent_idx

    entities: list[EntitySpan] = []
    entity_freq: dict[str, int] = {}

    for ent in doc.ents:
        if ent.label_ not in KEPT_ENTITY_TYPES:
            continue

        sent_idx = tok_to_sent.get(ent.start, 0)
        # The entity's syntactic head token (for proximity scoring later)
        head_tok_idx = ent.root.i

        entities.append(EntitySpan(
            text=ent.text,
            label=ent.label_,
            start=ent.start_char,
            end=ent.end_char,
            sent_idx=sent_idx,
            token_idx=head_tok_idx,
        ))

        # Normalize entity text for frequency counting (case-insensitive,
        # but keep original casing in the EntitySpan for display)
        key = ent.text.strip()
        entity_freq[key] = entity_freq.get(key, 0) + 1

    return ProcessedDoc(
        tokens=tokens,
        pos_tags=pos_tags,
        dep_labels=dep_labels,
        head_indices=head_indices,
        sent_boundaries=sent_boundaries,
        entities=entities,
        entity_freq=entity_freq,
    )


def get_primary_entities(processed: ProcessedDoc, top_k: int = 5) -> list[str]:
    """
    Rank entities by mention frequency and return the top-k — these are
    the candidate "target entities" for entity-centric framing labels.

    Args:
        processed: output of process_document
        top_k:     how many top entities to return

    Returns:
        List of entity text strings, sorted by frequency descending.
    """
    if not processed.entity_freq:
        return []
    ranked = sorted(processed.entity_freq.items(), key=lambda kv: kv[1], reverse=True)
    return [text for text, _count in ranked[:top_k]]


def batch_process(
    nlp: Any,
    texts: list[str],
    batch_size: int = 32,
    n_process: int = 1,
) -> list[ProcessedDoc]:
    """
    Process multiple documents using spaCy's nlp.pipe() for efficiency —
    significantly faster than calling process_document() in a loop because
    spaCy batches the transformer forward passes internally.

    Args:
        nlp:        loaded spaCy Language object
        texts:      list of cleaned article bodies
        batch_size: spaCy internal batch size
        n_process:  number of worker processes (CPU only; ignored for GPU/trf)

    Returns:
        List of ProcessedDoc, same order and length as input texts.
    """
    results: list[ProcessedDoc] = []

    # n_process > 1 is incompatible with GPU-loaded transformer pipelines
    effective_n_process = 1 if "transformer" in nlp.pipe_names else n_process

    docs = nlp.pipe(texts, batch_size=batch_size, n_process=effective_n_process)

    for text, doc in zip(texts, docs):
        if not text or not text.strip():
            results.append(ProcessedDoc())
            continue

        tokens, pos_tags, dep_labels, head_indices = [], [], [], []
        for tok in doc:
            tokens.append(tok.text)
            pos_tags.append(tok.pos_)
            dep_labels.append(tok.dep_)
            head_indices.append(tok.head.i)

        sent_boundaries = [(s.start, s.end) for s in doc.sents]
        tok_to_sent: dict[int, int] = {}
        for sent_idx, (start, end) in enumerate(sent_boundaries):
            for tok_i in range(start, end):
                tok_to_sent[tok_i] = sent_idx

        entities, entity_freq = [], {}
        for ent in doc.ents:
            if ent.label_ not in KEPT_ENTITY_TYPES:
                continue
            sent_idx = tok_to_sent.get(ent.start, 0)
            entities.append(EntitySpan(
                text=ent.text, label=ent.label_,
                start=ent.start_char, end=ent.end_char,
                sent_idx=sent_idx, token_idx=ent.root.i,
            ))
            key = ent.text.strip()
            entity_freq[key] = entity_freq.get(key, 0) + 1

        results.append(ProcessedDoc(
            tokens=tokens, pos_tags=pos_tags, dep_labels=dep_labels,
            head_indices=head_indices, sent_boundaries=sent_boundaries,
            entities=entities, entity_freq=entity_freq,
        ))

    return results
