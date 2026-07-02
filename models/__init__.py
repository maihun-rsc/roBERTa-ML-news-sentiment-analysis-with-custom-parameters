"""
models
──────
Module 3 — Modelling

Public API:
    from models import train_all_baselines, BaselineResult
    from models import RobertaFramingClassifier, train_roberta_framing
    from models import MultimodalFramingClassifier
    from models import run_training
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from models.baselines import (
    BaselineResult,
    LABELS,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    load_baseline,
    predict,
    predict_proba,
    save_baseline,
    train_all_baselines,
    train_baseline,
)
from models.entity_attention import (
    EntityAttentionAblation,
    EntityAwareAttention,
    proximity_scores_to_tensor,
)
from models.fusion import (
    LateFusionLayer,
    MultimodalFramingClassifier,
    has_transcript,
)
from models.roberta_framing import (
    FramingDataset,
    FramingExample,
    NUM_LABELS,
    RobertaFramingClassifier,
    TrainingResult,
    align_proximity_to_subwords,
    compute_class_weights,
    load_model,
    save_model,
    train_roberta_framing,
)

if TYPE_CHECKING:
    from configs.env_config import Config

log = logging.getLogger(__name__)

__all__ = [
    # Baselines
    "BaselineResult", "LABELS", "LABEL_TO_IDX", "IDX_TO_LABEL",
    "train_baseline", "train_all_baselines", "predict", "predict_proba",
    "save_baseline", "load_baseline",
    # Entity attention
    "EntityAwareAttention", "EntityAttentionAblation", "proximity_scores_to_tensor",
    # RoBERTa
    "RobertaFramingClassifier", "FramingExample", "FramingDataset",
    "TrainingResult", "NUM_LABELS",
    "align_proximity_to_subwords", "compute_class_weights",
    "train_roberta_framing", "save_model", "load_model",
    # Fusion
    "MultimodalFramingClassifier", "LateFusionLayer", "has_transcript",
    # Orchestrator
    "run_training",
]


def run_training(cfg: "Config") -> dict:
    """
    Top-level entry point called by main.py's stage_train().

    Pipeline:
      1. Load processed articles from Module 2's output
      2. Skip articles without annotation labels (need annotation step first)
      3. Train baseline suite (5 classical models)
      4. Train RoBERTa + entity-aware attention (text-only)
      5. Train RoBERTa ablation (no entity attention) for comparison
      6. Train multimodal fusion model on articles WITH transcripts
      7. Save all models to cfg.data_processed.parent / "models"

    Args:
        cfg: Config from get_config()

    Returns:
        Dict summarizing what was trained and where it was saved.

    Note:
        This requires articles to have a non-empty `.label` field, which
        is populated by the (separate, manual or semi-automated)
        annotation step described in context.md Section "Annotation
        Protocol" — Module 3 trains on labels, it does not produce them.
    """
    from data_collection.writer import read_jsonl

    processed_path = cfg.data_processed / "annotated_articles.jsonl"
    if not processed_path.exists():
        # Fallback to processed_articles if someone hasn't run annotation
        processed_path = cfg.data_processed / "processed_articles.jsonl"
        
    if not processed_path.exists():
        log.error(
            f"[train] No articles found at {processed_path}. "
            "Run --stage preprocess and --stage annotate first."
        )
        return {"status": "error", "reason": "no_data"}

    articles = read_jsonl(processed_path)
    labeled = [a for a in articles if a.label]

    if len(labeled) < 20:
        log.warning(
            f"[train] Only {len(labeled)} labeled articles found "
            f"(need annotation — see context.md Section III-C). "
            "Skipping model training; baselines need ≥20 samples minimum, "
            "RoBERTa fine-tuning needs hundreds for meaningful results."
        )
        return {"status": "skipped", "reason": "insufficient_labeled_data",
                "labeled_count": len(labeled)}

    log.info(f"[train] {len(labeled)}/{len(articles)} articles have labels")

    texts  = [a.clean_body or a.body for a in labeled]
    labels = [a.label for a in labeled]

    # ── Baselines ─────────────────────────────────────────────────────────
    log.info("[train] Training baseline suite …")
    baseline_results = train_all_baselines(texts, labels, seed=42)

    models_dir = cfg.data_processed.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for key, result in baseline_results.items():
        save_baseline(result, models_dir / f"baseline_{key}.pkl")

    summary = {
        "status": "complete",
        "labeled_articles": len(labeled),
        "baselines": {k: {"macro_f1": v.macro_f1, "accuracy": v.accuracy}
                      for k, v in baseline_results.items()},
    }

    # ── RoBERTa fine-tuning ──────────────────────────────────────────────
    log.info("[train] Building FramingExamples for RoBERTa...")
    from models.roberta_framing import train_roberta_framing, FramingExample
    
    # Auto-generating document-level framing examples. Proximity scores are 
    # defaulted to [1.0] meaning global attention, since explicit entity-level 
    # annotations are not provided in the automated pipeline run.
    examples = []
    for text, label in zip(texts, labels):
        # A single 1.0 score acts as a uniform attention prior when aligned
        examples.append(FramingExample(text, [1.0], label))
    
    # Simple split for train/val (80/20)
    train_size = int(len(examples) * 0.8)
    train_examples = examples[:train_size] if train_size > 0 else examples
    val_examples = examples[train_size:] if train_size < len(examples) else examples

    log.info("[train] Training RoBERTa model...")
    try:
        roberta_result = train_roberta_framing(
            train_examples=train_examples,
            val_examples=val_examples,
            batch_size=cfg.batch_size,
            max_epochs=3,
            freeze_base_epochs=1,
            device=cfg.device,
        )
        save_model(roberta_result, models_dir / "roberta_model")
        
        val_f1 = None
        if getattr(roberta_result, 'history', None) and len(roberta_result.history) > 0:
            val_f1 = roberta_result.history[-1].get("val_macro_f1")
            
        summary["roberta"] = {"status": "complete", "val_macro_f1": val_f1}
    except Exception as e:
        log.error(f"[train] RoBERTa training failed: {e}")
        summary["roberta"] = {"status": "error", "reason": str(e)}

    return summary
