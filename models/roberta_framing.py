"""
roberta_framing.py
───────────────────
The primary model: RoBERTa-base encoder + entity-aware attention head,
fine-tuned for 4-class entity-centric framing classification.

Implements the full pipeline from the paper's Section III-E/F:
    H = RoBERTa(x)
    c = EntityAwareAttention(H, proximity_scores)
    y_hat = softmax(W2 * ReLU(W1*c + b1) + b2)
    L = -Sum_k wk*yk*log(y_hat_k)     (class-weighted cross-entropy)

This is the "Proposed (text only)" model from Table III — target
macro-F1 = 0.814 with entity-aware attention, vs 0.801 without (ablation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    RobertaModel,
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup,
)

from models.entity_attention import (
    EntityAttentionAblation,
    EntityAwareAttention,
    proximity_scores_to_tensor,
)

log = logging.getLogger(__name__)

LABELS = ["Supportive", "Critical", "Neutral-Reporting", "Alarmist"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}
NUM_LABELS = len(LABELS)


# ════════════════════════════════════════════════════════════════
#  Model
# ════════════════════════════════════════════════════════════════

class RobertaFramingClassifier(nn.Module):
    """
    RoBERTa-base + entity-aware attention + 2-layer MLP classification head.

    Architecture (per model_config.yaml):
        encoder:      roberta-base, hidden_dim=768
        attention:    EntityAwareAttention(768, 256) — or ablation variant
        classifier:   Linear(768,256) -> ReLU -> Dropout -> Linear(256,4)

    Args:
        encoder_name:    HuggingFace model id (default 'roberta-base')
        num_labels:      output classes (4 — see LABELS above)
        attn_dim:        entity-attention bottleneck dimension
        dropout:         dropout in classification head and attention
        use_entity_attn: True = EntityAwareAttention, False = ablation
                         (plain attention with no proximity term — this
                         IS the "RoBERTa (no entity attn — ablation)" row)
        freeze_encoder:  if True, RoBERTa weights are frozen (used during
                         freeze_base_epochs per model_config.yaml — lets
                         the attention+classifier head warm up before the
                         encoder starts fine-tuning, reduces early
                         catastrophic forgetting of pretrained weights)
    """

    def __init__(
        self,
        encoder_name: str = "roberta-base",
        num_labels: int = NUM_LABELS,
        attn_dim: int = 256,
        dropout: float = 0.1,
        use_entity_attn: bool = True,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = RobertaModel.from_pretrained(encoder_name)
        hidden_dim = self.encoder.config.hidden_size  # 768 for roberta-base

        self.use_entity_attn = use_entity_attn
        if use_entity_attn:
            self.attention: nn.Module = EntityAwareAttention(hidden_dim, attn_dim, dropout)
        else:
            self.attention = EntityAttentionAblation(hidden_dim, attn_dim, dropout)

        # Two-layer feed-forward classifier, per Eq. in Section III-E:
        #   y_hat = softmax(W2 * ReLU(W1*c + b1) + b2)
        # NOTE: we return logits, not softmax — nn.CrossEntropyLoss applies
        # log-softmax internally and is more numerically stable than
        # computing softmax ourselves and feeding it to NLLLoss.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, num_labels),
        )

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        log.info("[model] Encoder frozen")

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True
        log.info("[model] Encoder unfrozen")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        proximity_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids:        (batch, seq_len) RoBERTa token ids
            attention_mask:    (batch, seq_len) padding mask, 1=real, 0=pad
            proximity_scores:  (batch, seq_len) per-subword s_i scores,
                                already aligned via align_proximity_to_subwords

        Returns:
            dict with:
              'logits':     (batch, num_labels) — pre-softmax class scores
              'attn_weights': (batch, seq_len) — alpha_i for interpretability
              'pooled':     (batch, hidden_dim) — the entity-weighted c vector
                            (useful as input to fusion.py for multimodal)
        """
        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = encoder_out.last_hidden_state  # H in the paper's notation

        c, alpha = self.attention(
            hidden_states=hidden_states,
            proximity_scores=proximity_scores,
            attention_mask=attention_mask,
        )

        logits = self.classifier(c)

        return {"logits": logits, "attn_weights": alpha, "pooled": c}


# ════════════════════════════════════════════════════════════════
#  Subword alignment — spaCy tokens -> RoBERTa BPE subwords
# ════════════════════════════════════════════════════════════════

def align_proximity_to_subwords(
    text: str,
    spacy_proximity_scores: list[float],
    tokenizer: RobertaTokenizerFast,
    max_length: int = 512,
) -> tuple[list[int], list[float]]:
    """
    RoBERTa's tokenizer splits words into BPE subwords, which do NOT align
    1:1 with spaCy's whitespace-aware tokens. This function re-maps Module
    2's per-spaCy-token proximity scores onto RoBERTa's subword sequence
    using the tokenizer's offset_mapping.

    Strategy: each RoBERTa subword inherits the proximity score of the
    spaCy token whose character span it falls within. Special tokens
    (<s>, </s>, <pad>) get score 0.0 — they're not part of any entity's
    syntactic neighborhood.

    Args:
        text:                    the ORIGINAL text (must match what
                                  spacy_proximity_scores was computed over)
        spacy_proximity_scores:  one score per spaCy token, in order
        tokenizer:               RoBERTa fast tokenizer (needed for
                                  offset_mapping — slow tokenizers don't
                                  support this)
        max_length:              truncation length, must match what's
                                  used in the actual model forward pass

    Returns:
        (input_ids, aligned_scores) — both length max_length (padded/truncated)
    """
    import spacy

    # We need spaCy's character offsets to map onto RoBERTa's offsets.
    # This requires re-tokenizing with spaCy here since we only have the
    # SCORES, not the original Doc object, at this call site. In practice,
    # the caller (run_training / dataset __getitem__) should pass the
    # spaCy Doc's token char-spans directly rather than re-tokenizing —
    # this fallback exists for standalone testing of this function.
    nlp = spacy.blank("en")
    doc = nlp(text)

    doc_len = len(list(doc))
    if len(spacy_proximity_scores) == 1 and doc_len > 1:
        # Broadcast single global prior (e.g. [1.0]) to all spaCy tokens
        spacy_proximity_scores = spacy_proximity_scores * doc_len
    elif doc_len != len(spacy_proximity_scores):
        log.warning(
            f"[align] Token count mismatch: spaCy retokenized to "
            f"{doc_len} tokens but {len(spacy_proximity_scores)} "
            f"scores were provided. Scores will be truncated/padded with 0."
        )

    spacy_spans: list[tuple[int, int]] = [(tok.idx, tok.idx + len(tok.text)) for tok in doc]

    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )

    input_ids = encoding["input_ids"]
    offsets   = encoding["offset_mapping"]

    aligned_scores: list[float] = []
    for (start, end) in offsets:
        if start == end:
            # Special token (<s>, </s>, <pad>) — offset (0,0) by convention
            aligned_scores.append(0.0)
            continue

        # Find which spaCy token this subword's char range overlaps with
        score = 0.0
        for tok_idx, (sp_start, sp_end) in enumerate(spacy_spans):
            if start < sp_end and end > sp_start:  # overlap test
                if tok_idx < len(spacy_proximity_scores):
                    score = spacy_proximity_scores[tok_idx]
                break
        aligned_scores.append(score)

    return input_ids, aligned_scores


# ════════════════════════════════════════════════════════════════
#  Dataset
# ════════════════════════════════════════════════════════════════

@dataclass
class FramingExample:
    """One training example: an article + target entity + gold label."""
    text: str
    proximity_scores: list[float]    # per-spaCy-token, pre-computed by Module 2
    label: str
    entity_text: str = ""             # for logging/debugging only


class FramingDataset(Dataset):
    """
    PyTorch Dataset wrapping FramingExample list, doing subword alignment
    and tensor conversion at __getitem__ time (not precomputed — keeps
    memory bounded for large corpora; the alignment cost is small relative
    to the RoBERTa forward pass it feeds).
    """

    def __init__(
        self,
        examples: list[FramingExample],
        tokenizer: RobertaTokenizerFast,
        max_length: int = 512,
    ) -> None:
        self.examples  = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]

        input_ids, aligned_scores = align_proximity_to_subwords(
            text=ex.text,
            spacy_proximity_scores=ex.proximity_scores,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        attention_mask = [1 if tid != self.tokenizer.pad_token_id else 0 for tid in input_ids]

        return {
            "input_ids":        torch.tensor(input_ids, dtype=torch.long),
            "attention_mask":   torch.tensor(attention_mask, dtype=torch.long),
            "proximity_scores": torch.tensor(aligned_scores, dtype=torch.float32),
            "label":            torch.tensor(LABEL_TO_IDX[ex.label], dtype=torch.long),
        }


# ════════════════════════════════════════════════════════════════
#  Class-weighted loss
# ════════════════════════════════════════════════════════════════

def compute_class_weights(labels: list[str]) -> torch.Tensor:
    """
    Inverse-frequency class weighting per the paper's Eq.:
        wk = N / (K * Nk)

    Compensates for the Neutral-Reporting class imbalance (~43% of corpus
    per context.md / the paper's dataset description).

    Args:
        labels: list of framing labels across the training set

    Returns:
        Tensor of shape (num_labels,), ready for nn.CrossEntropyLoss(weight=...)
    """
    n_total = len(labels)
    k = NUM_LABELS

    weights = torch.ones(k)
    for label, idx in LABEL_TO_IDX.items():
        n_k = sum(1 for l in labels if l == label)
        if n_k > 0:
            weights[idx] = n_total / (k * n_k)
        else:
            log.warning(f"[weights] No examples found for label '{label}' — weight set to 1.0")

    return weights


# ════════════════════════════════════════════════════════════════
#  Training loop
# ════════════════════════════════════════════════════════════════

@dataclass
class TrainingResult:
    model: RobertaFramingClassifier
    tokenizer: RobertaTokenizerFast
    history: list[dict[str, float]] = field(default_factory=list)
    best_val_f1: float = 0.0


def train_roberta_framing(
    train_examples: list[FramingExample],
    val_examples: list[FramingExample],
    encoder_name: str = "roberta-base",
    use_entity_attn: bool = True,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_epochs: int = 5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    gradient_clip: float = 1.0,
    freeze_base_epochs: int = 1,
    early_stopping_patience: int = 2,
    device: str = "cpu",
    seed: int = 42,
) -> TrainingResult:
    """
    Full fine-tuning loop matching model_config.yaml's training section.

    Args:
        train_examples, val_examples: FramingExample lists
        encoder_name:        HF model id
        use_entity_attn:      True for the proposed model, False for ablation
        batch_size:           per model_config.yaml (16 GPU / 4 CPU)
        learning_rate:        2e-5 per config
        max_epochs:           5 per config
        warmup_ratio:         0.1 per config (10% of total steps)
        weight_decay:         0.01 per config
        gradient_clip:        1.0 per config
        freeze_base_epochs:   freeze encoder for this many initial epochs
        early_stopping_patience: stop if val macro-F1 doesn't improve for
                                  this many consecutive epochs
        device:               'cuda' | 'mps' | 'cpu'
        seed:                 random seed

    Returns:
        TrainingResult with the best-checkpoint model + training history.
    """
    from sklearn.metrics import f1_score

    torch.manual_seed(seed)

    tokenizer = RobertaTokenizerFast.from_pretrained(encoder_name)
    model = RobertaFramingClassifier(
        encoder_name=encoder_name,
        use_entity_attn=use_entity_attn,
        freeze_encoder=(freeze_base_epochs > 0),
    ).to(device)

    train_labels = [ex.label for ex in train_examples]
    class_weights = compute_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_ds = FramingDataset(train_examples, tokenizer)
    val_ds   = FramingDataset(val_examples, tokenizer)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = len(train_loader) * max_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    history: list[dict[str, float]] = []
    best_val_f1 = 0.0
    best_state: dict[str, Any] | None = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        if epoch == freeze_base_epochs + 1:
            model.unfreeze_encoder()

        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                proximity_scores=batch["proximity_scores"],
            )
            loss = criterion(out["logits"], batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    proximity_scores=batch["proximity_scores"],
                )
                loss = criterion(out["logits"], batch["label"])
                val_loss += loss.item()

                preds = out["logits"].argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["label"].cpu().tolist())

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        log.info(
            f"[train] Epoch {epoch}/{max_epochs} | "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

        history.append({
            "epoch": epoch, "train_loss": avg_train_loss,
            "val_loss": avg_val_loss, "val_macro_f1": val_macro_f1,
        })

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            log.info(f"[train] ✓ New best val_macro_f1={best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                log.info(f"[train] Early stopping at epoch {epoch} "
                         f"(no improvement for {patience_counter} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return TrainingResult(
        model=model, tokenizer=tokenizer,
        history=history, best_val_f1=best_val_f1,
    )


def save_model(result: TrainingResult, save_dir: Path) -> None:
    """Save model weights + tokenizer for later inference."""
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), save_dir / "model.pt")
    result.tokenizer.save_pretrained(save_dir / "tokenizer")
    log.info(f"[model] Saved to {save_dir}")


def load_model(
    save_dir: Path,
    encoder_name: str = "roberta-base",
    use_entity_attn: bool = True,
    device: str = "cpu",
) -> tuple[RobertaFramingClassifier, RobertaTokenizerFast]:
    """Load a previously saved model + tokenizer."""
    model = RobertaFramingClassifier(encoder_name=encoder_name, use_entity_attn=use_entity_attn)
    model.load_state_dict(torch.load(save_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()

    tokenizer = RobertaTokenizerFast.from_pretrained(save_dir / "tokenizer")
    return model, tokenizer
