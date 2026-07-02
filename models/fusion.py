"""
fusion.py
─────────
Late fusion module combining the text-encoder representation (ct) and the
ASR-transcript-encoder representation (ca), per the paper's Section III-E:

    c_fused = Wf [ct ; ca] + bf

where [;] denotes concatenation and Wf projects back down to the model's
hidden dimension. This is the "Multimodal RoBERTa + ASR Fusion" row in
Table III — target macro-F1 = 0.821, a +0.7 improvement over text-only.

Both ct and ca come from the SAME RobertaFramingClassifier architecture
(one instance encodes the written article, a second instance — or the
same instance run twice — encodes the Whisper transcript from Module 2's
asr_cleaner output). Architecture is shared; weights may or may not be
shared depending on `share_encoder_weights`.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from models.roberta_framing import NUM_LABELS, RobertaFramingClassifier

log = logging.getLogger(__name__)


class LateFusionLayer(nn.Module):
    """
    Projects concatenated [ct ; ca] back to hidden_dim, per Eq.:
        c_fused = Wf [ct ; ca] + bf

    Args:
        hidden_dim: dimensionality of each individual representation
                    (768 for roberta-base) — the concatenated input is
                    2*hidden_dim, the output is back to hidden_dim.
        dropout:    dropout applied after the fusion projection
    """

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.Wf = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        nn.init.xavier_uniform_(self.Wf.weight)
        nn.init.zeros_(self.Wf.bias)

    def forward(self, c_text: torch.Tensor, c_audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c_text:  (batch, hidden_dim) — text encoder's entity-weighted
                     representation (the 'pooled' output of
                     RobertaFramingClassifier.forward)
            c_audio: (batch, hidden_dim) — ASR transcript encoder's
                     equivalent representation

        Returns:
            c_fused: (batch, hidden_dim)
        """
        concatenated = torch.cat([c_text, c_audio], dim=-1)   # (batch, 2*hidden_dim)
        fused = self.Wf(concatenated)                          # (batch, hidden_dim)
        fused = self.activation(fused)
        fused = self.dropout(fused)
        return fused


class MultimodalFramingClassifier(nn.Module):
    """
    Full multimodal model: two encoder branches (text + ASR transcript),
    each with their own entity-aware attention, fused via LateFusionLayer,
    then classified.

    Per context.md: "Late Fusion -- Framing Classifier -> 4-Class Output".

    Args:
        encoder_name:           HF model id, shared architecture for both branches
        num_labels:             output classes
        attn_dim:                entity-attention bottleneck dim
        dropout:                 dropout throughout
        share_encoder_weights:   if True, the text and audio branches use
                                 the SAME RoBERTa weights (tied) — saves
                                 memory and is a reasonable prior since
                                 both are English news language, just
                                 different registers (written vs spoken).
                                 If False, two independent encoders are
                                 trained — more capacity, more memory.
    """

    def __init__(
        self,
        encoder_name: str = "roberta-base",
        num_labels: int = NUM_LABELS,
        attn_dim: int = 256,
        dropout: float = 0.1,
        share_encoder_weights: bool = True,
    ) -> None:
        super().__init__()

        self.text_branch = RobertaFramingClassifier(
            encoder_name=encoder_name, num_labels=num_labels,
            attn_dim=attn_dim, dropout=dropout, use_entity_attn=True,
        )

        if share_encoder_weights:
            # Audio branch reuses the SAME encoder + attention module objects.
            # Only the classifier head differs (we don't use the audio
            # branch's classifier at all — see forward(); only its
            # encoder+attention 'pooled' output feeds the fusion layer).
            self.audio_branch = self.text_branch
        else:
            self.audio_branch = RobertaFramingClassifier(
                encoder_name=encoder_name, num_labels=num_labels,
                attn_dim=attn_dim, dropout=dropout, use_entity_attn=True,
            )

        hidden_dim = self.text_branch.encoder.config.hidden_size
        self.fusion = LateFusionLayer(hidden_dim=hidden_dim, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, num_labels),
        )

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        text_proximity_scores: torch.Tensor,
        audio_input_ids: torch.Tensor | None = None,
        audio_attention_mask: torch.Tensor | None = None,
        audio_proximity_scores: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass. If audio_* args are None, falls back to text-only
        prediction (using the text branch's own classifier) — this lets
        the SAME model class serve both Table III rows ("Proposed text
        only" and "Multimodal + ASR Fusion") depending on whether a
        broadcast transcript is available for a given article.

        Args:
            text_input_ids, text_attention_mask, text_proximity_scores:
                standard RobertaFramingClassifier inputs for the article text
            audio_input_ids, audio_attention_mask, audio_proximity_scores:
                same shape, but for the Whisper transcript. Optional.

        Returns:
            dict with 'logits', and 'text_attn'/'audio_attn' weights for
            interpretability where applicable.
        """
        text_out = self.text_branch(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            proximity_scores=text_proximity_scores,
        )
        c_text = text_out["pooled"]

        if audio_input_ids is None:
            # No transcript available for this article — text-only path.
            # Use the text branch's own classifier directly (equivalent
            # to running RobertaFramingClassifier alone).
            return {
                "logits": text_out["logits"],
                "text_attn": text_out["attn_weights"],
                "audio_attn": None,
                "modality": "text_only",
            }

        audio_out = self.audio_branch(
            input_ids=audio_input_ids,
            attention_mask=audio_attention_mask,
            proximity_scores=audio_proximity_scores,
        )
        c_audio = audio_out["pooled"]

        c_fused = self.fusion(c_text, c_audio)
        logits = self.classifier(c_fused)

        return {
            "logits": logits,
            "text_attn": text_out["attn_weights"],
            "audio_attn": audio_out["attn_weights"],
            "modality": "multimodal",
        }


def has_transcript(article_transcript: str, min_length: int = 50) -> bool:
    """
    Decide whether an article has a usable ASR transcript for the
    multimodal path, or should fall back to text-only.

    Args:
        article_transcript: Article.transcript field (from Module 2's
                             asr_cleaner output, empty string if no
                             broadcast source)
        min_length:          minimum character length to consider usable

    Returns:
        True if the multimodal path should be used.
    """
    return bool(article_transcript and len(article_transcript.strip()) >= min_length)
