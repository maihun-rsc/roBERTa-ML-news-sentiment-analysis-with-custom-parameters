"""
entity_attention.py
────────────────────
Implements the entity-aware attention mechanism exactly as formulated in
the paper's Section III-E (Mathematical Formulation):

    H = RoBERTa(x) = (h1, h2, ..., hn),   hi in R^d
    alpha_i = softmax(w^T tanh(We*hi + Ws*si))
    c = Sum_i  alpha_i * hi

where s_i is the syntactic proximity score from Module 2's
preprocessing.proximity_scorer (inverse shortest path length in the
dependency tree between token i and the target entity's head token).

This is the component that makes the model "entity-aware" rather than
a vanilla document classifier — c is computed PER TARGET ENTITY, so the
same article produces different weighted representations depending on
which entity's framing is being scored.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class EntityAwareAttention(nn.Module):
    """
    Entity-aware attention layer.

    Computes attention weights over encoder hidden states, conditioned on
    BOTH the hidden state content (We*hi) AND the syntactic proximity to
    the target entity (Ws*si), per the paper's Eq. in Section III-E.

    Args:
        hidden_dim: dimensionality of RoBERTa hidden states (768 for
                    roberta-base, per model_config.yaml)
        attn_dim:   internal projection dimension for the attention
                    scoring MLP (smaller than hidden_dim is standard —
                    this is a bottleneck, not a second encoder)
        dropout:    dropout applied to attention weights before pooling
    """

    def __init__(self, hidden_dim: int = 768, attn_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim   = attn_dim

        # We: projects token hidden states into attention space
        self.We = nn.Linear(hidden_dim, attn_dim, bias=False)
        # Ws: projects the scalar proximity score into the SAME attention
        # space so it can be added to We*hi before the tanh nonlinearity.
        # Per the equation, si is a scalar per token — Ws maps R^1 -> R^attn_dim.
        self.Ws = nn.Linear(1, attn_dim, bias=False)
        # w: the final scoring vector that collapses attn_dim -> scalar score
        self.w  = nn.Linear(attn_dim, 1, bias=True)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        # Xavier init is standard for attention projections — keeps the
        # tanh nonlinearity in its non-saturating range at initialization.
        nn.init.xavier_uniform_(self.We.weight)
        nn.init.xavier_uniform_(self.Ws.weight)
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,      # (batch, seq_len, hidden_dim)
        proximity_scores: torch.Tensor,    # (batch, seq_len)
        attention_mask: torch.Tensor | None = None,  # (batch, seq_len), 1=real token, 0=padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states:     RoBERTa's last_hidden_state output, H in the
                                paper's notation. Shape (batch, seq_len, hidden_dim).
            proximity_scores:  s_i values from proximity_scorer.compute_proximity_scores,
                                one per token. Shape (batch, seq_len).
            attention_mask:    standard transformer padding mask. Padded
                                positions get -inf score before softmax so
                                they contribute zero weight to c.

        Returns:
            c:      weighted document representation, shape (batch, hidden_dim).
                    This is the paper's "c = Sum_i alpha_i * hi".
            alpha:  the attention weights themselves, shape (batch, seq_len).
                    Returned for interpretability / visualization — lets you
                    inspect which tokens the model actually attended to for
                    a given entity.
        """
        batch, seq_len, _ = hidden_states.shape

        # proximity_scores: (batch, seq_len) -> (batch, seq_len, 1) for Ws projection
        s_i = proximity_scores.unsqueeze(-1)

        # We*hi : (batch, seq_len, attn_dim)
        we_hi = self.We(hidden_states)
        # Ws*si : (batch, seq_len, attn_dim)
        ws_si = self.Ws(s_i)

        # tanh(We*hi + Ws*si) : (batch, seq_len, attn_dim)
        combined = torch.tanh(we_hi + ws_si)

        # w^T * combined : (batch, seq_len, 1) -> (batch, seq_len)
        scores = self.w(combined).squeeze(-1)

        if attention_mask is not None:
            # Padding positions get -inf so softmax assigns them ~0 weight.
            # Use a large negative number rather than literal -inf to avoid
            # NaN propagation if an entire row were masked (shouldn't happen
            # in practice, but defensive against malformed batches).
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attention_mask == 0, mask_value)

        alpha = F.softmax(scores, dim=-1)          # (batch, seq_len)
        alpha = self.dropout(alpha)

        # c = Sum_i alpha_i * hi  — weighted sum over the sequence dimension
        # alpha: (batch, seq_len) -> (batch, seq_len, 1) for broadcasting
        c = torch.sum(alpha.unsqueeze(-1) * hidden_states, dim=1)  # (batch, hidden_dim)

        return c, alpha


class EntityAttentionAblation(nn.Module):
    """
    Ablation variant — standard self-attention WITHOUT the proximity term.
    This is the "RoBERTa (no entity attn — ablation)" row in the paper's
    Table III, used to isolate the contribution of entity-aware weighting
    (reported as +3.2 macro-F1 in the paper).

    Architecturally identical to EntityAwareAttention but with Ws removed —
    attention is purely a function of token content, not entity proximity.
    """

    def __init__(self, hidden_dim: int = 768, attn_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.We = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.w  = nn.Linear(attn_dim, 1, bias=True)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.We.weight)
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        proximity_scores: torch.Tensor | None = None,  # accepted but ignored — keeps call signature compatible
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.w(torch.tanh(self.We(hidden_states))).squeeze(-1)

        if attention_mask is not None:
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attention_mask == 0, mask_value)

        alpha = F.softmax(scores, dim=-1)
        alpha = self.dropout(alpha)
        c = torch.sum(alpha.unsqueeze(-1) * hidden_states, dim=1)
        return c, alpha


def proximity_scores_to_tensor(
    proximity_scores: list[float],
    seq_len: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Convert a Module 2 proximity score list (variable length, one per
    spaCy token) into a fixed-length tensor aligned with RoBERTa's
    tokenization (which uses a DIFFERENT tokenizer — BPE subwords, not
    spaCy tokens).

    IMPORTANT: This function assumes proximity_scores has ALREADY been
    re-aligned from spaCy token indices to RoBERTa subword indices by
    the caller (see models/roberta_framing.py's `align_proximity_to_subwords`).
    Padding/truncation to seq_len happens here.

    Args:
        proximity_scores: per-subword-token proximity scores (pre-aligned)
        seq_len:           target sequence length (RoBERTa's padded length)
        device:            torch device for the output tensor

    Returns:
        Tensor of shape (seq_len,), zero-padded or truncated as needed.
    """
    scores = list(proximity_scores[:seq_len])
    if len(scores) < seq_len:
        scores = scores + [0.0] * (seq_len - len(scores))
    return torch.tensor(scores, dtype=torch.float32, device=device)
