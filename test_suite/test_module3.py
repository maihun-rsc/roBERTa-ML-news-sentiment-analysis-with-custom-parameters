"""
test_module3.py
───────────────
Unit tests for Module 3 — Modelling.

Baseline tests run quickly (sklearn on tiny synthetic data).
RoBERTa/fusion tests use real roberta-base weights downloaded from
HuggingFace — these are SLOWER (model download + forward pass) but
still run on CPU in reasonable time for a handful of tiny-batch tests.
If no network access is available, the RoBERTa-dependent tests are
skipped automatically (see _roberta_available check below).

Run:
    python test_module3.py
    python test_module3.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _roberta_available() -> bool:
    """Check if roberta-base can actually be loaded (network-dependent)."""
    try:
        from transformers import RobertaTokenizerFast
        RobertaTokenizerFast.from_pretrained("roberta-base")
        return True
    except Exception:
        return False


_HAS_ROBERTA = _roberta_available()
_SKIP_REASON = "roberta-base not downloadable in this environment (no network access)"


# ════════════════════════════════════════════════════════════════
#  Test: baselines.py
# ════════════════════════════════════════════════════════════════
class TestBaselines(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Small synthetic dataset — enough samples per class for stratified split."""
        cls.texts = [
            "The minister was praised for his excellent handling of the crisis.",
            "Officials commended the government's swift and effective response.",
            "The president received widespread support for the new policy.",
            "Critics slammed the minister for failing to address the crisis.",
            "The government was condemned for its poor handling of the situation.",
            "Officials were blamed for the ongoing failures in policy.",
            "The ministry released a routine statement on trade figures today.",
            "Officials confirmed the meeting took place as scheduled.",
            "The department issued a standard quarterly report on Tuesday.",
            "Authorities warned of an imminent and catastrophic crisis unfolding.",
            "Experts cautioned of a looming disaster if action is not taken.",
            "The agency sounded the alarm over a rapidly escalating emergency.",
        ] * 3  # repeat to get enough samples for stratified train/val split
        cls.labels = (
            ["Supportive"] * 3 + ["Critical"] * 3 +
            ["Neutral-Reporting"] * 3 + ["Alarmist"] * 3
        ) * 3

    def test_build_vectorizer_unigram(self):
        from models.baselines import build_vectorizer
        vec = build_vectorizer(ngram_range=(1, 1))
        X = vec.fit_transform(["hello world", "world hello again"])
        self.assertEqual(X.shape[0], 2)

    def test_build_vectorizer_does_not_remove_stopwords(self):
        """CRITICAL: negation words must survive vectorization.
        Uses the class-level corpus (12+ documents) rather than a single
        sentence, since TfidfVectorizer's min_df=2 default requires at
        least 2 documents to compute document frequency at all."""
        from models.baselines import build_vectorizer
        vec = build_vectorizer()
        corpus = self.texts + ["the minister did not deny the allegations at all"]
        vec.fit(corpus)
        vocab = vec.vocabulary_
        self.assertIn("not", vocab)

    def test_train_baseline_naive_bayes(self):
        from models.baselines import train_baseline
        result = train_baseline("naive_bayes", self.texts, self.labels)
        self.assertIsNotNone(result.model)
        self.assertGreaterEqual(result.macro_f1, 0.0)
        self.assertLessEqual(result.macro_f1, 1.0)

    def test_train_baseline_logistic_regression(self):
        from models.baselines import train_baseline
        result = train_baseline("logistic_regression", self.texts, self.labels)
        self.assertIsNotNone(result.model)

    def test_train_baseline_svm(self):
        from models.baselines import train_baseline
        result = train_baseline("svm", self.texts, self.labels, ngram_range=(1, 2))
        self.assertIsNotNone(result.model)

    def test_train_baseline_rejects_mismatched_lengths(self):
        from models.baselines import train_baseline
        with self.assertRaises(ValueError):
            train_baseline("svm", ["text1", "text2"], ["label1"])

    def test_train_baseline_rejects_too_few_samples(self):
        from models.baselines import train_baseline
        with self.assertRaises(ValueError):
            train_baseline("svm", ["a", "b", "c"], ["Supportive", "Critical", "Alarmist"])

    def test_train_baseline_unknown_model_name(self):
        from models.baselines import train_baseline
        with self.assertRaises(ValueError):
            train_baseline("not_a_real_model", self.texts, self.labels)

    def test_train_all_baselines_returns_all_configs(self):
        from models.baselines import train_all_baselines
        results = train_all_baselines(self.texts, self.labels)
        expected_keys = {"naive_bayes", "logistic_regression", "svm_unigram",
                         "svm_bigram", "random_forest"}
        self.assertEqual(set(results.keys()), expected_keys)

    def test_predict_returns_valid_labels(self):
        from models.baselines import train_baseline, predict, LABELS
        result = train_baseline("logistic_regression", self.texts, self.labels)
        preds = predict(result, ["The minister was widely praised today."])
        self.assertEqual(len(preds), 1)
        self.assertIn(preds[0], LABELS)

    def test_predict_proba_logistic_regression(self):
        from models.baselines import train_baseline, predict_proba
        result = train_baseline("logistic_regression", self.texts, self.labels)
        proba = predict_proba(result, ["Test sentence here."])
        self.assertIsNotNone(proba)
        self.assertEqual(proba.shape[0], 1)
        self.assertEqual(proba.shape[1], 4)  # 4 classes

    def test_predict_proba_svm_returns_none(self):
        """LinearSVC has no predict_proba — must return None, not crash."""
        from models.baselines import train_baseline, predict_proba
        result = train_baseline("svm", self.texts, self.labels)
        proba = predict_proba(result, ["Test sentence here."])
        self.assertIsNone(proba)

    def test_save_and_load_baseline_roundtrip(self):
        import tempfile
        from models.baselines import train_baseline, save_baseline, load_baseline, predict

        result = train_baseline("logistic_regression", self.texts, self.labels)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            save_baseline(result, path)
            loaded = load_baseline(path)

            self.assertEqual(loaded.name, result.name)
            self.assertAlmostEqual(loaded.macro_f1, result.macro_f1)

            preds_orig   = predict(result, ["Test sentence."])
            preds_loaded = predict(loaded, ["Test sentence."])
            self.assertEqual(preds_orig, preds_loaded)


# ════════════════════════════════════════════════════════════════
#  Test: entity_attention.py (pure PyTorch — no network needed)
# ════════════════════════════════════════════════════════════════
class TestEntityAttention(unittest.TestCase):

    def test_entity_aware_attention_output_shapes(self):
        import torch
        from models.entity_attention import EntityAwareAttention

        batch, seq_len, hidden_dim = 2, 10, 768
        attn = EntityAwareAttention(hidden_dim=hidden_dim, attn_dim=64)

        hidden_states = torch.randn(batch, seq_len, hidden_dim)
        proximity     = torch.rand(batch, seq_len)
        mask          = torch.ones(batch, seq_len, dtype=torch.long)

        c, alpha = attn(hidden_states, proximity, mask)

        self.assertEqual(c.shape, (batch, hidden_dim))
        self.assertEqual(alpha.shape, (batch, seq_len))

    def test_entity_aware_attention_weights_sum_to_one(self):
        import torch
        from models.entity_attention import EntityAwareAttention

        attn = EntityAwareAttention(hidden_dim=768, attn_dim=64, dropout=0.0)
        hidden_states = torch.randn(1, 5, 768)
        proximity     = torch.rand(1, 5)
        mask          = torch.ones(1, 5, dtype=torch.long)

        _, alpha = attn(hidden_states, proximity, mask)
        self.assertAlmostEqual(alpha.sum().item(), 1.0, places=4)

    def test_entity_aware_attention_respects_padding_mask(self):
        """Padded positions should receive ~0 attention weight."""
        import torch
        from models.entity_attention import EntityAwareAttention

        attn = EntityAwareAttention(hidden_dim=768, attn_dim=64, dropout=0.0)
        hidden_states = torch.randn(1, 6, 768)
        proximity     = torch.rand(1, 6)
        # First 3 tokens real, last 3 padding
        mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.long)

        _, alpha = attn(hidden_states, proximity, mask)
        padded_weight_sum = alpha[0, 3:].sum().item()
        self.assertLess(padded_weight_sum, 1e-3)

    def test_entity_attention_ablation_ignores_proximity(self):
        """Ablation variant must produce the same output regardless of
        proximity_scores content — it doesn't use them at all."""
        import torch
        from models.entity_attention import EntityAttentionAblation

        torch.manual_seed(42)
        attn = EntityAttentionAblation(hidden_dim=768, attn_dim=64, dropout=0.0)
        attn.eval()

        hidden_states = torch.randn(1, 5, 768)
        mask = torch.ones(1, 5, dtype=torch.long)

        c1, alpha1 = attn(hidden_states, proximity_scores=torch.rand(1, 5), attention_mask=mask)
        c2, alpha2 = attn(hidden_states, proximity_scores=torch.zeros(1, 5), attention_mask=mask)

        torch.testing.assert_close(c1, c2)
        torch.testing.assert_close(alpha1, alpha2)

    def test_proximity_scores_to_tensor_padding(self):
        from models.entity_attention import proximity_scores_to_tensor
        scores = [0.5, 0.8, 0.3]
        tensor = proximity_scores_to_tensor(scores, seq_len=6)
        self.assertEqual(tensor.shape[0], 6)
        self.assertEqual(tensor[3].item(), 0.0)  # padding region

    def test_proximity_scores_to_tensor_truncation(self):
        from models.entity_attention import proximity_scores_to_tensor
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        tensor = proximity_scores_to_tensor(scores, seq_len=3)
        self.assertEqual(tensor.shape[0], 3)
        self.assertAlmostEqual(tensor[0].item(), 0.1, places=4)
        self.assertAlmostEqual(tensor[2].item(), 0.3, places=4)

    def test_attention_score_higher_for_closer_proximity(self):
        """With identical token content, a token with a higher proximity
        score should receive more attention than one with a lower score —
        PROVIDED the Ws weights map higher s_i to a positive contribution.
        Random initialization does not guarantee this sign, so we fix Ws
        explicitly to isolate and verify the mechanism itself works,
        rather than depending on initialization luck."""
        import torch
        from models.entity_attention import EntityAwareAttention

        torch.manual_seed(0)
        attn = EntityAwareAttention(hidden_dim=8, attn_dim=4, dropout=0.0)
        attn.eval()

        # Force Ws to a known positive-definite-ish mapping so that
        # higher proximity -> higher Ws*si -> higher post-tanh score,
        # isolating the mechanism's directionality from init randomness.
        with torch.no_grad():
            attn.Ws.weight.fill_(1.0)   # (attn_dim, 1) all positive
            attn.w.weight.fill_(1.0)    # (1, attn_dim) all positive, so
                                         # higher tanh(...) -> higher score

        # Two tokens with IDENTICAL hidden state content, different proximity
        hidden_states = torch.zeros(1, 2, 8)  # identical content, all zeros
        proximity = torch.tensor([[1.0, 0.0]])  # token 0 close, token 1 far
        mask = torch.ones(1, 2, dtype=torch.long)

        _, alpha = attn(hidden_states, proximity, mask)
        # With We*hi=0 for both tokens (identical zero content) and Ws/w
        # fixed positive, token 0's higher proximity must win.
        self.assertGreater(alpha[0, 0].item(), alpha[0, 1].item())


# ════════════════════════════════════════════════════════════════
#  Test: roberta_framing.py — class weights (no network needed)
# ════════════════════════════════════════════════════════════════
class TestComputeClassWeights(unittest.TestCase):

    def test_balanced_classes_get_equal_weight(self):
        from models.roberta_framing import compute_class_weights, LABELS
        labels = LABELS * 10  # 10 of each, perfectly balanced
        weights = compute_class_weights(labels)
        for w in weights:
            self.assertAlmostEqual(w.item(), 1.0, places=4)

    def test_imbalanced_classes_get_inverse_weight(self):
        from models.roberta_framing import compute_class_weights, LABEL_TO_IDX
        # Neutral-Reporting heavily overrepresented, per the paper's ~43% stat
        labels = (
            ["Neutral-Reporting"] * 43 + ["Supportive"] * 19 +
            ["Critical"] * 28 + ["Alarmist"] * 10
        )
        weights = compute_class_weights(labels)
        neutral_w = weights[LABEL_TO_IDX["Neutral-Reporting"]].item()
        alarmist_w = weights[LABEL_TO_IDX["Alarmist"]].item()
        # Rarer class (Alarmist) should get a HIGHER weight than the
        # overrepresented class (Neutral-Reporting)
        self.assertGreater(alarmist_w, neutral_w)

    def test_missing_class_does_not_crash(self):
        from models.roberta_framing import compute_class_weights
        labels = ["Supportive", "Critical"]  # missing Neutral-Reporting, Alarmist
        weights = compute_class_weights(labels)
        self.assertEqual(len(weights), 4)


# ════════════════════════════════════════════════════════════════
#  Test: fusion.py — has_transcript logic (no network needed)
# ════════════════════════════════════════════════════════════════
class TestFusionHelpers(unittest.TestCase):

    def test_has_transcript_true_for_long_text(self):
        from models.fusion import has_transcript
        transcript = "This is a sufficiently long transcript from a broadcast clip. " * 2
        self.assertTrue(has_transcript(transcript))

    def test_has_transcript_false_for_empty(self):
        from models.fusion import has_transcript
        self.assertFalse(has_transcript(""))

    def test_has_transcript_false_for_too_short(self):
        from models.fusion import has_transcript
        self.assertFalse(has_transcript("Too short."))

    def test_has_transcript_respects_custom_min_length(self):
        from models.fusion import has_transcript
        self.assertTrue(has_transcript("12345", min_length=5))
        self.assertFalse(has_transcript("1234", min_length=5))


# ════════════════════════════════════════════════════════════════
#  Test: RoBERTa-dependent (network required) — skipped gracefully
# ════════════════════════════════════════════════════════════════
@unittest.skipUnless(_HAS_ROBERTA, _SKIP_REASON)
class TestRobertaFramingModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from transformers import RobertaTokenizerFast
        cls.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def test_model_forward_pass_shapes(self):
        import torch
        from models.roberta_framing import RobertaFramingClassifier

        model = RobertaFramingClassifier(use_entity_attn=True)
        model.eval()

        batch, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch, seq_len))
        attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
        proximity = torch.rand(batch, seq_len)

        with torch.no_grad():
            out = model(input_ids, attention_mask, proximity)

        self.assertEqual(out["logits"].shape, (batch, 4))
        self.assertEqual(out["attn_weights"].shape, (batch, seq_len))
        self.assertEqual(out["pooled"].shape, (batch, 768))

    def test_align_proximity_to_subwords_length(self):
        from models.roberta_framing import align_proximity_to_subwords

        text = "Boris Johnson visited London yesterday."
        # Rough per-word proximity scores (blank-spacy tokenization assumed
        # to roughly match word count for this simple test sentence)
        fake_scores = [0.9, 0.8, 0.5, 0.3, 0.1, 0.05]

        input_ids, aligned = align_proximity_to_subwords(
            text, fake_scores, self.tokenizer, max_length=32
        )
        self.assertEqual(len(input_ids), 32)
        self.assertEqual(len(aligned), 32)

    def test_align_proximity_special_tokens_get_zero(self):
        from models.roberta_framing import align_proximity_to_subwords

        text = "Test"
        input_ids, aligned = align_proximity_to_subwords(
            text, [1.0], self.tokenizer, max_length=8
        )
        # First token is <s> (BOS) — should get score 0.0
        self.assertEqual(aligned[0], 0.0)

    def test_framing_dataset_getitem(self):
        from models.roberta_framing import FramingDataset, FramingExample

        examples = [
            FramingExample(
                text="The minister was praised for his work.",
                proximity_scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                label="Supportive",
            )
        ]
        ds = FramingDataset(examples, self.tokenizer, max_length=32)
        item = ds[0]

        self.assertEqual(item["input_ids"].shape[0], 32)
        self.assertEqual(item["attention_mask"].shape[0], 32)
        self.assertEqual(item["proximity_scores"].shape[0], 32)
        self.assertEqual(item["label"].item(), 0)  # Supportive = index 0

    def test_train_roberta_framing_smoke_test(self):
        """
        Tiny end-to-end training smoke test — 1 epoch, 6 examples, batch_size=2.
        Not testing for accuracy, just that the training loop runs without
        crashing end-to-end (forward, loss, backward, optimizer step, eval).
        """
        from models.roberta_framing import train_roberta_framing, FramingExample

        examples = [
            FramingExample("The minister was widely praised.", [0.9, 0.8, 0.5, 0.3], "Supportive"),
            FramingExample("Officials commended the response.", [0.9, 0.8, 0.5, 0.3], "Supportive"),
            FramingExample("Critics slammed the minister harshly.", [0.9, 0.8, 0.5, 0.3], "Critical"),
            FramingExample("The government was condemned widely.", [0.9, 0.8, 0.5, 0.3], "Critical"),
            FramingExample("The ministry released a routine report.", [0.9, 0.8, 0.5, 0.3], "Neutral-Reporting"),
            FramingExample("Officials confirmed the scheduled meeting.", [0.9, 0.8, 0.5, 0.3], "Neutral-Reporting"),
        ]

        result = train_roberta_framing(
            train_examples=examples,
            val_examples=examples,   # same tiny set, just to confirm the loop runs
            batch_size=2,
            max_epochs=1,
            freeze_base_epochs=1,
            device="cpu",
        )

        self.assertIsNotNone(result.model)
        self.assertEqual(len(result.history), 1)
        self.assertIn("val_macro_f1", result.history[0])

    def test_save_and_load_model_roundtrip(self):
        import tempfile
        import torch
        from models.roberta_framing import (
            RobertaFramingClassifier, TrainingResult, save_model, load_model,
        )

        model = RobertaFramingClassifier(use_entity_attn=True)
        result = TrainingResult(model=model, tokenizer=self.tokenizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "saved_model"
            save_model(result, save_dir)

            loaded_model, loaded_tokenizer = load_model(save_dir, use_entity_attn=True)
            self.assertIsInstance(loaded_model, RobertaFramingClassifier)


@unittest.skipUnless(_HAS_ROBERTA, _SKIP_REASON)
class TestMultimodalFusion(unittest.TestCase):

    def test_multimodal_text_only_path(self):
        import torch
        from models.fusion import MultimodalFramingClassifier

        model = MultimodalFramingClassifier(share_encoder_weights=True)
        model.eval()

        batch, seq_len = 1, 12
        input_ids = torch.randint(0, 1000, (batch, seq_len))
        mask = torch.ones(batch, seq_len, dtype=torch.long)
        proximity = torch.rand(batch, seq_len)

        with torch.no_grad():
            out = model(input_ids, mask, proximity)  # no audio_* args

        self.assertEqual(out["modality"], "text_only")
        self.assertEqual(out["logits"].shape, (batch, 4))
        self.assertIsNone(out["audio_attn"])

    def test_multimodal_fused_path(self):
        import torch
        from models.fusion import MultimodalFramingClassifier

        model = MultimodalFramingClassifier(share_encoder_weights=True)
        model.eval()

        batch, seq_len = 1, 12
        input_ids = torch.randint(0, 1000, (batch, seq_len))
        mask = torch.ones(batch, seq_len, dtype=torch.long)
        proximity = torch.rand(batch, seq_len)

        with torch.no_grad():
            out = model(
                input_ids, mask, proximity,
                audio_input_ids=input_ids, audio_attention_mask=mask,
                audio_proximity_scores=proximity,
            )

        self.assertEqual(out["modality"], "multimodal")
        self.assertEqual(out["logits"].shape, (batch, 4))
        self.assertIsNotNone(out["audio_attn"])

    def test_late_fusion_layer_output_shape(self):
        import torch
        from models.fusion import LateFusionLayer

        layer = LateFusionLayer(hidden_dim=768)
        c_text  = torch.randn(2, 768)
        c_audio = torch.randn(2, 768)
        fused = layer(c_text, c_audio)
        self.assertEqual(fused.shape, (2, 768))


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not _HAS_ROBERTA:
        print(f"\n⚠ NOTE: {_SKIP_REASON}")
        print("  RoBERTa-dependent tests will be SKIPPED, not failed.\n")

    verbosity = 2 if "-v" in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
