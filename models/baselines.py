"""
baselines.py
────────────
Classical ML baselines per context.md Module 3 spec:
    "Baseline classifiers: LogisticRegression, LinearSVC, MultinomialNB.
     Also used for TF-IDF vectorization in the baseline pipeline."

These establish the performance floor against which RoBERTa + entity-aware
attention is benchmarked (Table II in the paper — SVM bigram TF-IDF is the
strongest classical baseline at macro-F1 = 0.668).

Document-level classification only — these models don't have an entity-
aware mechanism, which is precisely the point: they're the "what happens
without entity awareness" comparison point.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

LABELS = ["Supportive", "Critical", "Neutral-Reporting", "Alarmist"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}


@dataclass
class BaselineResult:
    """Holds one trained baseline model + its vectorizer + eval metrics."""
    name: str
    model: Any
    vectorizer: TfidfVectorizer
    macro_f1: float = 0.0
    accuracy: float = 0.0
    per_class_f1: dict[str, float] = field(default_factory=dict)
    report: str = ""


# ── Vectorization ─────────────────────────────────────────────────────────────

def build_vectorizer(
    ngram_range: tuple[int, int] = (1, 1),
    max_features: int = 20_000,
    min_df: int = 2,
) -> TfidfVectorizer:
    """
    Build a TF-IDF vectorizer.

    Per context.md feature engineering section: word n-grams are the
    primary lexical feature. Bigrams (1,2) are what made SVM the
    strongest classical baseline in the paper's Table II — unigrams
    alone miss attribution-verb-plus-object patterns like "slammed the
    minister" that signal Critical framing.

    Args:
        ngram_range:  (1,1) for unigram, (1,2) for unigram+bigram
        max_features: vocabulary cap
        min_df:       minimum document frequency to keep a term

    Returns:
        Unfitted TfidfVectorizer.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,      # log-scale TF — standard improvement
        stop_words=None,        # CRITICAL: do not remove stopwords (negation matters)
    )


# ── Training ──────────────────────────────────────────────────────────────────

def _class_weighted(model_cls: type, **kwargs: Any) -> Any:
    """
    Instantiate a model with class_weight='balanced' — compensates for the
    Neutral-Reporting class imbalance (~43% of corpus per context.md).
    """
    return model_cls(class_weight="balanced", **kwargs)


def train_baseline(
    name: str,
    texts: list[str],
    labels: list[str],
    ngram_range: tuple[int, int] = (1, 1),
    max_features: int = 20_000,
    seed: int = 42,
) -> BaselineResult:
    """
    Train one classical baseline model end-to-end: vectorize, fit, hold out
    20% for an internal sanity-check eval (the REAL eval happens in
    evaluation/metrics.py on the proper test split — this is just a smoke
    check during training).

    Args:
        name:         'naive_bayes' | 'logistic_regression' | 'svm' | 'random_forest'
        texts:        list of article bodies (clean_body from Module 2)
        labels:       list of framing labels, same length as texts
        ngram_range:  TF-IDF n-gram range
        max_features: TF-IDF vocabulary cap
        seed:         random seed

    Returns:
        BaselineResult with fitted model + vectorizer + smoke-test metrics.
    """
    if len(texts) != len(labels):
        raise ValueError(f"texts ({len(texts)}) and labels ({len(labels)}) length mismatch")
    if len(texts) < 10:
        raise ValueError(f"Need at least 10 samples to train, got {len(texts)}")

    vectorizer = build_vectorizer(ngram_range=ngram_range, max_features=max_features)
    X = vectorizer.fit_transform(texts)
    y = np.array([LABEL_TO_IDX[l] for l in labels])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if len(set(y)) > 1 else None
    )

    model = _build_model(name, seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    accuracy = float((y_pred == y_val).mean())

    present_labels = sorted(set(y_val) | set(y_pred))
    target_names = [IDX_TO_LABEL[i] for i in present_labels]
    report = classification_report(
        y_val, y_pred, labels=present_labels, target_names=target_names, zero_division=0,
    )

    per_class = {}
    per_class_scores = f1_score(y_val, y_pred, average=None, labels=present_labels, zero_division=0)
    for label_idx, score in zip(present_labels, per_class_scores):
        per_class[IDX_TO_LABEL[label_idx]] = float(score)

    log.info(f"[baseline/{name}] macro-F1={macro_f1:.3f} acc={accuracy:.3f} "
             f"(n_train={X_train.shape[0]}, n_val={X_val.shape[0]})")

    return BaselineResult(
        name=name, model=model, vectorizer=vectorizer,
        macro_f1=macro_f1, accuracy=accuracy,
        per_class_f1=per_class, report=report,
    )


def _build_model(name: str, seed: int) -> Any:
    if name == "naive_bayes":
        # MultinomialNB doesn't support class_weight directly
        return MultinomialNB()
    elif name == "logistic_regression":
        return _class_weighted(
            LogisticRegression, max_iter=1000, random_state=seed, n_jobs=-1
        )
    elif name == "svm":
        return _class_weighted(LinearSVC, max_iter=2000, random_state=seed)
    elif name == "random_forest":
        return _class_weighted(
            RandomForestClassifier, n_estimators=200, random_state=seed, n_jobs=-1
        )
    else:
        raise ValueError(
            f"Unknown baseline '{name}'. Choose from: "
            "naive_bayes, logistic_regression, svm, random_forest"
        )


def train_all_baselines(
    texts: list[str],
    labels: list[str],
    seed: int = 42,
) -> dict[str, BaselineResult]:
    """
    Train the full baseline suite per Table II of the paper:
        Multinomial Naive Bayes
        Logistic Regression (unigram TF-IDF)
        SVM Linear (unigram TF-IDF)
        SVM Linear (bigram TF-IDF)   ← strongest classical baseline
        Random Forest (TF-IDF)

    Args:
        texts:  article bodies
        labels: framing labels
        seed:   random seed

    Returns:
        {model_key: BaselineResult} for all 5 configurations.
    """
    results: dict[str, BaselineResult] = {}

    configs = [
        ("naive_bayes",             "naive_bayes",         (1, 1)),
        ("logistic_regression",     "logistic_regression", (1, 1)),
        ("svm_unigram",             "svm",                 (1, 1)),
        ("svm_bigram",              "svm",                 (1, 2)),
        ("random_forest",           "random_forest",       (1, 1)),
    ]

    for key, model_name, ngram in configs:
        log.info(f"[baselines] Training {key} …")
        try:
            results[key] = train_baseline(
                name=model_name, texts=texts, labels=labels,
                ngram_range=ngram, seed=seed,
            )
        except Exception as e:
            log.error(f"[baselines] Failed to train {key}: {e}")

    # Summary table
    log.info("\n[baselines] ── Summary ──")
    for key, result in sorted(results.items(), key=lambda kv: -kv[1].macro_f1):
        log.info(f"  {key:25} macro-F1={result.macro_f1:.3f}  acc={result.accuracy:.3f}")

    return results


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(result: BaselineResult, texts: list[str]) -> list[str]:
    """
    Run inference with a trained baseline.

    Args:
        result: BaselineResult from train_baseline
        texts:  list of raw article bodies (will be vectorized with the
                SAME fitted vectorizer — do not refit)

    Returns:
        List of predicted framing labels.
    """
    X = result.vectorizer.transform(texts)
    preds = result.model.predict(X)
    return [IDX_TO_LABEL[p] for p in preds]


def predict_proba(result: BaselineResult, texts: list[str]) -> np.ndarray | None:
    """
    Return class probabilities if the model supports it.
    LinearSVC does NOT support predict_proba natively (no probabilistic
    output) — returns None in that case. Use decision_function as a
    confidence proxy if probabilities are required for SVM.
    """
    X = result.vectorizer.transform(texts)
    if hasattr(result.model, "predict_proba"):
        return result.model.predict_proba(X)
    return None


# ── Persistence ────────────────────────────────────────────────────────────────

def save_baseline(result: BaselineResult, path: Path) -> None:
    """Pickle a trained baseline (model + vectorizer) for later reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "name": result.name,
            "model": result.model,
            "vectorizer": result.vectorizer,
            "macro_f1": result.macro_f1,
            "accuracy": result.accuracy,
            "per_class_f1": result.per_class_f1,
        }, f)
    log.info(f"[baselines] Saved {result.name} → {path}")


def load_baseline(path: Path) -> BaselineResult:
    """Load a previously pickled baseline."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return BaselineResult(
        name=data["name"], model=data["model"], vectorizer=data["vectorizer"],
        macro_f1=data["macro_f1"], accuracy=data["accuracy"],
        per_class_f1=data["per_class_f1"],
    )
