"""
main.py
───────
Orchestrator for the News Sentiment Analysis pipeline.
Run this file directly to execute any or all pipeline stages.

Usage:
    # Full pipeline from scratch
    python main.py

    # Only specific stages
    python main.py --stage collect
    python main.py --stage preprocess
    python main.py --stage train
    python main.py --stage evaluate
    python main.py --stage analyse

    # Dry-run: validate config and environment, then exit
    python main.py --dry-run

    # Collect from specific outlets only
    python main.py --stage collect --outlets bbc cnn fox

Environment:
    The script auto-detects Antigravity IDE vs Kaggle vs local CPU.
    No environment variables required — detection is fully automatic.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# Force UTF-8 for Windows terminal to avoid UnicodeEncodeError on box-drawing characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Suppress annoying C++ TensorFlow warnings (these bypass python logging)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning" # allow them so they can be captured to file

# ── Make project root importable regardless of CWD ──────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Internal imports ─────────────────────────────────────────────────────────
from configs.env_config import get_config, Config
from data_collection import run_collection, read_all_raw

log = logging.getLogger(__name__)


# ── Argument parsing ─────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="News Sentiment Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--stage",
        choices=["collect", "preprocess", "annotate", "train", "evaluate", "analyse", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    p.add_argument(
        "--outlets",
        nargs="*",
        help="Restrict collection to specific outlet slugs (e.g. bbc cnn)",
    )
    p.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Override max articles per outlet",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and environment, then exit without running",
    )
    return p


# ── Utility helpers ──────────────────────────────────────────────────────────

def _load_outlets_cfg(cfg: Config, filter_names: list[str] | None = None) -> tuple[list[dict], list[str]]:
    """Load outlets.json and return (outlet_list, user_agents)."""
    outlets_path = cfg.config_dir / "outlets.json"
    if not outlets_path.exists():
        # Try relative to script location
        outlets_path = _ROOT / "configs" / "outlets.json"

    if not outlets_path.exists():
        raise FileNotFoundError(
            f"outlets.json not found at {outlets_path}. "
            "Make sure configs/ directory is present."
        )

    with open(outlets_path, encoding="utf-8") as f:
        raw = json.load(f)

    outlets = raw["outlets"]
    user_agents = raw.get("user_agents", [
        "Mozilla/5.0 (compatible; NewsSentimentBot/1.0)"
    ])

    if filter_names:
        outlets = [o for o in outlets if o["name"] in filter_names]
        if not outlets:
            raise ValueError(
                f"None of the specified outlets {filter_names} found in outlets.json. "
                f"Available: {[o['name'] for o in raw['outlets']]}"
            )
        log.info(f"Filtering to outlets: {[o['name'] for o in outlets]}")

    return outlets, user_agents


def _seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    log.debug(f"Random seed set to {seed}")


def _print_banner(cfg: Config) -> None:
    print("\n" + "=" * 62)
    print("  News Sentiment Analysis Pipeline")
    print("  VIT Bhopal University - CSA4029")
    print("=" * 62)
    print(f"  Environment : {cfg.env}")
    print(f"  Device      : {cfg.device}")
    print(f"  Data (raw)  : {cfg.data_raw}")
    print(f"  Data (proc) : {cfg.data_processed}")
    print("=" * 62 + "\n")


# ── Stage: Data Collection ───────────────────────────────────────────────────

def stage_collect(
    cfg: Config,
    outlets: list[dict],
    user_agents: list[str],
    seed: int,
) -> dict[str, int]:
    """
    Run Module 1 — Data Collection.

    Scrapes all configured outlets, applies fallback datasets where needed,
    deduplicates, and writes JSONL files to cfg.data_raw.

    Returns:
        Dict mapping outlet slug → number of articles written.
    """
    log.info("━━ STAGE 1: DATA COLLECTION ━━")
    t0 = time.perf_counter()

    counts = run_collection(
        cfg=cfg,
        outlets_cfg=outlets,
        user_agents=user_agents,
        seed=seed,
    )

    elapsed = time.perf_counter() - t0
    total   = sum(counts.values())

    log.info(f"\n[collect] ✓ Complete in {elapsed:.1f}s")
    log.info(f"[collect] Total articles collected: {total}")
    for outlet, count in counts.items():
        log.info(f"  {outlet:12}: {count:5} articles")

    return counts


# ── Stage stubs (Modules 2-5 — implemented in later sprints) ────────────────

def stage_preprocess(cfg: Config) -> None:
    """Module 2 — Preprocessing: NER, tokenisation, entity spans."""
    log.info("━━ STAGE 2: PREPROCESSING ━━")
    from preprocessing import run_preprocessing
    articles = read_all_raw(cfg.data_raw)
    log.info(f"[preprocess] Loaded {len(articles)} raw articles")
    run_preprocessing(articles, cfg)


def stage_annotate(cfg: Config) -> None:
    """Module 2.5 — Auto-Annotation: Zero-shot dataset generation."""
    log.info("━━ STAGE 2.5: AUTO-ANNOTATION ━━")
    from data_collection.auto_annotate import annotate_dataset
    processed_path = cfg.data_processed / "processed_articles.jsonl"
    annotated_path = cfg.data_processed / "annotated_articles.jsonl"
    annotate_dataset(processed_path, annotated_path, limit=cfg.max_articles_per_outlet or 0)


def stage_train(cfg: Config) -> None:
    """Module 3 — Model training: baselines + RoBERTa fine-tuning."""
    log.info("━━ STAGE 3: MODEL TRAINING ━━")
    from models import run_training
    run_training(cfg)


def stage_evaluate(cfg: Config) -> None:
    """Module 4 — Evaluation: metrics, statistical tests."""
    log.info("━━ STAGE 4: EVALUATION ━━")
    from evaluation import run_evaluation
    run_evaluation(cfg)


def stage_analyse(cfg: Config) -> None:
    """Module 5 — Analysis: cross-source heatmaps, entity profiles."""
    log.info("━━ STAGE 5: ANALYSIS ━━")
    from analysis import run_analysis
    run_analysis(cfg)


# ── Dry-run ──────────────────────────────────────────────────────────────────

def dry_run(cfg: Config, outlets: list[dict]) -> None:
    """Validate environment and config without running anything."""
    print("\n[dry-run] Environment check")
    print(f"  Python       : {sys.version.split()[0]}")
    print(f"  Environment  : {cfg.env}")
    print(f"  Device       : {cfg.device}")
    print(f"  Raw data dir : {cfg.data_raw}  (exists: {cfg.data_raw.exists()})")

    print(f"\n[dry-run] Outlets configured: {len(outlets)}")
    for o in outlets:
        feeds = len(o.get("rss_feeds", []))
        print(f"  {o['name']:12} | {feeds} RSS feeds | fallback: {o.get('fallback','?')}")

    print("\n[dry-run] Dependency check")
    deps = [
        ("requests",    "requests"),
        ("feedparser",  "feedparser"),
        ("newspaper3k", "newspaper"),
        ("datasets",    "datasets"),
        ("spacy",       "spacy"),
        ("torch",       "torch"),
        ("transformers","transformers"),
        ("sklearn",     "sklearn"),
        ("scipy",       "scipy"),
        ("pandas",      "pandas"),
        ("numpy",       "numpy"),
    ]
    all_ok = True
    for label, mod in deps:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  [OK] {label:20} {ver}")
        except ImportError:
            print(f"  [FAIL] {label:20} NOT INSTALLED")
            all_ok = False

    if all_ok:
        print("\n[dry-run] [OK] All dependencies present. Ready to run.")
    else:
        print("\n[dry-run] [FAIL] Some dependencies missing. Run: pip install -r requirements.txt")

    sys.exit(0 if all_ok else 1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = _build_parser().parse_args()
    cfg    = get_config(log_level=args.log_level)

    if args.max_articles is not None:
        cfg.max_articles_per_outlet = args.max_articles

    _seed_everything(args.seed)
    _print_banner(cfg)

    outlets, user_agents = _load_outlets_cfg(cfg, filter_names=args.outlets)

    if args.dry_run:
        dry_run(cfg, outlets)
        return  # unreachable after sys.exit, but explicit is good

    stage  = args.stage
    seed   = args.seed

    if stage in ("collect", "all"):
        stage_collect(cfg, outlets, user_agents, seed)

    if stage in ("preprocess", "all"):
        stage_preprocess(cfg)

    if stage in ("annotate", "all"):
        stage_annotate(cfg)

    if stage in ("train", "all"):
        stage_train(cfg)

    if stage in ("evaluate", "all"):
        stage_evaluate(cfg)

    if stage in ("analyse", "all"):
        stage_analyse(cfg)

    log.info("\n[main] Pipeline finished.")


if __name__ == "__main__":
    main()
