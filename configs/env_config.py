"""
env_config.py
─────────────
Detects runtime environment (Antigravity IDE / Kaggle / generic local)
and returns a unified Config object used by every module.

No global state. Import get_config() and pass it down explicitly.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


# ── Environment fingerprint ──────────────────────────────────────────────────

def _detect_env() -> str:
    """Return 'kaggle' | 'antigravity' | 'local'."""
    if Path("/kaggle/input").exists():
        return "kaggle"
    if os.getenv("ANTIGRAVITY_HOME") or os.getenv("ANTIGRAVITY_WORKSPACE"):
        return "antigravity"
    return "local"


def _detect_device() -> str:
    """Return 'cuda' | 'mps' | 'cpu' — best available."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            log.info(f"GPU detected: {name}")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            log.info("Apple Silicon MPS detected")
            return "mps"
    except ImportError:
        pass
    log.info("No GPU found — running on CPU")
    return "cpu"


# ── Config dataclass ─────────────────────────────────────────────────────────

@dataclass
class Config:
    # Environment
    env: str = "local"
    device: str = "cpu"

    # Paths — resolved at init
    project_root: Path = field(default_factory=Path.cwd)
    data_raw: Path      = field(init=False)
    data_processed: Path = field(init=False)
    data_fallback: Path  = field(init=False)
    log_dir: Path        = field(init=False)
    config_dir: Path     = field(init=False)

    # Scraping
    request_timeout: int  = 15          # seconds per request
    max_retries: int       = 3
    retry_backoff: float   = 2.0        # exponential backoff base
    rate_limit_delay: float = 0.8       # seconds between requests per outlet
    max_articles_per_outlet: int = 500
    min_body_chars: int    = 150        # skip stubs shorter than this

    # Fallback datasets
    use_mind: bool     = True
    use_ccnews: bool   = True
    use_semeval: bool  = True
    mind_split: str    = "train"        # 'train' | 'dev'
    fallback_max: int  = 1000           # max articles from fallback per outlet

    # Compute
    batch_size_gpu: int = 16
    batch_size_cpu: int = 4
    num_workers: int    = 2

    # Logging
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        # Kaggle paths differ from local
        if self.env == "kaggle":
            base = Path("/kaggle/working/news_sentiment")
        elif self.env == "antigravity":
            ag_home = os.getenv("ANTIGRAVITY_WORKSPACE", str(Path.cwd()))
            base = Path(ag_home) / "news_sentiment"
        else:
            base = self.project_root

        self.data_raw       = base / "data" / "raw"
        self.data_processed = base / "data" / "processed"
        self.data_fallback  = base / "data" / "fallback"
        self.log_dir        = base / "logs"
        self.config_dir     = base / "configs"

        # Create dirs if they don't exist
        for d in [self.data_raw, self.data_processed, self.data_fallback, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def batch_size(self) -> int:
        return self.batch_size_gpu if self.device == "cuda" else self.batch_size_cpu


def get_config(log_level: str = "INFO") -> Config:
    """
    Entry point — call once at the top of main.py or any runner.

    Returns:
        Config: fully resolved configuration object.
    """
    env    = _detect_env()
    device = _detect_device()

    cfg = Config(env=env, device=device, log_level=log_level)

    # Set up advanced logging (Errors/Warnings to file, INFO to console)
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    import sys
    import warnings
    
    # 1. Console Handler (Filter out WARNINGs)
    class NoWarningFilter(logging.Filter):
        def filter(self, record):
            return record.levelno != logging.WARNING
            
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(NoWarningFilter())
    logger.addHandler(console_handler)
    
    # 2. File Handler (Only WARNING and ERROR)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(cfg.log_dir / "warnings_and_errors.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.WARNING)
    logger.addHandler(file_handler)
    
    # 3. Capture Python warnings into logging system
    logging.captureWarnings(True)

    log.info(f"Environment : {env}")
    log.info(f"Device      : {device}")
    log.info(f"Data (raw)  : {cfg.data_raw}")
    log.info(f"Warnings logged to: {cfg.log_dir / 'warnings_and_errors.log'}")

    return cfg
