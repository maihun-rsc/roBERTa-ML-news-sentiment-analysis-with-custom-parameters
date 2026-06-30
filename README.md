# Understanding How News Articles Shape Public Opinion

**A Comprehensive Machine Learning Pipeline for Media Framing and Sentiment Analysis**

**Department:** School of Computing Science and Artificial Intelligence, VIT Bhopal University
**Author:** Rananjay Singh Chauhan (23BAI10031)  
**Supervisor:** Dr. Manorama Chouhan (manoramachouhan@vitbhopal.ac.in)

---

## 📖 What We Are Actually Trying to Do

Standard sentiment analysis is built for product reviews and tweets — text that wears its feelings on its sleeve. News is different. A BBC dispatch doesn't say "I'm sad about this." It says "the minister *admitted*" instead of "the minister *said*." The emotion is in the verb choice, the adjective selection, the source quoted, and the entity framed. That's framing — and framing is what this system detects.

The system takes a news article (text or broadcast transcript), a named entity within it, and outputs one of four labels:

- **Supportive** — the entity is portrayed positively
- **Critical** — the entity is blamed, questioned, condemned
- **Neutral-Reporting** — wire-service register; factual, balanced
- **Alarmist** — crisis/urgency framing regardless of strict valence

We do this across a diverse selection of outlets (listed in `configs/outlets.json`).
Then we ask: does framing diverge by outlet? By topic? By entity type?  
**That's the research question. The code answers it empirically.**

---

## 🏗️ Architecture Overview

The pipeline is modular by design, with five independently executable stages, orchestrated by `main.py`:

```text
news_sentiment/
├── main.py                     ← orchestrator: runs all modules in sequence
├── configs/
│   ├── outlets.json            ← outlet names, RSS URLs, scrape strategies
│   ├── model_config.yaml       ← hyperparameters, paths, label maps
│   └── env_config.py           ← path handling and configuration schema
│
├── data_collection/            ← MODULE 1
│   ├── scraper.py              ← newspaper3k + retry logic
│   ├── rss_collector.py        ← RSS feed parser per outlet
│   ├── fallback_loader.py      ← CC-News / MIND / SemEval datasets
│   ├── deduplicator.py         ← SHA-256 content hashing
│   └── schema.py               ← Article dataclass + validators
│
├── preprocessing/              ← MODULE 2
│   ├── cleaner.py              ← HTML strip, unicode norm, boilerplate removal
│   ├── ner_pipeline.py         ← spaCy en_core_web_trf, entity extraction
│   ├── proximity_scorer.py     ← syntactic distance via dep parse tree
│   └── asr_cleaner.py          ← Whisper transcript noise removal
│
├── models/                     ← MODULE 3
│   ├── baselines.py            ← LR, SVM, Naive Bayes (sklearn)
│   ├── roberta_framing.py      ← RoBERTa fine-tuner (HuggingFace)
│   ├── entity_attention.py     ← Entity-aware attention layer
│   └── fusion.py               ← Late fusion: text + ASR transcript
│
├── evaluation/                 ← MODULE 4
│   ├── metrics.py              ← Macro-F1, confusion matrix, per-class
│   ├── statistical_tests.py    ← Mann-Whitney U, Kruskal-Wallis, ANOVA
│   └── kappa.py                ← Fleiss' Kappa, Cohen's Kappa
│
└── analysis/                   ← MODULE 5
    ├── cross_source.py         ← Outlet × Topic framing heatmaps
    └── entity_profiler.py      ← Per-entity framing profiles
```

---

## 🗃️ Module 1 — Data Collection

### What it does

1. Reads `configs/outlets.json` — 10 outlets, each with RSS URL(s) and a scrape strategy flag.
2. For each outlet: parse RSS → extract article URLs → scrape full text via `newspaper3k`.
3. On any failure (403, timeout, parse error, empty body): falls back to public datasets.
4. Deduplicates using SHA-256 hash of the article body.
5. Validates and writes to `data/raw/{outlet_name}.jsonl`.

### Fallback datasets

| Dataset       | Use case                          | Access method             |
|---------------|-----------------------------------|---------------------------|
| MIND          | Topic diversity, entity density   | `datasets` library (HF)   |
| CC-News       | International outlet diversity    | `datasets` library (HF)   |
| SemEval-2017  | Entity-level gold labels          | Manual download + loader  |

### Article schema (every record written to JSONL)

```python
@dataclass
class Article:
    article_id: str        # SHA-256 of body (first 16 chars)
    source: str            # outlet name
    title: str             # headline
    body: str              # full article text
    url: str               # canonical URL
    date: str              # ISO 8601
    topic: str             # auto-inferred from RSS category
    entities: list[str]    # empty at collection time; filled in Module 2
    label: str             # empty at collection time; filled in annotation
    transcript: str        # empty unless ASR — filled in Module 2
```

---

## ⚙️ Tech Stack — Every Component Explained

### Language and Runtime
**Python 3.10+** — 3.12 in the sandbox, 3.10 on Kaggle's default kernel.  
Why not 3.13? transformers and torch lag on new Python versions. 3.10–3.12 is the safe window for the entire stack.

### Environment Detection
The code detects at runtime whether it's running on Kaggle (checks `/kaggle/input`) or Antigravity IDE (checks for `ANTIGRAVITY_HOME` env var or falls back to local paths). This sets:
- Data paths (Kaggle: `/kaggle/working/`, local: `./data/`)
- Device (`cuda` if available, `mps` if Apple Silicon, else `cpu`)
- Batch sizes (Kaggle T4: 16, local CPU: 4)
- Logging verbosity

### Data Collection Layer
- **`newspaper3k==0.2.8`**: Downloads and parses article HTML, extracting the core text while ignoring boilerplate. 
- **`feedparser==6.0.12`**: Parses RSS feeds. This is used before newspaper3k to cleanly gather article URLs and summaries.
- **`requests==2.33.1`**: HTTP client with retry logic (exponential backoff) and rotating User-Agent headers to reduce 403 blocks.
- **`datasets` (HuggingFace)**: Streams fallback datasets like MIND and CC-News when direct web scraping fails or yields zero articles.

### Deduplication
- Uses **SHA-256 hashes** of the article body. Any article whose hash exists in the seen-set is skipped.

### Storage Format
- **JSONL (JSON Lines)**: Data is stored with one JSON object per line, ensuring robust portability and immediate readiness for Pandas and PyTorch.

### Preprocessing Layer (Module 2 preview)
- **`spaCy` (en_core_web_trf)**: Transformer-backed NER, identifying PERSON, ORG, GPE, EVENT, NORP entities.
- **`Whisper` (openai-whisper)**: ASR for broadcast transcripts.

### Modelling Layer (Module 3 preview)
- **`transformers`**: `roberta-base` as the primary encoder, fine-tuned with a 4-class framing head and entity-aware attention.
- **`torch`**: Training loop, optimizer (AdamW), scheduler (linear warmup + cosine decay), class-weighted cross-entropy loss.
- **`scikit-learn`**: Training loops, baseline classifiers, and TF-IDF vectorization.

### Evaluation Layer (Module 4 preview)
- **`scipy.stats`**: Mann-Whitney U, Kruskal-Wallis H-test for outlet divergence significance.
- **`sklearn.metrics`**: F1, precision, recall, confusion matrix.
- **`statsmodels`**: ANOVA, post-hoc Tukey HSD.
- **Custom Fleiss' Kappa**: written from scratch.

### Analysis Layer (Module 5 preview)
- **`seaborn` + `matplotlib`**: outlet × topic framing heatmaps, entity framing profiles, distribution plots.
- **`pandas`**: all aggregation and groupby operations.

---

## 🛠️ Coding Conventions

- **Type hints everywhere** — Python 3.10+ union syntax (`str | None`)
- **Dataclasses** for all data structures — no raw dicts passed between modules
- **Logging** via `logging` stdlib — not print statements. Level set by env.
- **Docstrings** — Google style, one per function
- **No global state** — config passed explicitly; nothing imported from `__main__`
- **Fail loudly** — exceptions are caught, logged, and re-raised with context. No silent swallowing.
- **Reproducibility** — all random seeds set via `utils.seed_everything(seed=42)`

---

## ❌ What This Is Not

- Not a real-time system. This is a batch research pipeline.
- Not a production scraper. We respect `robots.txt` and rate limits.
- Not claiming the model is objective truth. Framing detection is linguistic measurement, not fact-checking.

---

## 🏃 Sprint Plan

| Sprint | Module | Deliverable |
|--------|--------|-------------|
| 1 (now)| Data Collection | `data_collection/` — fully tested, dual-env |
| 2      | Preprocessing | `preprocessing/` — spaCy pipeline + ASR cleaner |
| 3      | Modelling | `models/` — baselines + RoBERTa fine-tuner |
| 4      | Evaluation | `metrics/` — all statistical tests |
| 5      | Analysis | `analysis/` — heatmaps + entity profiles |
| 6      | Integration | `main.py` + end-to-end test on 100 articles |

---

## 🚀 How to Run

1. **Install Dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute the Pipeline:**  
   ```bash
   python main.py
   ```

*(Note: Data is automatically saved into the `data/` directory, which is excluded from version control to protect data privacy and reduce repository bloat).*

---

## 📬 Contact & Contributions

For academic inquiries, peer reviews, or collaboration regarding the methodologies used in this study, please reach out to:  

**Rananjay Singh Chauhan**  
✉️ **Email:** 
- [rjchauhan.work@gmail.com](mailto:rjchauhan.work@gmail.com) |
- [rananjaychauhan93@gmail.com](mailto:rananjaychauhan93@gmail.com) |
- [rananjay.23bai10080@vitbhopal.ac.in](mailto:rananjay.23bai10080@vitbhopal.ac.in) |
🔗 **LinkedIn:** [linkedin.com/in/maihun-rsc](https://www.linkedin.com/in/maihun-rsc/)  
💻 **GitHub:** [github.com/maihun-rsc](https://github.com/maihun-rsc)  

