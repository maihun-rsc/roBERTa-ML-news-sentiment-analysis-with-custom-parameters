# Data Collection (`data_collection/`)

The Data Collection module is the ingestion engine for the project. Its primary responsibility is to gather raw, unstructured news articles from various media outlets across the globe and normalize them into a structured, consistent dataset ready for text preprocessing and machine learning analysis.

## Role and Importance
Robust machine learning relies entirely on the quality and volume of its training data. This module ensures that our sentiment and framing classifiers are fed high-quality, real-world data directly from the source. By handling network timeouts, bot-blocking measures, pagination, and data schema enforcement, this module guarantees that the downstream pipeline always receives clean, structured JSONL files.

## Files and Workflow

### 1. `rss_collector.py`
- **Purpose:** Discovers and parses recent article URLs from the configured RSS feeds.
- **How it works:** It utilizes the `feedparser` library to read XML feeds defined in `outlets.json`. It extracts the article links, publication dates, and titles, returning a list of targets for the scraper.

### 2. `scraper.py`
- **Purpose:** Downloads the full HTML body of the articles and extracts the core textual content.
- **How it works:** Using `requests` and `BeautifulSoup4` (or `newspaper3k` concepts), it navigates to the URLs identified by the `rss_collector`. It extracts the primary article text while stripping out ads, navigation bars, and boilerplate HTML. It incorporates polite scraping practices (rate limiting, user-agent rotation) to prevent IP bans.

### 3. `fallback_loader.py`
- **Purpose:** Provides supplementary data when live scraping fails or yields insufficient articles.
- **How it works:** If an outlet completely blocks our scraper or experiences an outage (yielding 0 articles), this script uses the HuggingFace `datasets` library to stream backup articles from established public datasets (such as CC-News, MIND, or SemEval). This ensures our pipeline never halts due to unexpected network errors.

### 4. `deduplicator.py`
- **Purpose:** Prevents duplicate articles from polluting the dataset across multiple runs.
- **How it works:** It generates unique hashes for each article based on its URL and content. These hashes are checked against a persistent local store (`seen_ids.txt`). Any duplicates are dropped before writing to disk.

### 5. `schema.py`
- **Purpose:** Enforces a rigid data structure for all collected articles.
- **How it works:** Defines the `Article` Pydantic model, ensuring every piece of data has a `title`, `body`, `source`, `url`, `date`, `topic`, and `region`. It also contains heuristics to automatically infer an article's topic based on keywords.

### 6. `writer.py`
- **Purpose:** The orchestrator of the data collection phase. 
- **How it works:** Contains `run_collection()`, which acts as the main entry point for the module. It sequentially calls the collector, scraper, fallback loader, and deduplicator. Finally, it writes the validated `Article` objects into line-delimited JSON (`.jsonl`) files in the raw data directory, ready for the preprocessing stage.
