# Configurations (`configs/`)

This directory serves as the centralized configuration hub for the entire "Understanding How News Articles Shape Public Opinion" pipeline. It manages everything from environment variables and hyperparameters to the specific media outlets targeted by the data collection system.

## Role and Importance
By isolating configurations from the core logic, this module ensures that the pipeline remains **modular, scalable, and secure**. Hardcoding values within scripts can lead to brittle code; extracting them here allows researchers to quickly adjust scraping thresholds, alter the targeted news outlets, or update model hyper-parameters without touching the underlying Python logic.

## Files and Workflow

### 1. `env_config.py`
- **Purpose:** Manages environment variables and global pipeline settings (e.g., directory paths, rate limits, request timeouts, and maximum article counts).
- **How it works:** It uses `pydantic-settings` to define a strongly-typed `Config` schema. It reads values from a `.env` file (if present) or falls back to sensible defaults. This ensures that the pipeline always has access to validated, type-safe configuration parameters throughout its execution.

### 2. `outlets.json`
- **Purpose:** Defines the complete list of media outlets to be tracked, scraped, and analyzed.
- **How it works:** Contains an array of JSON objects detailing each outlet's metadata (e.g., name, display name, region, RSS feeds, and fallback strategies). It also stores a pool of user agents to rotate during scraping (to avoid rate limits) and keyword mappings for topic inference. 
- **Recent Updates:** We specifically curate a balanced list of outlets across the political spectrum and global regions (e.g., BBC, CNN, The Hindu, The Indian Express, The Wire) to ensure the dataset prevents model bias.

### 3. `model_config.yaml`
- **Purpose:** Stores the hyper-parameters and architecture settings for the Machine Learning models used in the later sentiment and framing analysis stages.
- **How it works:** Organized into sections for data splits (train/test ratios), text feature extraction (TF-IDF max features, n-grams), and specific model parameters (Random Forest estimators, SVM kernels, AdaBoost learning rates). This file is parsed by the training scripts to initialize the classifiers uniformly.
