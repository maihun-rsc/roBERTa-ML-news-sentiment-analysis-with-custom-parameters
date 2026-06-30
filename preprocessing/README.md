# Module 2: Preprocessing & Feature Extraction

This module transforms raw, unstructured scraped articles and noisy ASR broadcast transcripts into clean, structured data ready for machine learning models. 

Text normalization is inherently messy. This stage normalizes the data by removing HTML noise, handling punctuation, extracting named entities, and calculating the syntactic distance between tokens and target entities.

## 📁 File Structure & Responsibilities

| File | Purpose |
|------|---------|
| `cleaner.py` | Performs foundational cleaning: HTML stripping, unicode normalization, boilerplate removal, and whitespace standardization. |
| `ner_pipeline.py` | Uses `spaCy` (transformer-backed `en_core_web_trf`) to perform Named Entity Recognition (NER). It identifies and extracts entities like PERSON, ORG, GPE, EVENT, and NORP. |
| `proximity_scorer.py` | Analyzes the dependency parse tree to compute the "syntactic distance" between evaluative words (adjectives/verbs) and the target entity. This is crucial for entity-centric framing, ensuring the model doesn't falsely attribute document-level sentiment to the specific entity. |
| `asr_cleaner.py` | Cleans and normalizes noisy Automated Speech Recognition (ASR) transcripts generated via OpenAI's `Whisper` model, handling disfluencies and speaker artifacts. |

## ⚙️ How it Fits into the Pipeline

1. **Input:** Receives raw JSONL files from the Data Collection module (`data/raw/`).
2. **Processing:** Cleans the text, runs it through the spaCy transformer pipeline to identify target entities, and calculates proximity scores.
3. **Output:** Saves the processed articles to `data/processed/`, which are then fed into the Machine Learning models in Module 3.
