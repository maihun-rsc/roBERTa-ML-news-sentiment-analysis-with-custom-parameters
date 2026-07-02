# Module 3: Machine Learning Models

This module contains the core analytical and classification logic. It ingests the preprocessed, structurally quantified text (along with its calculated syntactic entity-proximity scores) and outputs the predicted rhetorical framing label for the target entity.

## 📁 File Structure & Responsibilities

| File | Purpose |
|------|---------|
| `baselines.py` | Implements traditional Machine Learning baselines for comparison, utilizing `scikit-learn`. Includes standard models like Logistic Regression (LR), Support Vector Machines (SVM), and Naive Bayes, along with TF-IDF vectorization. |
| `roberta_framing.py` | The main powerhouse. A fine-tuning script utilizing HuggingFace's `transformers` library to adapt the `roberta-base` encoder. It uses a custom 4-class framing classification head and a class-weighted cross-entropy loss function optimized via AdamW. |
| `entity_attention.py` | A custom architectural layer injected between the RoBERTa encoder and the classification head. It forces the model to attend specifically to the target entity and its immediate syntactic context, preventing the model from lazily assigning the general document sentiment to the entity. |
| `fusion.py` | Implements "Late Fusion" mechanics. It allows the model to intelligently merge predictions/features from both the written text article and the ASR broadcast transcript when analyzing multimodal outlets (like BBC or CNN broadcast clips). |

## ⚙️ How it Fits into the Pipeline

1. **Input:** Receives clean, preprocessed data and target entities from Module 2.
2. **Processing:** Passes the data through the entity-aware RoBERTa model to classify the framing into one of four categories: *Supportive, Critical, Neutral-Reporting, or Alarmist*.
3. **Output:** Predicted labels are appended to the dataset, ready for statistical evaluation and analysis in Modules 4 and 5.
