# Module 5: Analysis & Visualization

This is the final stage of the pipeline. While Module 4 handles the raw mathematics of model evaluation, Module 5 handles the interpretation and visualization of the results, specifically aimed at answering the core research questions regarding media framing bias.

## 📁 File Structure & Responsibilities

| File | Purpose |
|------|---------|
| `cross_source.py` | Generates Outlet × Topic framing heatmaps using `seaborn` and `matplotlib`. This script visualizes exactly how different international outlets (like BBC vs. RT vs. Firstpost) diverge when reporting on identical topics (like Politics or Conflict). |
| `entity_profiler.py` | Generates framing profiles for specific named entities. It aggregates the data to show whether a specific person or organization is consistently framed favorably (Supportive) or negatively (Critical/Alarmist) across the global media spectrum. |

## ⚙️ How it Fits into the Pipeline
It ingests the final classified dataset and outputs high-quality, IEEE-standard graphs, heatmaps, and distribution plots into the `data/` and `logs/` directories, serving as the visual evidence for the research paper.
