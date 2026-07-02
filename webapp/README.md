# Web Dashboard

A standalone feature of the project, providing a beautiful, user-friendly graphical interface to interact directly with the underlying Zero-Shot inference engine.

## 📁 File Structure & Responsibilities

| File / Folder | Purpose |
|------|---------|
| `server.py` | A `FastAPI` backend that routes user requests, handles article scraping from raw URLs, processes pasted text, triggers the Zero-Shot inference, and serves the UI. |
| `static/` | Contains the CSS and JavaScript necessary to render the premium, dynamic frontend UI. |

## 🚀 How to Run the Dashboard

```bash
python -m uvicorn webapp.server:app --reload
```
Navigate to `http://localhost:8000` in your web browser. You can input any news URL or raw text, and the dashboard will automatically detect the entities and predict their framing in real-time.
