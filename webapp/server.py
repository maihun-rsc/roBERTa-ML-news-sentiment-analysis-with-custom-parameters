"""
server.py
─────────
FastAPI backend for the News Sentiment Analysis live dashboard.
Serves the custom HTML/CSS/JS frontend and provides an API endpoint 
for Zero-Shot entity-centric framing detection.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging
import json
from transformers import pipeline
from collections import Counter
from pathlib import Path
import os

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("webapp")

app = FastAPI(title="News Sentiment API")

# Mount the static directory
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Create static dir if it doesn't exist
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Load model globally
import torch
import torch.nn.functional as F

CUSTOM_MODEL_LOADED = False
custom_model = None
custom_tokenizer = None
classifier = None

try:
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    custom_model_dir = BASE_DIR.parent / "data" / "models" / "roberta_model"
    
    if custom_model_dir.exists():
        import sys
        if str(BASE_DIR.parent) not in sys.path:
            sys.path.insert(0, str(BASE_DIR.parent))
            
        from models.roberta_framing import load_model, IDX_TO_LABEL
        log.info(f"Loading Custom Entity-Aware RoBERTa from {custom_model_dir}...")
        custom_model, custom_tokenizer = load_model(custom_model_dir, device=device_str)
        CUSTOM_MODEL_LOADED = True
    else:
        log.warning("No custom model found in data/models/roberta_model. Falling back to Zero-Shot BART.")
        device_int = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device_int)
except Exception as e:
    log.warning(f"Could not load custom model ({type(e).__name__}: {e}). Falling back to Zero-Shot BART.")
    try:
        device_int = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device_int)
    except Exception as e2:
        log.error(f"Failed to load fallback model: {e2}")



class AnalysisRequest(BaseModel):
    url: str | None = None
    raw_text: str | None = None
    entity: str | None = None
    outlet: str | None = None


@app.get("/")
async def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/analyze")
async def analyze(req: AnalysisRequest):
    if not CUSTOM_MODEL_LOADED and not classifier:
        raise HTTPException(status_code=503, detail=f"Model failed to load. CUSTOM={CUSTOM_MODEL_LOADED}")
    
    if not req.url and not req.raw_text:
        raise HTTPException(status_code=400, detail="Must provide either url or raw_text.")
        
    text = ""
    title = "Raw Text Input"
    
    if req.raw_text:
        text = req.raw_text
        if req.outlet:
            title = f"{req.outlet} Article"
            
        # Log to database for data harvesting
        try:
            db_path = BASE_DIR.parent / "data" / "user_queries.jsonl"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(db_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"outlet": req.outlet, "text": text, "entity": req.entity}) + "\n")
        except Exception as e:
            log.warning(f"Failed to log user query: {e}")
    else:
        try:
            import newspaper
            from newspaper import Config
            
            # Spoof a real browser to bypass basic 403 Forbidden bot-blockers
            config = Config()
            config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            config.request_timeout = 10
            
            article = newspaper.Article(req.url, config=config)
            article.download()
            article.parse()
            text = article.text
            title = article.title
            if not text:
                raise HTTPException(status_code=400, detail="Failed to extract text from URL.")
        except ImportError:
            raise HTTPException(status_code=500, detail="newspaper3k is not installed.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to scrape article: {str(e)}")
        
    # Auto-detect entity if not provided
    entity_clean = req.entity.strip() if req.entity else ""
    
    if not entity_clean:
        doc = nlp(text[:5000]) # limit to first 5000 chars for speed
        entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]
        if not entities:
            raise HTTPException(status_code=404, detail="No entity provided, and Auto-NER failed to detect any prominent entities in the text.")
        entity_clean = Counter(entities).most_common(1)[0][0]
        
    if entity_clean.lower() not in text.lower():
        # Sometimes newspaper3k only scrapes a short cookie notice if the site blocks bots.
        # Providing the length helps diagnose if it's a scraping issue.
        raise HTTPException(
            status_code=404, 
            detail=f"The entity '{entity_clean}' was not found. (Scraped {len(text)} chars from URL). The site may be blocking our scraper, or the entity is misspelled."
        )
        
    # Truncate text for the model (BART has a 1024 token limit)
    idx = text.lower().find(entity_clean.lower())
    start = max(0, idx - 1000)
    end = min(len(text), idx + 1500)
    context = text[start:end]
    if start > 0: context = "..." + context
    if end < len(text): context = context + "..."
        
    try:
        if CUSTOM_MODEL_LOADED and custom_model and custom_tokenizer:
            engine_name = "Custom Entity-Aware RoBERTa (Fine-Tuned)"
            # Tokenize the context
            inputs = custom_tokenizer(
                context, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            input_ids = inputs["input_ids"].to(device_str)
            attention_mask = inputs["attention_mask"].to(device_str)
            
            # Syntactic proximity weights: simplified to global attention [1.0] for now
            # as fallback to avoid expensive dependency parsing in realtime.
            seq_len = input_ids.shape[1]
            proximity_scores = torch.ones((1, seq_len), dtype=torch.float).to(device_str)
            
            with torch.no_grad():
                out = custom_model(input_ids, attention_mask, proximity_scores)
                logits = out["logits"]
                probs = F.softmax(logits, dim=-1).squeeze().tolist()
                
            scores = {IDX_TO_LABEL[i]: prob for i, prob in enumerate(probs)}
            
        else:
            engine_name = "facebook/bart-large-mnli (Zero-Shot Fallback)"
            candidate_labels = [
                "supportive or endorsing",
                "critical, blaming, or questioning",
                "neutral, factual reporting",
                "alarmist, crisis, or threatening"
            ]
            
            hypothesis_template = f"In this text, the entity '{entity_clean}' is portrayed in a way that is {{}}."
            
            result = classifier(
                context,
                candidate_labels,
                hypothesis_template=hypothesis_template,
                multi_label=False
            )
            label_map = {
                "supportive or endorsing": "Supportive",
                "critical, blaming, or questioning": "Critical",
                "neutral, factual reporting": "Neutral-Reporting",
                "alarmist, crisis, or threatening": "Alarmist"
            }
            scores = {label_map[label]: float(score) for label, score in zip(result["labels"], result["scores"])} # type: ignore
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
    
    return {
        "title": title,
        "auto_entity": entity_clean if not req.entity else None,
        "context": context,
        "scores": scores,
        "engine": engine_name
    }
