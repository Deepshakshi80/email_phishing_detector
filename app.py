"""
app.py
──────
FastAPI backend for the Email Phishing Prediction service.

Endpoints:
    GET  /health          → liveness probe
    POST /predict         → single email feature prediction
    POST /predict/batch   → batch prediction (list of emails)

Run:
    uvicorn app:app --reload --port 8000
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("model", "phishing_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

FEATURES = [
    "num_words",
    "num_unique_words",
    "num_stopwords",
    "num_links",
    "num_unique_domains",
    "num_email_addresses",
    "num_spelling_errors",
    "num_urgent_keywords",
]

# ── Load artifacts at startup ─────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError(
        "Model artifacts not found. Run `python train_model.py` first."
    )

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Email Phishing Prediction API",
    description="Predicts whether an email is phishing or legitimate based on extracted features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class EmailFeatures(BaseModel):
    num_words:           int   = Field(..., ge=0, description="Total word count")
    num_unique_words:    int   = Field(..., ge=0, description="Unique word count")
    num_stopwords:       int   = Field(..., ge=0, description="Stop-word count")
    num_links:           int   = Field(..., ge=0, description="Number of hyperlinks")
    num_unique_domains:  int   = Field(..., ge=0, description="Unique domains in links")
    num_email_addresses: int   = Field(..., ge=0, description="Email addresses mentioned")
    num_spelling_errors: int   = Field(..., ge=0, description="Spelling error count")
    num_urgent_keywords: int   = Field(..., ge=0, description="Urgent-language keyword count")

    model_config = {
        "json_schema_extra": {
            "example": {
                "num_words": 120,
                "num_unique_words": 80,
                "num_stopwords": 30,
                "num_links": 5,
                "num_unique_domains": 3,
                "num_email_addresses": 1,
                "num_spelling_errors": 4,
                "num_urgent_keywords": 2,
            }
        }
    }


class PredictionResult(BaseModel):
    label:       int   = Field(..., description="0 = Legitimate, 1 = Phishing")
    verdict:     str   = Field(..., description="Human-readable verdict")
    confidence:  float = Field(..., description="Model confidence (0–1) for predicted class")
    phishing_probability: float = Field(..., description="Probability the email is phishing")


class BatchRequest(BaseModel):
    emails: List[EmailFeatures]


class BatchResult(BaseModel):
    results: List[PredictionResult]
    total:   int
    phishing_count: int
    legitimate_count: int


# ── Helper ────────────────────────────────────────────────────────────────────
def _predict_one(features: EmailFeatures) -> PredictionResult:
    X = np.array([[getattr(features, f) for f in FEATURES]])
    X_scaled = scaler.transform(X)
    label    = int(model.predict(X_scaled)[0])
    proba    = model.predict_proba(X_scaled)[0]
    phishing_prob = float(proba[1])
    confidence    = float(proba[label])
    verdict = "⚠️ Phishing" if label == 1 else "✅ Legitimate"
    return PredictionResult(
        label=label,
        verdict=verdict,
        confidence=round(confidence, 4),
        phishing_probability=round(phishing_prob, 4),
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Utility"])
def health():
    return {"status": "ok", "model": "RandomForestClassifier", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(features: EmailFeatures):
    """
    Predict whether a single email is phishing or legitimate.
    """
    try:
        return _predict_one(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResult, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    """
    Predict phishing status for a batch of emails.
    """
    if not request.emails:
        raise HTTPException(status_code=400, detail="emails list is empty")
    try:
        results = [_predict_one(e) for e in request.emails]
        phishing_count   = sum(r.label for r in results)
        legitimate_count = len(results) - phishing_count
        return BatchResult(
            results=results,
            total=len(results),
            phishing_count=phishing_count,
            legitimate_count=legitimate_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
