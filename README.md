# 🛡️ Email Phishing Prediction

A full-stack machine learning application that predicts whether an email is **phishing** or **legitimate** based on 8 extracted text and link features.

---

## 📁 Project Structure

```
project1/
├── email_phishing_data.csv   # 524,846-row dataset
├── requirements.txt          # Python dependencies
├── train_model.py            # Training script → saves model/
├── model/
│   ├── phishing_model.pkl    # Trained RandomForestClassifier
│   └── scaler.pkl            # StandardScaler
├── app.py                    # FastAPI REST backend (port 8000)
├── frontend.py               # Streamlit UI (port 8501)
└── README.md
```

---

## 🧠 Dataset

| Property       | Value              |
|----------------|--------------------|
| Total rows     | 524,846            |
| Phishing (1)   | 6,949  (~1.3%)     |
| Legitimate (0) | 517,897 (~98.7%)   |
| Features       | 8 numeric          |

**Features:**

| Column                | Description                       |
|-----------------------|-----------------------------------|
| `num_words`           | Total word count                  |
| `num_unique_words`    | Distinct word count               |
| `num_stopwords`       | Stop-word count                   |
| `num_links`           | Hyperlinks embedded               |
| `num_unique_domains`  | Unique link domains               |
| `num_email_addresses` | Email addresses in body           |
| `num_spelling_errors` | Spelling mistakes                 |
| `num_urgent_keywords` | Words like "urgent", "act now"    |

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (one-time)

```bash
python train_model.py
```

This will:
- Load and split the dataset (80/20)
- Apply **SMOTE** to handle the severe class imbalance
- Train a **RandomForestClassifier** (150 trees)
- Print evaluation metrics (accuracy, ROC-AUC, classification report)
- Save `model/phishing_model.pkl` and `model/scaler.pkl`

### 3. Start the FastAPI backend

```bash
uvicorn app:app --reload --port 8000
```

Swagger UI available at: http://localhost:8000/docs

### 4. Launch the Streamlit frontend

```bash
streamlit run frontend.py
```

UI available at: http://localhost:8501

---

## 🔌 API Endpoints

| Method | Path              | Description                       |
|--------|-------------------|-----------------------------------|
| GET    | `/health`         | Liveness probe                    |
| POST   | `/predict`        | Single email prediction           |
| POST   | `/predict/batch`  | Batch prediction (list of emails) |

### Example `/predict` request

```json
{
  "num_words": 120,
  "num_unique_words": 80,
  "num_stopwords": 30,
  "num_links": 5,
  "num_unique_domains": 3,
  "num_email_addresses": 1,
  "num_spelling_errors": 4,
  "num_urgent_keywords": 2
}
```

### Example response

```json
{
  "label": 1,
  "verdict": "⚠️ Phishing",
  "confidence": 0.8933,
  "phishing_probability": 0.8933
}
```

---

## 🖥️ Frontend Pages

| Page                  | Description                                              |
|-----------------------|----------------------------------------------------------|
| 🔍 Single Prediction  | Enter features manually, get verdict + gauge chart       |
| 📊 EDA & Insights     | Dataset distributions, correlation heatmap, importances  |
| 📂 Batch Prediction   | Upload CSV, download results with predictions            |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────┐
│         Streamlit Frontend              │
│  • Single prediction form               │
│  • EDA charts (Plotly)                  │
│  • Batch CSV upload / download          │
└────────────────┬───────────────────────┘
                 │  HTTP (REST)
                 ▼
┌────────────────────────────────────────┐
│         FastAPI Backend (port 8000)     │
│  GET  /health                           │
│  POST /predict                          │
│  POST /predict/batch                    │
└────────────────┬───────────────────────┘
                 │  joblib.load()
                 ▼
┌────────────────────────────────────────┐
│    RandomForestClassifier + Scaler      │
│    Trained on 524K emails w/ SMOTE      │
└────────────────────────────────────────┘
```

---

## 🧪 Tech Stack

| Layer      | Technology                                   |
|------------|----------------------------------------------|
| ML         | scikit-learn, imbalanced-learn (SMOTE)       |
| Backend    | FastAPI, Uvicorn, Pydantic                   |
| Frontend   | Streamlit, Plotly                            |
| Data       | pandas, NumPy                                |
| Persistence| joblib                                       |
