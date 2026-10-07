"""
train_model.py
──────────────
Trains a RandomForestClassifier on the email phishing dataset,
handles class imbalance with SMOTE, evaluates the model, and
saves the trained model + scaler to the model/ directory.

Run once:
    python train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
from imblearn.over_sampling import SMOTE

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH  = "email_phishing_data.csv"
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

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
TARGET = "label"

RANDOM_STATE = 42


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    print(f"[1/5] Loading dataset from '{path}' …")
    df = pd.read_csv(path)
    print(f"      Rows: {len(df):,}  |  Phishing: {df[TARGET].sum():,}  |  Legit: {(df[TARGET]==0).sum():,}")
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def preprocess(X_train, X_test):
    print("[2/5] Scaling features …")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def resample(X_train, y_train):
    print("[3/5] Applying SMOTE to balance classes …")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"      After SMOTE — Phishing: {y_res.sum():,}  |  Legit: {(y_res==0).sum():,}")
    return X_res, y_res


def train(X_train, y_train):
    print("[4/5] Training RandomForestClassifier …")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    print("[5/5] Evaluating …")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    importances = model.feature_importances_
    print("\n  Feature Importances:")
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        print(f"    {feat:<25} {imp:.4f}")


def save_artifacts(model, scaler):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n[OK] Model  saved -> {MODEL_PATH}")
    print(f"[OK] Scaler saved -> {SCALER_PATH}")


def main():
    print("=" * 55)
    print("  Email Phishing Prediction — Model Training")
    print("=" * 55)

    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test)
    X_res, y_res = resample(X_train_scaled, y_train)
    model = train(X_res, y_res)
    evaluate(model, X_test_scaled, y_test)
    save_artifacts(model, scaler)


if __name__ == "__main__":
    main()
