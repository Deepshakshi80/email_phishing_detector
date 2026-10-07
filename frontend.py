"""
frontend.py — Streamlit Cloud entry point
─────────────────────────────────────────
Self-contained: trains the model at startup (cached), no FastAPI needed.
Deploy directly to Streamlit Community Cloud.

    streamlit run frontend.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH = "email_phishing_data.csv"

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

FEATURE_LABELS = {
    "num_words":           "Total Words",
    "num_unique_words":    "Unique Words",
    "num_stopwords":       "Stop Words",
    "num_links":           "Hyperlinks",
    "num_unique_domains":  "Unique Domains",
    "num_email_addresses": "Email Addresses",
    "num_spelling_errors": "Spelling Errors",
    "num_urgent_keywords": "Urgent Keywords",
}

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title  { font-size:2.2rem; font-weight:700; color:#1f2328; }
    .sub-title   { font-size:1rem; color:#57606a; margin-bottom:1.5rem; }
    .verdict-phish { background:#fff1f0; border-left:5px solid #e53e3e;
                     padding:1rem; border-radius:6px; margin:0.5rem 0; }
    .verdict-legit { background:#f0fff4; border-left:5px solid #38a169;
                     padding:1rem; border-radius:6px; margin:0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Data & Model (cached — trained once per session) ──────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Training model on first launch — please wait (~2 min)…")
def get_model():
    """Train RandomForest + SMOTE, return (model, scaler, metrics)."""
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_s, y_train)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_res, y_res)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc":  round(roc_auc_score(y_test, y_prob), 4),
    }
    return model, scaler, metrics


def predict(payload: dict, model, scaler) -> dict:
    X = pd.DataFrame([payload], columns=FEATURES)
    X_s   = scaler.transform(X)
    label = int(model.predict(X_s)[0])
    proba = model.predict_proba(X_s)[0]
    return {
        "label":                label,
        "verdict":              "⚠️ Phishing" if label == 1 else "✅ Legitimate",
        "confidence":           round(float(proba[label]), 4),
        "phishing_probability": round(float(proba[1]), 4),
    }


# ── Load everything ───────────────────────────────────────────────────────────
df = load_dataset()
model, scaler, metrics = get_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Phishing Detector")
    st.markdown("---")
    st.success("✅ Model Ready", icon="🟢")
    st.caption(
        f"Accuracy: **{metrics['accuracy']*100:.1f}%** | "
        f"ROC-AUC: **{metrics['roc_auc']}**"
    )
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🔍 Single Prediction", "📊 EDA & Insights", "📂 Batch Prediction"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Model: RandomForestClassifier\nDataset: 524,846 emails\nFeatures: 8")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — Single Prediction
# ════════════════════════════════════════════════════════════════════════════
if page == "🔍 Single Prediction":
    st.markdown('<p class="main-title">🛡️ Email Phishing Detector</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Enter the extracted features of an email and get an instant phishing verdict.</p>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📝 Text Statistics")
            num_words           = st.number_input("Total Words",     min_value=0, value=120, step=1)
            num_unique_words    = st.number_input("Unique Words",    min_value=0, value=80,  step=1)
            num_stopwords       = st.number_input("Stop Words",      min_value=0, value=30,  step=1)
            num_spelling_errors = st.number_input("Spelling Errors", min_value=0, value=0,   step=1)
        with col2:
            st.markdown("#### 🔗 Link & Address Statistics")
            num_links           = st.number_input("Hyperlinks",      min_value=0, value=3, step=1)
            num_unique_domains  = st.number_input("Unique Domains",  min_value=0, value=2, step=1)
            num_email_addresses = st.number_input("Email Addresses", min_value=0, value=1, step=1)
            num_urgent_keywords = st.number_input("Urgent Keywords", min_value=0, value=1, step=1)
        submitted = st.form_submit_button("🔍 Predict", use_container_width=True, type="primary")

    if submitted:
        payload = {
            "num_words": num_words, "num_unique_words": num_unique_words,
            "num_stopwords": num_stopwords, "num_links": num_links,
            "num_unique_domains": num_unique_domains, "num_email_addresses": num_email_addresses,
            "num_spelling_errors": num_spelling_errors, "num_urgent_keywords": num_urgent_keywords,
        }
        result = predict(payload, model, scaler)

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        label = result["label"]
        if label == 1:
            st.markdown(
                f'<div class="verdict-phish"><strong>⚠️ PHISHING EMAIL DETECTED</strong><br>'
                f'Confidence: {result["confidence"]*100:.1f}%</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="verdict-legit"><strong>✅ LEGITIMATE EMAIL</strong><br>'
                f'Confidence: {result["confidence"]*100:.1f}%</div>',
                unsafe_allow_html=True,
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Verdict",              result["verdict"])
        c2.metric("Phishing Probability", f'{result["phishing_probability"]*100:.1f}%')
        c3.metric("Confidence",           f'{result["confidence"]*100:.1f}%')

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result["phishing_probability"] * 100,
            title={"text": "Phishing Risk Score (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#e53e3e" if label == 1 else "#38a169"},
                "steps": [
                    {"range": [0,  40], "color": "#f0fff4"},
                    {"range": [40, 70], "color": "#fffbeb"},
                    {"range": [70, 100], "color": "#fff1f0"},
                ],
                "threshold": {
                    "line": {"color": "#1f2328", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            number={"suffix": "%"},
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Feature bar
        feature_vals = [payload[f] for f in FEATURES]
        fig_feat = px.bar(
            x=[FEATURE_LABELS[f] for f in FEATURES],
            y=feature_vals,
            labels={"x": "Feature", "y": "Value"},
            title="Input Feature Values",
            color=feature_vals,
            color_continuous_scale="Blues",
        )
        fig_feat.update_layout(showlegend=False, height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig_feat, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — EDA & Insights
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Insights":
    st.markdown('<p class="main-title">📊 Dataset EDA & Model Insights</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Exploratory analysis of 524,846 emails — distributions, correlations, and feature importances.</p>',
        unsafe_allow_html=True,
    )

    total   = len(df)
    phish   = int(df["label"].sum())
    legit   = total - phish
    phish_p = phish / total * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Emails", f"{total:,}")
    k2.metric("Phishing",     f"{phish:,}",  f"{phish_p:.2f}%")
    k3.metric("Legitimate",   f"{legit:,}",  f"{100-phish_p:.2f}%")
    k4.metric("Features",     "8")
    st.markdown("---")

    # Class distribution
    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie = px.pie(
            values=[legit, phish], names=["Legitimate", "Phishing"],
            title="Class Distribution",
            color_discrete_sequence=["#38a169", "#e53e3e"], hole=0.4,
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(height=350, margin=dict(t=50, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        class_df = pd.DataFrame({"Class": ["Legitimate", "Phishing"], "Count": [legit, phish]})
        fig_bar = px.bar(
            class_df, x="Class", y="Count", title="Email Count by Class",
            color="Class", color_discrete_map={"Legitimate": "#38a169", "Phishing": "#e53e3e"},
            text="Count",
        )
        fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_bar.update_layout(showlegend=False, height=350, margin=dict(t=50, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Feature distribution
    st.markdown("#### Feature Distributions by Class")
    selected_feat = st.selectbox("Select feature", FEATURES, format_func=lambda f: FEATURE_LABELS[f])
    df_plot = df[[selected_feat, "label"]].copy()
    df_plot["Class"] = df_plot["label"].map({0: "Legitimate", 1: "Phishing"})
    cap = df_plot[selected_feat].quantile(0.99)
    df_plot[selected_feat] = df_plot[selected_feat].clip(upper=cap)
    fig_hist = px.histogram(
        df_plot, x=selected_feat, color="Class", barmode="overlay", nbins=50,
        title=f"{FEATURE_LABELS[selected_feat]} Distribution (capped at 99th pct)",
        color_discrete_map={"Legitimate": "#38a169", "Phishing": "#e53e3e"}, opacity=0.75,
    )
    fig_hist.update_layout(height=350, margin=dict(t=50, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # Correlation heatmap
    st.markdown("#### Feature Correlation Heatmap")
    corr = df[FEATURES + ["label"]].corr()
    fig_heat = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Pearson Correlation Matrix",
    )
    fig_heat.update_layout(height=450, margin=dict(t=50, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # Feature importances
    st.markdown("#### Model Feature Importances")
    fi_df = pd.DataFrame({
        "Feature":    [FEATURE_LABELS[f] for f in FEATURES],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)
    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        title="RandomForest Feature Importances",
        color="Importance", color_continuous_scale="Blues",
    )
    fig_fi.update_layout(showlegend=False, height=350, margin=dict(t=50, b=10))
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    with st.expander("🗂️ Raw Dataset Preview (first 500 rows)"):
        st.dataframe(df.head(500), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — Batch Prediction
# ════════════════════════════════════════════════════════════════════════════
elif page == "📂 Batch Prediction":
    st.markdown('<p class="main-title">📂 Batch Email Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Upload a CSV with the 8 feature columns and get predictions for all rows.</p>',
        unsafe_allow_html=True,
    )

    template_csv = pd.DataFrame(columns=FEATURES).to_csv(index=False)
    st.download_button("⬇️ Download CSV Template", data=template_csv,
                       file_name="phishing_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        missing = [c for c in FEATURES if c not in batch_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        st.success(f"Loaded {len(batch_df):,} rows. Running predictions…")

        X = batch_df[FEATURES]
        X_s   = scaler.transform(X)
        preds  = model.predict(X_s)
        probas = model.predict_proba(X_s)[:, 1]

        result_df = batch_df[FEATURES].copy()
        result_df["phishing_probability"] = np.round(probas, 4)
        result_df["prediction"]           = preds
        result_df["verdict"]              = result_df["prediction"].map(
            {0: "✅ Legitimate", 1: "⚠️ Phishing"}
        )

        total_b = len(result_df)
        phish_b = int(result_df["prediction"].sum())
        legit_b = total_b - phish_b

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Emails",  f"{total_b:,}")
        c2.metric("🔴 Phishing",   f"{phish_b:,}")
        c3.metric("🟢 Legitimate", f"{legit_b:,}")

        fig_res = px.pie(
            values=[legit_b, phish_b], names=["Legitimate", "Phishing"],
            color_discrete_sequence=["#38a169", "#e53e3e"],
            hole=0.4, title="Batch Prediction Breakdown",
        )
        fig_res.update_layout(height=320, margin=dict(t=50, b=10))
        st.plotly_chart(fig_res, use_container_width=True)

        st.markdown("#### Prediction Results")
        st.dataframe(
            result_df.style.map(
                lambda v: "background-color:#fff1f0" if v == "⚠️ Phishing"
                else ("background-color:#f0fff4" if v == "✅ Legitimate" else ""),
                subset=["verdict"],
            ),
            use_container_width=True, height=400,
        )

        st.download_button(
            "⬇️ Download Results CSV",
            data=result_df.to_csv(index=False),
            file_name="phishing_predictions.csv",
            mime="text/csv",
        )
