import os

# MUST be set before tensorflow is imported — forces TF to skip GPU/CUDA probing.
# On Streamlit Cloud's sandboxed containers, TF's CUDA init can segfault instead
# of failing gracefully, which is what was crashing the app.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import tensorflow as tf

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Fraud Intelligence Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed decision threshold (as decided): only flag fraud if prob >= 0.75
FRAUD_THRESHOLD = 0.75

# ============================================================
# THEME — Deep-ledger banking aesthetic
# Ink navy background, steel-blue + muted gold accents, mono for
# every number (feels like a ledger, not a generic dashboard).
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #0D1321;
    --surface: #141C2E;
    --surface-2: #1B2740;
    --border: #263252;
    --steel: #4C8FC9;
    --steel-dim: #2E5478;
    --gold: #D8973C;
    --text: #E8ECF3;
    --text-dim: #8B96AD;
    --approve: #3E9C73;
    --block: #C1443C;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(76,143,201,0.08), transparent),
        var(--ink);
    color: var(--text);
}

/* ---------- Header ---------- */
.hdr-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.28em;
    color: var(--steel);
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.hdr-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    color: var(--text);
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.01em;
}
.hdr-sub {
    font-family: 'Inter', sans-serif;
    color: var(--text-dim);
    font-size: 0.95rem;
    margin-bottom: 0;
}
.hdr-rule {
    height: 1px;
    background: linear-gradient(90deg, var(--steel) 0%, var(--border) 35%, transparent 100%);
    margin: 1.3rem 0 1.8rem 0;
}

/* ---------- Section labels ---------- */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
.field-group-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin: 0.6rem 0 0.2rem 0;
}

/* ---------- Form / inputs ---------- */
[data-testid="stForm"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.6rem 1.6rem 1.2rem 1.6rem;
}
[data-testid="stNumberInput"] label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    color: var(--text-dim);
    text-transform: uppercase;
}
[data-testid="stNumberInput"] input {
    font-family: 'IBM Plex Mono', monospace;
    background: var(--ink) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--steel) !important;
    box-shadow: 0 0 0 1px var(--steel) !important;
}

/* Submit button */
[data-testid="stFormSubmitButton"] button {
    background: var(--steel) !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 0.8rem !important;
    font-weight: 600;
    color: #0A0E17 !important;
    padding: 0.6rem 0 !important;
    transition: background 0.15s ease;
}
[data-testid="stFormSubmitButton"] button:hover {
    background: #63A6DA !important;
}

/* ---------- Result placeholder ---------- */
.idle-panel {
    border: 1px dashed var(--border);
    border-radius: 10px;
    padding: 3.2rem 1.5rem;
    text-align: center;
    color: var(--text-dim);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
}

/* ---------- Verdict stamp ---------- */
.stamp-wrap {
    display: flex;
    align-items: center;
    gap: 1.4rem;
    padding: 1.2rem 0 0.6rem 0;
}
.stamp {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.55rem 1.1rem;
    border: 3px solid currentColor;
    border-radius: 6px;
    transform: rotate(-3deg);
    display: inline-block;
    white-space: nowrap;
}
.stamp-approve { color: var(--approve); }
.stamp-block { color: var(--block); }
.stamp-caption {
    font-family: 'Inter', sans-serif;
    color: var(--text-dim);
    font-size: 0.9rem;
    line-height: 1.4;
}

/* ---------- Probability readout ---------- */
.prob-readout {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.6rem;
    font-weight: 600;
    line-height: 1;
    margin: 0.4rem 0 0.1rem 0;
}
.prob-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-dim);
}
.prob-threshold {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-dim);
    margin-top: 0.3rem;
}

/* ---------- Cards ---------- */
.info-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
}

/* ---------- Streamlit native overrides ---------- */
[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--text) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--text-dim) !important;
    text-transform: uppercase;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em;
}
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }

.footer-note {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    color: var(--text-dim);
    text-align: center;
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)


PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#8B96AD", size=12),
    title_font=dict(family="Space Grotesk, sans-serif", color="#E8ECF3", size=15),
    margin=dict(t=50, l=10, r=10, b=10),
)


# ============================================================
# LOAD MODEL + SCALER
# ============================================================
@st.cache_resource
def load_artifacts():
    """
    Loads the trained ANN (Keras) model + the StandardScaler used during training.
    NOTE: The ANN was saved via model.save('ann_smote_model.keras'), so it must be
    loaded with tf.keras.models.load_model(), NOT joblib.load(). joblib is only for
    the scaler (a plain sklearn object).
    """
    model_path = os.path.join(BASE_DIR, "ann_smote_model.keras")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None, None

    if not os.path.exists(scaler_path):
        st.error(
            f"scaler.pkl not found at: {scaler_path}. "
            "Save it in your notebook with joblib.dump(scaler, 'scaler.pkl') "
            "and push it to the repo — the ANN was trained on scaled features."
        )
        return None, None

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


model, scaler = load_artifacts()


# ============================================================
# FEATURE ENGINEERING (same logic used in training)
# ============================================================
def engineer_features(amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest):
    errorBalanceOrg = (oldbalanceOrg + amount) - newbalanceOrig
    errorBalanceOrig = (oldbalanceDest + amount) - newbalanceDest

    return pd.DataFrame([{
        "amount": amount,
        "errorBalanceOrg": errorBalanceOrg,
        "errorBalanceOrig": errorBalanceOrig
    }])


# ============================================================
# PREDICTION
# ============================================================
def predict_fraud(model, scaler, input_data: pd.DataFrame):
    """
    Scales features and runs the ANN. The model has a single sigmoid output,
    so calling it directly returns the fraud probability
    (there's no predict_proba like sklearn/RandomForest).
    """
    scaled_input = scaler.transform(input_data).astype("float32")
    # Using model(...) directly instead of model.predict() — predict() runs
    # through TF's batching/threading pipeline which is NOT thread-safe and
    # segfaults when called from Streamlit's non-main session thread. A direct
    # call uses TF's lightweight, thread-safe execution path.
    prob = float(model(scaled_input, training=False).numpy()[0][0])
    is_fraud = prob >= FRAUD_THRESHOLD
    return prob, is_fraud


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hdr-eyebrow">RISK &amp; COMPLIANCE · REAL-TIME SCORING</div>
<div class="hdr-title">🏦 Fraud Intelligence Engine</div>
<p class="hdr-sub">Enter a transaction below to score it for fraud risk against balance-flow patterns.</p>
<div class="hdr-rule"></div>
""", unsafe_allow_html=True)

# ============================================================
# LAYOUT
# ============================================================
col_input, col_result = st.columns([1, 1.25], gap="large")

with col_input:
    st.markdown('<div class="section-label">Transaction Details</div>', unsafe_allow_html=True)
    with st.form("transaction_form", border=True):
        amount = st.number_input("Transaction Amount (₹)", min_value=0.0, value=1000.0, step=500.0)

        st.markdown('<div class="field-group-label">Sender</div>', unsafe_allow_html=True)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            oldbalanceOrg = st.number_input("Balance Before", min_value=0.0, value=50000.0, step=500.0)
        with col_s2:
            newbalanceOrig = st.number_input("Balance After", min_value=0.0, value=49000.0, step=500.0)

        st.markdown('<div class="field-group-label">Receiver</div>', unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            oldbalanceDest = st.number_input("Balance Before", min_value=0.0, value=4000.0, step=500.0)
        with col_r2:
            newbalanceDest = st.number_input("Balance After", min_value=0.0, value=5000.0, step=500.0)

        submitted = st.form_submit_button("Analyze Transaction", use_container_width=True)

with col_result:
    st.markdown('<div class="section-label">Risk Assessment</div>', unsafe_allow_html=True)

    if not submitted:
        st.markdown(
            '<div class="idle-panel">AWAITING TRANSACTION INPUT<br>'
            '<span style="opacity:0.6;">Results will render here once analyzed</span></div>',
            unsafe_allow_html=True
        )

    elif amount > oldbalanceOrg:
        st.markdown(
            '<div class="stamp-wrap">'
            '<div class="stamp stamp-block">Rejected</div>'
            '<div class="stamp-caption">Insufficient funds — sender cannot cover this transaction.</div>'
            '</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<p style="color:var(--text-dim);font-family:\'IBM Plex Mono\',monospace;font-size:0.85rem;">'
            f'Available: ₹{oldbalanceOrg:,.2f} &nbsp;·&nbsp; Requested: ₹{amount:,.2f}</p>',
            unsafe_allow_html=True
        )

        fig = go.Figure(data=[
            go.Bar(name='Available Balance', x=['Balance'], y=[oldbalanceOrg], marker_color='#3E9C73'),
            go.Bar(name='Transaction Amount', x=['Balance'], y=[amount], marker_color='#C1443C')
        ])
        fig.update_layout(barmode='group', height=280, title="Fund Availability", **PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    elif model is None or scaler is None:
        st.error("Model or scaler could not be loaded. Check that both files exist in the repo.")

    else:
        with st.spinner("Scoring transaction..."):
            input_data = engineer_features(
                amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
            )
            prob, is_fraud = predict_fraud(model, scaler, input_data)

            if is_fraud:
                st.markdown(
                    '<div class="stamp-wrap">'
                    '<div class="stamp stamp-block">Blocked</div>'
                    '<div class="stamp-caption">This transaction matches learned fraud patterns.</div>'
                    '</div>', unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="stamp-wrap">'
                    '<div class="stamp stamp-approve">Approved</div>'
                    '<div class="stamp-caption">Fraud probability is below the decision threshold.</div>'
                    '</div>', unsafe_allow_html=True
                )

            readout_color = "var(--block)" if is_fraud else "var(--approve)"
            st.markdown(f"""
                <div class="prob-label">Fraud Probability</div>
                <div class="prob-readout" style="color:{readout_color};">{prob*100:.2f}%</div>
                <div class="prob-threshold">Decision threshold: {FRAUD_THRESHOLD*100:.0f}%</div>
            """, unsafe_allow_html=True)

            st.progress(prob)
            st.write("")

            # Error Balance Chart
            errorBalanceOrg = input_data["errorBalanceOrg"].iloc[0]
            errorBalanceOrig = input_data["errorBalanceOrig"].iloc[0]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Sender Error', 'Receiver Error'],
                y=[abs(errorBalanceOrg), abs(errorBalanceOrig)],
                marker_color=['#4C8FC9', '#D8973C'],
                text=[f'₹{abs(errorBalanceOrg):,.2f}', f'₹{abs(errorBalanceOrig):,.2f}'],
                textposition='auto'
            ))
            fig.update_layout(title="Balance Discrepancy Analysis", height=280, **PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)

            # Technical Details
            with st.expander("Technical Metrics"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Fraud Probability", f"{prob:.2%}")
                col2.metric("Threshold Used", f"{FRAUD_THRESHOLD}")
                col3.metric("Decision", "FRAUD" if is_fraud else "LEGIT")

                st.markdown('<div class="field-group-label" style="margin-top:1rem;">Raw Feature Values (before scaling)</div>', unsafe_allow_html=True)
                st.dataframe(input_data, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="footer-note">FRAUD INTELLIGENCE ENGINE · KISLAY KUMAR · NIT WARANGAL</div>',
    unsafe_allow_html=True
)
