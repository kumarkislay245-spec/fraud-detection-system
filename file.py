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
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed decision threshold (as decided): only flag fraud if prob >= 0.75
FRAUD_THRESHOLD = 0.75


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
    so model.predict() directly returns the fraud probability
    (there's no predict_proba like sklearn/RandomForest).
    """
    scaled_input = scaler.transform(input_data)
    prob = float(model.predict(scaled_input, verbose=0)[0][0])
    is_fraud = prob >= FRAUD_THRESHOLD
    return prob, is_fraud


# ============================================================
# HEADER
# ============================================================
st.title("🏦 Fraud Intelligence & Risk Engine")
st.markdown("Real-time transaction monitoring with automated balance validation.")
st.divider()

# ============================================================
# LAYOUT
# ============================================================
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    with st.form("transaction_form", border=True):
        st.subheader("📥 Transaction Details")

        amount = st.number_input("Transaction Amount (₹)", min_value=0.0, value=1000.0, step=500.0)

        st.markdown("**Sender Details**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            oldbalanceOrg = st.number_input("Balance Before", min_value=0.0, value=50000.0, step=500.0)
        with col_s2:
            newbalanceOrig = st.number_input("Balance After", min_value=0.0, value=49000.0, step=500.0)

        st.markdown("**Receiver Details**")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            oldbalanceDest = st.number_input("Balance Before", min_value=0.0, value=4000.0, step=500.0)
        with col_r2:
            newbalanceDest = st.number_input("Balance After", min_value=0.0, value=5000.0, step=500.0)

        submitted = st.form_submit_button("🔍 Analyze Transaction", type="primary", use_container_width=True)

with col_result:
    if not submitted:
        st.info("👈 Enter transaction details and click Analyze to get results.")

    elif amount > oldbalanceOrg:
        st.error("### ❌ TRANSACTION REJECTED: INSUFFICIENT FUNDS")
        st.warning(f"Sender has **₹{oldbalanceOrg:,.2f}** but trying to send **₹{amount:,.2f}**.")

        fig = go.Figure(data=[
            go.Bar(name='Available Balance', x=['Balance'], y=[oldbalanceOrg], marker_color='green'),
            go.Bar(name='Transaction Amount', x=['Balance'], y=[amount], marker_color='red')
        ])
        fig.update_layout(barmode='group', height=300, title="Fund Availability")
        st.plotly_chart(fig, use_container_width=True)

    elif model is None or scaler is None:
        st.error("❌ Model or scaler could not be loaded. Check that both files exist in the repo.")

    else:
        with st.spinner("Analyzing transaction..."):
            input_data = engineer_features(
                amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
            )
            prob, is_fraud = predict_fraud(model, scaler, input_data)

            st.subheader("📊 Risk Assessment Report")
            st.markdown(f"**Fraud Probability: {prob * 100:.2f}%** &nbsp;|&nbsp; Threshold: {FRAUD_THRESHOLD}")

            if is_fraud:
                st.progress(prob, text="High Risk")
                st.error("### 🚨 TRANSACTION BLOCKED")
                st.markdown("This transaction matches known fraud patterns.")
            else:
                st.progress(prob, text="Low Risk")
                st.success("### ✅ TRANSACTION APPROVED")
                st.markdown(f"Fraud probability ({prob*100:.2f}%) is below the {FRAUD_THRESHOLD*100:.0f}% threshold.")

            # Error Balance Chart
            errorBalanceOrg = input_data["errorBalanceOrg"].iloc[0]
            errorBalanceOrig = input_data["errorBalanceOrig"].iloc[0]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Sender Error', 'Receiver Error'],
                y=[abs(errorBalanceOrg), abs(errorBalanceOrig)],
                marker_color=['#FF4B4B', '#00CC96'],
                text=[f'₹{abs(errorBalanceOrg):,.2f}', f'₹{abs(errorBalanceOrig):,.2f}'],
                textposition='auto'
            ))
            fig.update_layout(title="Balance Discrepancy Analysis", height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Technical Details
            with st.expander("⚙️ Technical Metrics"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Fraud Probability", f"{prob:.2%}")
                col2.metric("Threshold Used", f"{FRAUD_THRESHOLD}")
                col3.metric("Decision", "FRAUD" if is_fraud else "LEGIT")

                st.markdown("**Raw Feature Values (before scaling):**")
                st.dataframe(input_data, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    "<br><p style='text-align:center; color:gray;'>Kislay Kumar | NIT Warangal</p>",
    unsafe_allow_html=True
)
