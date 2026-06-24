import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Fraud Intelligence Engine",
    page_icon="🏦",
    layout="wide"
)

# --- MODEL & THRESHOLD LOADING ---
import os

# Yeh add karo file ke bilkul upar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model_path     = os.path.join(BASE_DIR, 'fraud_model(1).pkl')
    threshold_path = os.path.join(BASE_DIR, 'threshold(1).pkl')

    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")  # debug ke liye
        return None, 0.7

    model     = joblib.load(model_path)
    threshold = joblib.load(threshold_path) if os.path.exists(threshold_path) else 0.7
    return model, threshold
    
model, threshold = load_model()

# --- HEADER ---
st.title("🏦 Fraud Intelligence & Risk Engine")
st.markdown("Real-time transaction monitoring with automated balance validation.")
st.divider()

# --- LAYOUT ---
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

    else:
        # --- CHECK 1: Insufficient Funds ---
        if amount > oldbalanceOrg:
            st.error("### ❌ TRANSACTION REJECTED: INSUFFICIENT FUNDS")
            st.warning(f"Sender has **₹{oldbalanceOrg:,.2f}** but trying to send **₹{amount:,.2f}**.")

            fig = go.Figure(data=[
                go.Bar(name='Available Balance', x=['Balance'], y=[oldbalanceOrg], marker_color='green'),
                go.Bar(name='Transaction Amount', x=['Balance'], y=[amount], marker_color='red')
            ])
            fig.update_layout(barmode='group', height=300, title="Fund Availability")
            st.plotly_chart(fig, use_container_width=True)

        # --- CHECK 2: Model not found ---
        elif model is None:
            st.error("❌ fraud_model.pkl not found. Please add model file.")

        else:
            with st.spinner("Analyzing transaction..."):

                # --- FIXED FEATURE ENGINEERING ---
                errorBalanceOrg  = (oldbalanceOrg  + amount) - newbalanceOrig   # sender error
                errorBalanceOrig = (oldbalanceDest + amount) - newbalanceDest   # receiver error

                input_data = pd.DataFrame([{
                    'amount':           amount,
                    'errorBalanceOrg':  errorBalanceOrg,
                    'errorBalanceOrig': errorBalanceOrig
                }])

                # --- PREDICTION WITH THRESHOLD ---
                prob = float(model.predict_proba(input_data)[0][1])
                is_fraud = prob >= threshold

                st.subheader("📊 Risk Assessment Report")
                st.markdown(f"**Fraud Probability: {prob * 100:.2f}%** &nbsp;|&nbsp; Threshold: {threshold}")

                # Result
                if is_fraud:
                    st.progress(prob, text="High Risk")
                    st.error("### 🚨 TRANSACTION BLOCKED")
                    st.markdown("This transaction matches known fraud patterns.")
                else:
                    st.progress(prob, text="Low Risk")
                    st.success("### ✅ TRANSACTION APPROVED")
                    st.markdown("No fraud patterns detected.")

                # Error Balance Chart
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
                    col2.metric("Threshold Used", f"{threshold}")
                    col3.metric("Decision", "FRAUD" if is_fraud else "LEGIT")

                    st.markdown("**Feature Values Sent to Model:**")
                    st.dataframe(input_data, use_container_width=True)

# --- FOOTER ---
st.markdown(
    "<br><p style='text-align:center; color:gray;'>Kislay Kumar | NIT Warangal</p>",
    unsafe_allow_html=True
)
