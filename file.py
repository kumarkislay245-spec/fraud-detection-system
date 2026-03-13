import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import plotly.graph_objects as go

# --- 1. PREMIUM PAGE CONFIG ---
st.set_page_config(
    page_title="Fraud Intelligence Engine",
    page_icon="🏦",
    layout="wide"
)


# --- 2. MODEL LOADING ---
@st.cache_resource
def load_xgboost_model():
    model_path = 'model.json'
    if os.path.exists(model_path):
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model
    return None


model = load_xgboost_model()

# --- 3. HEADER SECTION ---
st.title("🏦 Fraud Intelligence & Risk Engine")
st.markdown("Real-time transaction monitoring with automated balance validation.")
st.divider()

# --- 4. SPLIT-SCREEN LAYOUT ---
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    with st.form("transaction_form", border=True):
        st.subheader("📥 Transaction Telemetry")

        amount = st.number_input("Transaction Amount (₹)", min_value=0.0, value=1000.0, step=500.0)

        st.markdown("**Sender Details**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            oldbalanceOrg = st.number_input("Initial Balance", min_value=0.0, value=5000.0, step=500.0)
        with col_s2:
            newbalanceOrig = st.number_input("Final Balance", min_value=0.0, value=4000.0, step=500.0)

        st.markdown("**Receiver Details**")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            oldbalanceDest = st.number_input("Initial Balance", min_value=0.0, value=0.0, step=500.0)
        with col_r2:
            newbalanceDest = st.number_input("Final Balance", min_value=0.0, value=1000.0, step=500.0)

        submitted = st.form_submit_button("🔍 Analyze Risk Profile", type="primary", use_container_width=True)

with col_result:
    if not submitted:
        st.info("👈 Enter transaction details and click 'Analyze Risk Profile' to generate a report.")
    else:
        # --- BUSINESS LOGIC CHECK: Insufficient Funds ---
        if amount > oldbalanceOrg:
            st.error("### ❌ TRANSACTION REJECTED: INSUFFICIENT FUNDS")
            st.warning(f"The sender only has **₹{oldbalanceOrg:,.2f}**, but is attempting to send **₹{amount:,.2f}**.")

            # Visual check for the user
            fig = go.Figure(data=[
                go.Bar(name='Available', x=['Balance Status'], y=[oldbalanceOrg], marker_color='green'),
                go.Bar(name='Requested', x=['Balance Status'], y=[amount], marker_color='red')
            ])
            fig.update_layout(barmode='group', height=300, title="Fund Availability Comparison")
            st.plotly_chart(fig, use_container_width=True)

        elif model is None:
            st.error("System Error: 'model.json' offline.")
        else:
            with st.spinner("Analyzing transaction patterns..."):
                # Feature Engineering
                errorBalanceOrg = newbalanceOrig + amount - oldbalanceOrg
                errorBalanceOrig = oldbalanceDest + amount - newbalanceDest

                input_data = pd.DataFrame([{
                    'amount': amount,
                    'errorBalanceOrg': errorBalanceOrg,
                    'errorBalanceOrig': errorBalanceOrig
                }])

                # AI Prediction
                prob = float(model.predict_proba(input_data)[0][1])

                st.subheader("📊 AI Risk Assessment Report")

                # Dynamic Progress Bar
                st.markdown(f"**Fraud Probability Score: {prob * 100:.2f}%**")
                if prob > 0.10:
                    st.progress(prob, text="Critical Risk Level")
                    st.error("### 🚨 ACTION: TRANSACTION BLOCKED")
                    st.markdown("AI Model matched this transaction with known fraud signatures.")
                else:
                    st.progress(prob, text="Safe Margin")
                    st.success("### ✅ ACTION: TRANSACTION APPROVED")

                # Comparison Chart for Balances
                fig = go.Figure()
                fig.add_trace(go.Bar(x=['Sender', 'Receiver'], y=[abs(errorBalanceOrg), abs(errorBalanceOrig)],
                                     marker_color=['#FF4B4B', '#00CC96']))
                fig.update_layout(title="Detected Discrepancy (Error Balance)", height=300)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("⚙️ View Technical Metrics"):
                    st.write(f"Sender Error: ₹{errorBalanceOrg}")
                    st.write(f"Receiver Error: ₹{errorBalanceOrig}")
                    st.dataframe(input_data)

# --- FOOTER ---
st.markdown("<br><p style='text-align: center; color: gray;'>Kislay Kumar | NIT Warangal</p>", unsafe_allow_html=True)