<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=Fraud%20Detection%20System&fontSize=44&fontColor=ffffff&fontAlignY=38&desc=Real-Time%20Financial%20Fraud%20Intelligence%20%7C%206.3M%20Transactions&descAlignY=60&descColor=a8d8c2" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

[![GitHub Repo](https://img.shields.io/badge/View%20Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/kumarkislay245-spec/fraud-detection-system)
[![Author](https://img.shields.io/badge/Author-Kislay%20Kumar-50c88c?style=for-the-badge)](https://github.com/kumarkislay245-spec)

</div>

---

## 🏦 What Is This?

An **enterprise-grade fraud detection engine** that analyzes financial transactions in real time and scores them for fraud risk. Built to handle the hardest class of ML problems — extreme class imbalance — where only **0.13% of 6.3 million transactions** are actually fraudulent.

The system combines a trained **XGBoost classifier** with custom feature engineering and a live **Streamlit dashboard** that lets anyone enter a transaction and instantly see a risk assessment with supporting visualizations.

---

## 📊 The Core Challenge — Imbalanced Classification

```
Total transactions:   6,300,000
Fraudulent cases:         8,213   (≈ 0.13%)
Legitimate cases:     6,291,787   (≈ 99.87%)
```

A naive model that predicts "not fraud" for every transaction gets **99.87% accuracy** — and catches zero fraud. That's the trap this project is specifically designed to avoid.

**The solution:** Tune XGBoost's `scale_pos_weight` parameter to penalize missing fraud cases far more than false alarms — because in finance, a missed fraud costs far more than a blocked legitimate transaction.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **XGBoost Risk Engine** | Trained on 6.3M records with imbalance-aware configuration |
| ⚙️ **Custom Feature Engineering** | `errorBalanceOrg` and `errorBalanceDest` expose hidden balance discrepancies |
| 🛡️ **Pre-ML Business Logic** | Auto-rejects transactions where `amount > sender balance` before calling the model |
| 📊 **Interactive Dashboard** | Split-screen Streamlit UI — inputs on left, live risk assessment on right |
| 📈 **Plotly Visualizations** | Bar charts comparing fund availability and balance error metrics |
| 🔢 **Probability Scoring** | Continuous fraud probability (0–100%), not just a binary yes/no flag |
| ⚡ **Cached Model Loading** | `@st.cache_resource` ensures the model loads once and stays in memory |

---

## 🧠 How It Works

```
User enters transaction details
(amount, sender balances, receiver balances)
        │
        ▼
Business Logic Check: amount > sender balance?
   YES → Reject immediately (Insufficient Funds)
   NO  → Continue to ML model
        │
        ▼
Feature Engineering:
  errorBalanceOrg  = newBalanceOrig + amount - oldBalanceOrg
  errorBalanceDest = oldBalanceDest + amount - newBalanceDest
        │
        ▼
XGBoost model predicts fraud probability
        │
        ▼
prob > 0.10 → 🚨 TRANSACTION BLOCKED
prob ≤ 0.10 → ✅ TRANSACTION APPROVED
        │
        ▼
Display Plotly charts + technical metrics
```

### Why `errorBalance` features?

Raw transaction amounts don't tell the full story. A fraudster might move ₹10,000 but the sender's balance doesn't drop by ₹10,000 — there's a discrepancy. The `errorBalance` features capture exactly this mismatch, which raw amounts completely miss.

```python
# The key feature engineering step
errorBalanceOrg  = newBalanceOrig + amount - oldBalanceOrg  # should be ~0 if legit
errorBalanceDest = oldBalanceDest + amount - newBalanceDest  # should be ~0 if legit
```

Any non-zero value here signals that money didn't flow the way the transaction claims.

---

## 📁 Project Structure

```
fraud-detection-system/
│
├── file.py                  # Streamlit app (full dashboard + ML inference)
├── model.json               # Trained XGBoost model (JSON format)
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/kumarkislay245-spec/fraud-detection-system.git
cd fraud-detection-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the app
```bash
https://fraud-detection-system-tvmo2rt687twsxnf2lnm4m.streamlit.app/
```
---

## 📦 Dependencies

```txt
streamlit
xgboost
pandas
plotly
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🖥️ Dashboard Preview

> *The app opens in a split-screen layout. Left panel: enter transaction amount, sender's opening/closing balance, and receiver's opening/closing balance. Right panel: instant AI risk score with a progress bar, approval/rejection verdict, and Plotly bar charts showing balance discrepancy metrics.*

**Example — Legitimate Transaction:**
- Amount: ₹1,000 | Sender balance: ₹5,000 → ₹4,000 | Receiver: ₹0 → ₹1,000
- Result: ✅ **APPROVED** — balances reconcile, low error score

**Example — Fraudulent Pattern:**
- Amount: ₹50,000 | Sender balance: ₹50,000 → ₹0 | Receiver: ₹0 → ₹0 (money vanishes)
- Result: 🚨 **BLOCKED** — `errorBalanceDest` spikes, model scores high fraud probability

---

## 🔑 Key Technical Decisions

**Why XGBoost over Logistic Regression or Random Forest?**
XGBoost handles tabular financial data extremely well and natively supports `scale_pos_weight` for imbalanced classes. It also trains faster on large datasets compared to deep learning approaches.

**Why a threshold of 0.10 instead of 0.50?**
At 0.50 threshold, the model misses most fraud because the prior probability is only 0.13%. Lowering the threshold to 0.10 dramatically improves Recall (catching real fraud) at the cost of a few more false positives — an acceptable trade-off in financial fraud detection.

**Why JSON model format over pickle?**
`model.json` (XGBoost's native format) is safer, version-stable, and more portable than Python pickle — it can be loaded in any XGBoost version without compatibility issues.

---

## 👨‍💻 Author

**Kislay Kumar** — NIT Warangal

[![GitHub](https://img.shields.io/badge/GitHub-kumarkislay245--spec-181717?style=flat-square&logo=github)](https://github.com/kumarkislay245-spec)

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=100&section=footer" width="100%"/>
</div>
