<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=Fraud%20Detection%20System&fontSize=44&fontColor=ffffff&fontAlignY=38&desc=Real-Time%20Financial%20Fraud%20Intelligence%20%7C%20ANN%20%2B%20SMOTE&descAlignY=60&descColor=a8d8c2" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

[![GitHub Repo](https://img.shields.io/badge/View%20Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/kumarkislay245-spec/fraud-detection-system)
[![Author](https://img.shields.io/badge/Author-Kislay%20Kumar-50c88c?style=for-the-badge)](https://github.com/kumarkislay245-spec)

</div>

---

## 🏦 What Is This?

A **fraud detection engine** that analyzes financial transactions in real time and scores them for fraud risk. Built to handle one of the hardest classes of ML problems — **extreme class imbalance** — where fraud makes up a tiny fraction of a percent of all transactions.

The system combines a trained **Artificial Neural Network (ANN)**, balanced using **SMOTE (Synthetic Minority Over-sampling Technique)**, with custom feature engineering and a live **Streamlit dashboard** that lets anyone enter a transaction and instantly see a risk assessment with supporting visualizations.

---

## 📊 The Core Challenge — Extreme Class Imbalance

```
Total transactions:   1,10,000
Fraudulent cases:            120   (≈ 0.109%)
Legitimate cases:      1,09,880   (≈ 99.891%)
```

A naive model that predicts "not fraud" for every single transaction would still score **99.89% accuracy** — while catching zero actual fraud. That's the exact trap this project is designed to avoid: **accuracy is a meaningless metric here**, so the focus is on precision/recall for the fraud class instead.

**The solution:** Rather than relying on class weighting alone, the training data is rebalanced with **SMOTE**, which generates synthetic fraud examples so the ANN sees enough minority-class patterns to actually learn them — applied only to the training split to prevent data leakage into evaluation.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **ANN Risk Engine** | 5-layer feedforward network (64 → 32 → 16 → 8 → 1) trained on SMOTE-balanced data |
| ⚖️ **SMOTE Oversampling** | Synthetic minority-class samples generated for the training split only |
| ⚙️ **Custom Feature Engineering** | `errorBalanceOrg` and `errorBalanceOrig` expose hidden balance discrepancies |
| 🛡️ **Pre-ML Business Logic** | Auto-rejects transactions where `amount > sender balance` before calling the model |
| 📏 **Feature Scaling** | `StandardScaler` fit on training data, applied consistently at inference |
| 📊 **Interactive Dashboard** | Split-screen Streamlit UI — inputs on left, live risk assessment on right |
| 📈 **Plotly Visualizations** | Bar charts comparing fund availability and balance error metrics |
| 🔢 **Probability Scoring** | Continuous fraud probability (0–100%), not just a binary yes/no flag |
| ⚡ **Cached Model Loading** | `@st.cache_resource` ensures the model + scaler load once and stay in memory |

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
  errorBalanceOrg  = oldBalanceOrg  + amount - newBalanceOrig
  errorBalanceOrig = oldBalanceDest + amount - newBalanceDest
        │
        ▼
StandardScaler transforms features
        │
        ▼
ANN (trained on SMOTE-balanced data) outputs fraud probability
        │
        ▼
prob ≥ 0.75 → 🚨 TRANSACTION BLOCKED
prob <  0.75 → ✅ TRANSACTION APPROVED
        │
        ▼
Display Plotly charts + technical metrics
```

### Why `errorBalance` features?

Raw transaction amounts don't tell the full story. A fraudster might move ₹10,000 but the sender's or receiver's balance doesn't change the way it should — there's a discrepancy. The `errorBalance` features capture exactly this mismatch, which raw amounts completely miss.

```python
# The key feature engineering step
errorBalanceOrg  = oldBalanceOrg  + amount - newBalanceOrig   # should be ~0 if legit
errorBalanceOrig = oldBalanceDest + amount - newBalanceDest   # should be ~0 if legit
```

Any non-zero value here signals that money didn't flow the way the transaction claims.

---

## 📁 Project Structure

```
fraud-detection-system/
│
├── app.py                   # Streamlit app (full dashboard + ML inference)
├── ann_smote_model.keras    # Trained ANN model (Keras format)
├── scaler.pkl                # StandardScaler fitted on training data
├── requirements.txt          # Python dependencies
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
streamlit run app.py
```

---

## 📦 Dependencies

```txt
streamlit
tensorflow
scikit-learn
pandas
numpy
plotly
joblib
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
- Result: ✅ **APPROVED** — balances reconcile, low fraud probability

**Example — Fraudulent Pattern:**
- Amount: ₹50,000 | Sender balance: ₹50,000 → ₹0 | Receiver: ₹0 → ₹0 (money vanishes)
- Result: 🚨 **BLOCKED** — `errorBalanceOrig` spikes, model scores high fraud probability

---

## 🔑 Key Technical Decisions

**Why ANN + SMOTE instead of a tree-based model?**
With only 120 fraud cases out of 1.10 lakh transactions, tree-based models (Decision Tree, Random Forest, XGBoost) were also tested, but struggled to generalize fraud patterns from so few real examples. SMOTE lets the ANN train on a much richer, synthetically balanced view of the minority class — applied strictly to the training split only, so evaluation still reflects real-world class distribution.

**Why a threshold of 0.75 instead of 0.5?**
Threshold sweeps across the SMOTE-trained ANN's output probabilities showed that 0.75 gave the best precision-recall balance for this dataset — high enough to avoid excessive false positives, while still catching genuine fraud patterns.

**Why `.keras` format + a separate `scaler.pkl`?**
The ANN is saved in Keras's native `.keras` format (not pickle) for stability across TensorFlow versions. Because the network was trained on scaled features, the exact `StandardScaler` used during training is persisted separately with `joblib` and applied identically at inference — skipping this step would silently break predictions.

---

## 👨‍💻 Author

**Kislay Kumar** — NIT Warangal

[![GitHub](https://img.shields.io/badge/GitHub-kumarkislay245--spec-181717?style=flat-square&logo=github)](https://github.com/kumarkislay245-spec)

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=100&section=footer" width="100%"/>
</div>
