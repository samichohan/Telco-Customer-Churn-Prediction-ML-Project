from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "model" / "churn_pipeline.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# ----------------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, #1a2250 0%, #0B1020 55%) fixed;
}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px;}

.hero {
    background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 55%, #06B6D4 100%);
    border-radius: 20px; padding: 28px 34px; margin-bottom: 24px;
    box-shadow: 0 10px 40px rgba(59,130,246,0.25);
}
.hero h1 {color: #fff; font-size: 2rem; margin: 0 0 6px 0; font-weight: 800; letter-spacing: -0.5px;}
.hero p {color: rgba(255,255,255,0.88); margin: 0; font-size: 1rem;}

[data-testid="stForm"] {
    background: #131A2E; border: 1px solid #222C4A; border-radius: 18px; padding: 22px;
}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {
    background: #0F1630; border-radius: 10px; padding: 8px 18px; color: #AEB7D3;
}
.stTabs [aria-selected="true"] {background: #6C63FF !important; color: #fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}

label, .stSlider label, .stSelectbox label, .stNumberInput label {color: #C9D1EA !important; font-weight: 600 !important;}
div[data-baseweb="select"] > div, .stNumberInput input {
    background: #0F1630 !important; border: 1px solid #26335A !important; border-radius: 10px !important;
}

.stButton > button, [data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #6C63FF, #3B82F6); color: white; border: 0;
    border-radius: 12px; padding: 0.7rem 1rem; font-weight: 700; font-size: 1rem;
    box-shadow: 0 6px 20px rgba(108,99,255,0.35); transition: transform .15s ease;
}
[data-testid="stFormSubmitButton"] > button:hover {transform: translateY(-2px); color: #fff;}

.card {
    background: #131A2E; border: 1px solid #222C4A; border-radius: 18px;
    padding: 22px; margin-bottom: 16px;
}
.card h3 {color: #E6E9F2; margin: 0 0 12px 0; font-size: 1.05rem; font-weight: 700;}
.muted {color: #8F9BC0; font-size: 0.85rem;}

.gauge {width: 200px; height: 200px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 6px auto 14px auto;}
.gauge-inner {width: 160px; height: 160px; border-radius: 50%; background: #131A2E; display: flex; flex-direction: column; align-items: center; justify-content: center;}
.gauge-value {font-size: 2.1rem; font-weight: 800; color: #fff; line-height: 1.1;}
.gauge-label {font-size: 0.75rem; color: #8F9BC0; text-transform: uppercase; letter-spacing: 1px;}

.badge {display: inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 700; font-size: 0.9rem;}
.badge-low {background: rgba(34,197,94,0.15); color: #4ADE80; border: 1px solid rgba(34,197,94,0.4);}
.badge-med {background: rgba(245,158,11,0.15); color: #FBBF24; border: 1px solid rgba(245,158,11,0.4);}
.badge-high {background: rgba(239,68,68,0.15); color: #F87171; border: 1px solid rgba(239,68,68,0.4);}

.factor {display: flex; gap: 12px; align-items: flex-start; padding: 10px 0; border-bottom: 1px solid #1D2745;}
.factor:last-child {border-bottom: 0;}
.factor .dot {width: 9px; height: 9px; border-radius: 50%; margin-top: 7px; flex-shrink: 0;}
.factor .t {color: #E6E9F2; font-weight: 600; font-size: 0.92rem;}
.factor .d {color: #8F9BC0; font-size: 0.82rem;}

.stat {background: #0F1630; border: 1px solid #222C4A; border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;}
.stat .n {color: #fff; font-weight: 800; font-size: 1.25rem;}
.stat .l {color: #8F9BC0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px;}

.empty {text-align: center; padding: 50px 10px;}
.empty .big {font-size: 3rem;}
.empty .t {color: #E6E9F2; font-weight: 700; margin-top: 8px;}
.empty .d {color: #8F9BC0; font-size: 0.9rem; margin-top: 4px;}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📡 About the Model")
    st.markdown(
        '<p class="muted">Logistic Regression (C=0.1, class_weight=balanced) '
        "trained on 7,043 Telco customers. Preprocessing and model are saved "
        "as one scikit-learn pipeline.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="stat"><div class="n">78.6%</div><div class="l">Recall (churn class)</div></div>'
        '<div class="stat"><div class="n">50.9%</div><div class="l">Precision (churn class)</div></div>'
        '<div class="stat"><div class="n">0.841</div><div class="l">ROC-AUC</div></div>'
        '<div class="stat"><div class="n">0.618</div><div class="l">F1-score</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### 🎯 Risk Levels")
    st.markdown(
        '<p class="muted">🟢 Low: below 30%<br>🟠 Medium: 30% to 50%<br>🔴 High: 50% and above</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption(
        "The prediction is a statistical estimate, not a guarantee. "
        "Use it to prioritise retention outreach, not as a final decision."
    )

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>Telco Customer Churn Predictor</h1>'
    "<p>Enter a customer's profile to estimate their probability of leaving, "
    "see the main risk drivers and get suggested retention actions.</p></div>",
    unsafe_allow_html=True,
)

left, right = st.columns([3, 2], gap="large")

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
with left:
    with st.form("customer_form"):
        tab1, tab2, tab3 = st.tabs(["👤 Customer", "📄 Account & Billing", "🌐 Services"])

        with tab1:
            c1, c2 = st.columns(2)
            gender = c1.selectbox("Gender", ["Male", "Female"])
            senior = c2.selectbox("Senior Citizen", ["No", "Yes"])
            partner = c1.selectbox("Has Partner", ["Yes", "No"])
            dependents = c2.selectbox("Has Dependents", ["No", "Yes"])

        with tab2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            c1, c2 = st.columns(2)
            contract = c1.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = c2.selectbox("Paperless Billing", ["Yes", "No"])
            payment = c1.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = c2.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.5)
            st.caption("Total charges are estimated as tenure × monthly charges.")

        with tab3:
            c1, c2 = st.columns(2)
            phone = c1.selectbox("Phone Service", ["Yes", "No"])
            internet = c2.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            multiple = c1.selectbox("Multiple Lines", ["No", "Yes"])
            online_sec = c2.selectbox("Online Security", ["No", "Yes"])
            online_bkp = c1.selectbox("Online Backup", ["No", "Yes"])
            device_prot = c2.selectbox("Device Protection", ["No", "Yes"])
            tech = c1.selectbox("Tech Support", ["No", "Yes"])
            stream_tv = c2.selectbox("Streaming TV", ["No", "Yes"])
            stream_mov = c1.selectbox("Streaming Movies", ["No", "Yes"])
            st.caption(
                "If Phone Service or Internet Service is 'No', the dependent "
                "add-ons are set automatically to match the dataset rules."
            )

        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

# ----------------------------------------------------------------------------
# Result panel
# ----------------------------------------------------------------------------
with right:
    if not submitted:
        st.markdown(
            '<div class="card"><div class="empty"><div class="big">🔮</div>'
            '<div class="t">No prediction yet</div>'
            '<div class="d">Fill in the customer details and click '
            "<b>Predict Churn Risk</b> to see the result.</div></div></div>",
            unsafe_allow_html=True,
        )
    else:
        # Apply dataset rules
        if phone == "No":
            multiple = "No phone service"
        if internet == "No":
            online_sec = online_bkp = device_prot = tech = stream_tv = stream_mov = (
                "No internet service"
            )

        total = tenure * monthly

        row = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "SeniorCitizen": 1 if senior == "Yes" else 0,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone,
                    "MultipleLines": multiple,
                    "InternetService": internet,
                    "OnlineSecurity": online_sec,
                    "OnlineBackup": online_bkp,
                    "DeviceProtection": device_prot,
                    "TechSupport": tech,
                    "StreamingTV": stream_tv,
                    "StreamingMovies": stream_mov,
                    "Contract": contract,
                    "PaperlessBilling": paperless,
                    "PaymentMethod": payment,
                    "MonthlyCharges": monthly,
                    "TotalCharges": total,
                }
            ]
        )

        prob = float(model.predict_proba(row)[0, 1])
        pred = int(model.predict(row)[0])

        if prob < 0.30:
            level, badge, color = "Low Risk", "badge-low", "#22C55E"
        elif prob < 0.50:
            level, badge, color = "Medium Risk", "badge-med", "#F59E0B"
        else:
            level, badge, color = "High Risk", "badge-high", "#EF4444"

        deg = prob * 360
        st.markdown(
            f'<div class="card"><h3>Churn Prediction</h3>'
            f'<div class="gauge" style="background: conic-gradient({color} {deg}deg, #1E2A47 0deg);">'
            f'<div class="gauge-inner"><div class="gauge-value">{prob:.1%}</div>'
            f'<div class="gauge-label">churn probability</div></div></div>'
            f'<div style="text-align:center;"><span class="badge {badge}">{level}</span>'
            f'<div class="muted" style="margin-top:10px;">Model decision: '
            f'<b>{"Likely to churn" if pred == 1 else "Likely to stay"}</b></div></div></div>',
            unsafe_allow_html=True,
        )

        # Risk factors (rule-based, from EDA findings)
        factors = []
        if contract == "Month-to-month":
            factors.append(("risk", "Month-to-month contract", "No long-term commitment. This group churns the most."))
        else:
            factors.append(("safe", f"{contract} contract", "Long-term contracts are linked to much lower churn."))
        if tenure <= 6:
            factors.append(("risk", "Very new customer", "Most churn happens in the first months of the relationship."))
        elif tenure >= 36:
            factors.append(("safe", "Long-standing customer", "Customers with long tenure are less likely to leave."))
        if payment == "Electronic check":
            factors.append(("risk", "Pays by electronic check", "This payment method has the highest churn rate."))
        elif "automatic" in payment:
            factors.append(("safe", "Automatic payments", "Automatic payment users churn far less."))
        if internet == "Fiber optic":
            factors.append(("risk", "Fiber optic internet", "Fiber customers churn more than DSL customers."))
        if internet != "No" and tech == "No":
            factors.append(("risk", "No tech support", "Customers without tech support are more likely to leave."))
        elif tech == "Yes":
            factors.append(("safe", "Has tech support", "Tech support is linked to better retention."))
        if monthly >= 85:
            factors.append(("risk", "High monthly bill", "Higher monthly charges are linked to more churn."))

        items = "".join(
            f'<div class="factor"><div class="dot" style="background:{"#EF4444" if k == "risk" else "#22C55E"};"></div>'
            f'<div><div class="t">{t}</div><div class="d">{d}</div></div></div>'
            for k, t, d in factors
        )
        st.markdown(
            f'<div class="card"><h3>Key Risk Factors</h3>{items}'
            f'<div class="muted" style="margin-top:10px;">Based on patterns found in the exploratory analysis, '
            f"not on individual model coefficients.</div></div>",
            unsafe_allow_html=True,
        )

        # Suggested actions
        actions = []
        if contract == "Month-to-month":
            actions.append("Offer a discount or bonus service for switching to a 1-year or 2-year plan.")
        if payment == "Electronic check":
            actions.append("Offer a small cashback for moving to automatic bank or card payments.")
        if internet != "No" and tech == "No":
            actions.append("Offer a free trial or bundle of Tech Support.")
        if internet == "Fiber optic":
            actions.append("Check fiber service quality and pricing for this customer.")
        if tenure <= 6:
            actions.append("Schedule an onboarding follow-up call.")
        if not actions:
            actions.append("No urgent action needed. Keep monitoring with regular scoring.")
        if pred == 1 and len(actions) < 2:
            actions.append("Contact the customer proactively with a tailored retention offer.")

        action_items = "".join(
            f'<div class="factor"><div class="dot" style="background:#6C63FF;"></div>'
            f'<div><div class="d" style="color:#C9D1EA;">{a}</div></div></div>'
            for a in actions
        )
        st.markdown(
            f'<div class="card"><h3>Suggested Retention Actions</h3>{action_items}</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "This output is an estimate, not a guarantee. The model uses class_weight='balanced', "
            "so it flags more customers as risky (precision is about 51%)."
        )
