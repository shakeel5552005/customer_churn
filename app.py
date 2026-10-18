"""
CUSTOMER CHURN PREDICTION & RETENTION ANALYTICS SYSTEM
Optimized for Virtual Environment + Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve, auc)

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Churn Analytics", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

# ============================================================
# PREMIUM CSS
# ============================================================
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg,#f5f7fa 0%,#e8ecf3 100%); }
.main-header {
    background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    padding: 28px; border-radius: 20px; color: white; margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(102,126,234,0.35);
}
.main-header h1 { margin:0; font-size:2.2rem; font-weight:800; }
.main-header p  { margin:6px 0 0 0; opacity:0.95; font-size:1rem; }
.kpi-card {
    background:white; padding:20px; border-radius:14px;
    box-shadow:0 4px 20px rgba(0,0,0,0.06);
    border-left:5px solid #667eea; height:100%;
}
.kpi-label { font-size:0.75rem; color:#6b7280; letter-spacing:0.8px;
             text-transform:uppercase; font-weight:700; }
.kpi-value { font-size:1.8rem; font-weight:800; color:#1f2937; margin-top:6px; }
.kpi-delta { font-size:0.75rem; color:#9ca3af; }
.section-title { font-size:1.35rem; font-weight:700; color:#1f2937;
                 border-bottom:3px solid #667eea; padding-bottom:6px;
                 margin:14px 0 12px; display:inline-block; }
.risk-low      { background:#d1fae5; color:#065f46; padding:6px 14px;
                 border-radius:20px; font-weight:700; display:inline-block;}
.risk-medium   { background:#fef3c7; color:#92400e; padding:6px 14px;
                 border-radius:20px; font-weight:700; display:inline-block;}
.risk-high     { background:#fed7aa; color:#9a3412; padding:6px 14px;
                 border-radius:20px; font-weight:700; display:inline-block;}
.risk-critical { background:#fecaca; color:#991b1b; padding:6px 14px;
                 border-radius:20px; font-weight:700; display:inline-block;}
.rec-card { background:white; padding:16px; border-radius:12px;
            border-left:5px solid #10b981; box-shadow:0 2px 12px rgba(0,0,0,0.05);
            margin-bottom:10px; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1e293b 0%,#0f172a 100%);
}
section[data-testid="stSidebar"] * { color:#e2e8f0; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# STEP 0: RAW DATA GENERATOR
# ============================================================
def generate_raw_dataset(n=1200, seed=42):
    np.random.seed(seed)
    regions   = ["North","South","East","West","Central"]
    segments  = ["Basic","Standard","Premium","Enterprise"]
    subs      = ["Basic","Standard","Premium"]
    contracts = ["Month-to-month","One year","Two year"]
    payments  = ["Credit Card","Bank Transfer","E-Wallet","Cash"]
    internet  = ["Fiber","DSL","Cable","None"]

    df = pd.DataFrame({
        "Customer_ID":       [f"CUST{str(i).zfill(5)}" for i in range(1,n+1)],
        "Customer_Name":     [f"Customer_{i}" for i in range(1,n+1)],
        "Signup_Date":       pd.date_range("2020-01-01", periods=n, freq="6h"),
        "Customer_Gender":   np.random.choice(["Male","Female"," male","FEMALE",""],
                                              n, p=[0.45,0.4,0.05,0.05,0.05]),
        "Customer_Age":      np.random.randint(18,75,n),
        "Region":            np.random.choice(regions+["north","EAST"], n,
                                              p=[0.19,0.19,0.19,0.19,0.19,0.03,0.02]),
        "Customer_Segment":  np.random.choice(segments, n, p=[0.35,0.3,0.25,0.1]),
        "Subscription_Type": np.random.choice(subs, n, p=[0.45,0.35,0.2]),
        "Contract_Type":     np.random.choice(contracts, n, p=[0.5,0.3,0.2]),
        "Tenure_Months":     np.random.randint(1,72,n),
        "Monthly_Charges":   np.round(np.random.uniform(20,150,n), 2),
        "Payment_Method":    np.random.choice(payments, n),
        "Auto_Pay":          np.random.choice(["Yes","No","yes","NO"], n,
                                              p=[0.5,0.35,0.1,0.05]),
        "Internet_Service":  np.random.choice(internet, n, p=[0.4,0.3,0.2,0.1]),
        "Device_Count":      np.random.randint(1,6,n),
        "Avg_Monthly_Usage_GB": np.round(np.random.uniform(1,200,n), 1),
        "Support_Tickets":   np.random.poisson(1.5,n),
        "Complaints":        np.random.poisson(0.8,n),
        "Late_Payments":     np.random.poisson(0.6,n),
        "Last_Interaction_Days": np.random.randint(1,180,n),
        "Satisfaction_Score":np.random.randint(1,11,n),
        "NPS_Score":         np.random.randint(0,11,n),
        "Discount_Percent":  np.random.choice([0,5,10,15,20], n, p=[0.5,0.15,0.15,0.1,0.1]),
        "Referral_Count":    np.random.poisson(0.9,n),
    })

    for c in ["Monthly_Charges","Satisfaction_Score","Customer_Age","Region","Payment_Method"]:
        idx = np.random.choice(df.index, size=int(0.03*n), replace=False)
        df.loc[idx, c] = np.nan

    dup = df.sample(15, random_state=1)
    df = pd.concat([df, dup], ignore_index=True)

    df.loc[df.sample(5, random_state=2).index, "Monthly_Charges"] = 9999

    score = (
        -1.2
        + (df["Contract_Type"]=="Month-to-month").astype(int)*1.5
        + (df["Contract_Type"]=="One year").astype(int)*0.3
        - (df["Tenure_Months"]/24)*0.9
        - (df["Satisfaction_Score"].fillna(5)-5)*0.35
        - (df["NPS_Score"]-5)*0.12
        + df["Complaints"]*0.55
        + df["Late_Payments"]*0.45
        + df["Support_Tickets"]*0.28
        - (df["Auto_Pay"].str.lower()=="yes").astype(int)*0.5
        + (df["Monthly_Charges"].fillna(80)/100)*0.6
        - df["Referral_Count"]*0.2
        + (df["Internet_Service"]=="Fiber").astype(int)*0.4
        - (df["Discount_Percent"]/100)*1.5
        + (df["Last_Interaction_Days"]/100)*0.3
        + np.random.normal(0,0.9,len(df))
    )
    prob = 1/(1+np.exp(-score))
    df["Churned"] = np.where(np.random.random(len(df))<prob, "Yes","No")
    return df


# ============================================================
# STEP 1: DATA LOADING
# ============================================================
@st.cache_data(show_spinner="📥 Loading raw dataset...")
def load_raw_data():
    for p in ["data/customer_churn.csv", "customer_churn.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p), "CSV file"
    return generate_raw_dataset(), "Synthetic (generated)"


# ============================================================
# STEP 2: DATA CLEANING
# ============================================================
@st.cache_data(show_spinner="🧹 Cleaning data...")
def clean_data(raw):
    df = raw.copy()
    report = {}

    before = len(df)
    df = df.drop_duplicates(subset=["Customer_ID"]).reset_index(drop=True)
    report["duplicates_removed"] = before - len(df)

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    cat_std = {
        "Customer_Gender": {"male":"Male","MALE":"Male","FEMALE":"Female","":"Unknown"},
        "Region":          {"north":"North","EAST":"East"},
        "Auto_Pay":        {"yes":"Yes","YES":"Yes","no":"No","NO":"No"},
    }
    for col, mapping in cat_std.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping).str.title()

    df["Customer_Age"]       = pd.to_numeric(df["Customer_Age"], errors="coerce")
    df["Monthly_Charges"]    = pd.to_numeric(df["Monthly_Charges"], errors="coerce")
    df["Satisfaction_Score"] = pd.to_numeric(df["Satisfaction_Score"], errors="coerce")
    if "Signup_Date" in df.columns:
        df["Signup_Date"] = pd.to_datetime(df["Signup_Date"], errors="coerce")

    cap = df["Monthly_Charges"].quantile(0.99)
    outliers = (df["Monthly_Charges"] > cap).sum()
    df.loc[df["Monthly_Charges"] > cap, "Monthly_Charges"] = cap
    report["outliers_capped"] = int(outliers)

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns
    missing_before = int(df.isna().sum().sum())
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].mode().shape[0] > 0 and c != "Churned":
            df[c] = df[c].fillna(df[c].mode()[0])
    report["missing_values_filled"] = missing_before

    df["Churned"] = df["Churned"].astype(str).str.strip().str.title()
    df["Churn_Flag"] = df["Churned"].map({"No":0,"Yes":1}).astype(int)

    df["Customer_Age"]       = df["Customer_Age"].clip(18, 100)
    df["Satisfaction_Score"] = df["Satisfaction_Score"].clip(1, 10)
    df["NPS_Score"]          = df["NPS_Score"].clip(0, 10)
    df["Tenure_Months"]      = df["Tenure_Months"].clip(0, 120)

    report["final_rows"]  = len(df)
    report["final_cols"]  = df.shape[1]
    return df, report


# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
@st.cache_data(show_spinner="⚙️ Engineering features...")
def engineer_features(df_clean):
    df = df_clean.copy()

    df["Tenure_Group"] = pd.cut(
        df["Tenure_Months"], bins=[-1,6,18,36,100],
        labels=["New Customer","Developing Customer","Established Customer","Loyal Customer"]
    ).astype(str)

    df["Monthly_Charge_Group"] = pd.cut(
        df["Monthly_Charges"], bins=[0,50,90,130,10000],
        labels=["Low","Medium","High","Very High"]
    ).astype(str)

    df["Engagement_Level"] = pd.cut(
        df["Avg_Monthly_Usage_GB"], bins=[-1,30,90,10000],
        labels=["Low","Medium","High"]
    ).astype(str)

    df["Support_Issue_Level"] = pd.cut(
        df["Support_Tickets"] + df["Complaints"], bins=[-1,1,3,1000],
        labels=["Low","Medium","High"]
    ).astype(str)

    df["Payment_Risk"] = np.where(
        (df["Late_Payments"]>=2) | (df["Auto_Pay"]=="No"), "High","Low")

    df["Customer_Value"] = df["Monthly_Charges"] * df["Tenure_Months"]
    df["Customer_Value_Tier"] = pd.qcut(
        df["Customer_Value"], q=3,
        labels=["Bronze","Silver","Gold"], duplicates="drop").astype(str)

    df["Satisfaction_Category"] = pd.cut(
        df["Satisfaction_Score"], bins=[0,4,7,11],
        labels=["Dissatisfied","Neutral","Satisfied"]).astype(str)

    df["Activity_Status"] = np.where(
        df["Last_Interaction_Days"]>90, "Inactive","Active")

    df["Complaint_Rate"] = df["Complaints"] / np.maximum(df["Tenure_Months"], 1)

    return df
#------------------------------
def show_training_animation():
    """Beautiful loading animation while models train."""
    placeholder = st.empty()
    
    placeholder.markdown("""
    <style>
    @keyframes pulse-brain {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.15); opacity: 0.8; }
    }
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    @keyframes dot-bounce {
        0%, 80%, 100% { transform: scale(0); opacity: 0.3; }
        40% { transform: scale(1); opacity: 1; }
    }
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    @keyframes progress-fill {
        0% { width: 0%; }
        30% { width: 35%; }
        60% { width: 70%; }
        90% { width: 92%; }
        100% { width: 100%; }
    }
    
    .training-overlay {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1e1b4b 100%);
        z-index: 999999;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 40px;
    }
    .training-box {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(102,126,234,0.3);
        border-radius: 30px;
        padding: 60px 80px;
        text-align: center;
        max-width: 600px;
        box-shadow: 0 20px 60px rgba(102,126,234,0.3);
    }
    .brain-icon {
        font-size: 5rem;
        animation: pulse-brain 1.5s ease-in-out infinite;
        display: inline-block;
        margin-bottom: 20px;
    }
    .spinner-ring {
        width: 80px;
        height: 80px;
        border: 5px solid rgba(102,126,234,0.2);
        border-top: 5px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    .training-title {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 20px 0 10px 0;
        letter-spacing: -0.5px;
    }
    .training-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        margin-bottom: 30px;
        font-weight: 400;
    }
    .dots {
        display: inline-flex;
        gap: 6px;
        margin-left: 6px;
    }
    .dots span {
        width: 8px;
        height: 8px;
        background: #667eea;
        border-radius: 50%;
        display: inline-block;
        animation: dot-bounce 1.4s ease-in-out infinite;
    }
    .dots span:nth-child(1) { animation-delay: 0s; }
    .dots span:nth-child(2) { animation-delay: 0.2s; }
    .dots span:nth-child(3) { animation-delay: 0.4s; }
    
    .progress-container {
        width: 100%;
        height: 8px;
        background: rgba(102,126,234,0.2);
        border-radius: 10px;
        overflow: hidden;
        margin: 25px 0 20px 0;
    }
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        background-size: 200% 100%;
        border-radius: 10px;
        animation: progress-fill 8s ease-out forwards, shimmer 2s linear infinite;
    }
    
    .steps-list {
        text-align: left;
        margin-top: 25px;
        color: #94a3b8;
        font-size: 0.9rem;
    }
    .steps-list .step {
        padding: 8px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .steps-list .step.active { color: #667eea; font-weight: 600; }
    .steps-list .step.done { color: #10b981; }
    .step-icon {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: rgba(102,126,234,0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
    }
    .step.active .step-icon {
        background: #667eea;
        animation: pulse-brain 1s ease-in-out infinite;
    }
    .step.done .step-icon {
        background: #10b981;
        color: white;
    }
    </style>
    
    <div class="training-overlay">
        <div class="training-box">
            <div class="brain-icon">🤖</div>
            <div class="spinner-ring"></div>
            <div class="training-title">
                Training Models
                <span class="dots">
                    <span></span><span></span><span></span>
                </span>
            </div>
            <div class="training-subtitle">
                Please wait while we train 5 ML models. This happens only once.
            </div>
            
            <div class="progress-container">
                <div class="progress-bar"></div>
            </div>
            
            <div class="steps-list">
                <div class="step active">
                    <div class="step-icon">⏳</div>
                    <span>Logistic Regression</span>
                </div>
                <div class="step active">
                    <div class="step-icon">⏳</div>
                    <span>K-Nearest Neighbors</span>
                </div>
                <div class="step active">
                    <div class="step-icon">⏳</div>
                    <span>Decision Tree</span>
                </div>
                <div class="step active">
                    <div class="step-icon">⏳</div>
                    <span>Random Forest</span>
                </div>
                <div class="step active">
                    <div class="step-icon">⏳</div>
                    <span>Gradient Boosting</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    return placeholder
#------------------------------

# ============================================================
# STEP 4: TRAIN MODELS
# ============================================================
FEATURES = [
    "Customer_Gender","Customer_Age","Region","Customer_Segment",
    "Subscription_Type","Contract_Type","Tenure_Months","Monthly_Charges",
    "Payment_Method","Auto_Pay","Internet_Service","Device_Count",
    "Avg_Monthly_Usage_GB","Support_Tickets","Complaints","Late_Payments",
    "Last_Interaction_Days","Satisfaction_Score","NPS_Score",
    "Discount_Percent","Referral_Count",
]

@st.cache_resource(show_spinner="🤖 Training 5 models...")
def train_models(df_clean):
    X = df_clean[FEATURES].copy()
    y = df_clean["Churn_Flag"].values

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        encoders[c] = le

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced"), True),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=9), True),
        "Decision Tree":       (DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42), False),
        "Random Forest":       (RandomForestClassifier(n_estimators=100, max_depth=10,
                                                       class_weight="balanced", random_state=42, n_jobs=-1), False),
        "Gradient Boosting":   (GradientBoostingClassifier(n_estimators=80, random_state=42), False),
    }

    results = {}
    for name,(mdl,uses_scaler) in models.items():
        if uses_scaler:
            mdl.fit(X_tr_s, y_tr); y_p = mdl.predict(X_te_s); y_pr = mdl.predict_proba(X_te_s)[:,1]
        else:
            mdl.fit(X_tr, y_tr);   y_p = mdl.predict(X_te);   y_pr = mdl.predict_proba(X_te)[:,1]

        results[name] = {
            "model": mdl, "uses_scaler": uses_scaler,
            "y_pred": y_p, "y_proba": y_pr,
            "accuracy":  accuracy_score(y_te, y_p),
            "precision": precision_score(y_te, y_p, zero_division=0),
            "recall":    recall_score(y_te, y_p, zero_division=0),
            "f1":        f1_score(y_te, y_p, zero_division=0),
            "roc_auc":   roc_auc_score(y_te, y_pr),
            "cm":        confusion_matrix(y_te, y_p),
            "report":    classification_report(y_te, y_p, output_dict=True, zero_division=0),
        }

    best = max(results, key=lambda k: (
        results[k]["f1"]*0.4 + results[k]["recall"]*0.4 + results[k]["roc_auc"]*0.2))

    return {
        "results": results, "best_name": best,
        "encoders": encoders, "scaler": scaler,
        "X_test": X_te, "y_test": y_te, "y_train": y_tr,
        "feature_cols": X.columns.tolist(), "cat_cols": cat_cols,
    }


def predict_one(pipeline, rec):
    X = pd.DataFrame([rec])[pipeline["feature_cols"]]
    for c in pipeline["cat_cols"]:
        le = pipeline["encoders"][c]
        v  = str(X[c].iloc[0])
        X[c] = le.transform([v if v in le.classes_ else le.classes_[0]])
    m = pipeline["results"][pipeline["best_name"]]["model"]
    if pipeline["results"][pipeline["best_name"]]["uses_scaler"]:
        return m.predict_proba(pipeline["scaler"].transform(X))[0,1]
    return m.predict_proba(X.values)[0,1]


def kpi(label, value, color="#667eea", delta=None):
    return f"""<div class="kpi-card" style="border-left-color:{color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {f'<div class="kpi-delta">{delta}</div>' if delta else ''}</div>"""


# ============================================================
# RUN PIPELINE
# ============================================================
raw_df, source = load_raw_data()
clean_df, clean_report = clean_data(raw_df)
df = engineer_features(clean_df)
pipeline = train_models(clean_df)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:8px 0 18px;'>
        <div style='font-size:2.5rem;'>📊</div>
        <div style='font-weight:700;color:#fff;font-size:1.05rem;'>Churn Analytics</div>
        <div style='font-size:0.72rem;color:#94a3b8;'>Retention Suite v1.0</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠 Home",
        "🧹 Data Cleaning Report",
        "🔍 Customer Explorer",
        "📊 Churn Dashboard",
        "📈 Churn Analysis",
        "🤖 Model Comparison",
        "🎚️ Threshold Analysis",
        "🔮 Churn Prediction",
        "🎯 Retention Dashboard",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.78rem;color:#94a3b8;line-height:1.9;'>
    <b style='color:#cbd5e1;'>Dataset</b><br>
    Source: {source}<br>
    Rows: {len(df):,} • Cols: {df.shape[1]}<br>
    Churn Rate: {df['Churn_Flag'].mean()*100:.1f}%<br><br>
    <b style='color:#cbd5e1;'>Best Model</b><br>
    {pipeline['best_name']}
    </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE 1: HOME
# ============================================================
if page == "🏠 Home":
    st.markdown('<div class="main-header"><h1>🎯 Customer Churn Prediction & Retention Analytics</h1>'
                '<p>End-to-end ML platform — Clean → Engineer → Analyze → Predict → Retain</p></div>',
                unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi("Total Customers", f"{len(df):,}"), unsafe_allow_html=True)
    c2.markdown(kpi("Churn Rate", f"{df['Churn_Flag'].mean()*100:.1f}%", "#ef4444"), unsafe_allow_html=True)
    c3.markdown(kpi("Active Customers", f"{(df['Churn_Flag']==0).sum():,}", "#10b981"), unsafe_allow_html=True)
    c4.markdown(kpi("Revenue at Risk", f"${df[df['Churn_Flag']==1]['Monthly_Charges'].sum():,.0f}", "#f59e0b"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    a,b = st.columns([1.15, 1])

    with a:
        st.markdown('<div class="section-title">🎯 Business Problem</div>', unsafe_allow_html=True)
        st.markdown("""
        A subscription company is losing customers. This platform helps answer:
        - **Who** will churn?
        - **Why** (which factors)?
        - **How much revenue** is at risk?
        - **What action** retains them?
        """)

        st.markdown('<div class="section-title">🚀 Objective & Workflow</div>', unsafe_allow_html=True)
        st.markdown("""
        1. **Load** raw data → 2. **Clean** (dupes, missing, outliers, standardize)
        3. **Engineer** 9+ features → 4. **Analyze** churn drivers
        5. **Train** 5 models → 6. **Tune threshold** → 7. **Score & Retain**
        """)

        st.markdown('<div class="section-title">⚖️ Target Balance</div>', unsafe_allow_html=True)
        fig = px.pie(values=[(df['Churn_Flag']==0).sum(), (df['Churn_Flag']==1).sum()],
                     names=["Active","Churned"], hole=0.55,
                     color_discrete_sequence=["#10b981","#ef4444"])
        fig.update_layout(height=260, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with b:
        st.markdown('<div class="section-title">📋 Dataset Summary</div>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            "Property": ["Raw Records","After Cleaning","Numeric Features","Categorical Features",
                         "Missing Values (raw)","Duplicate Rows Removed","Churned","Active"],
            "Value": [len(raw_df), len(df),
                      df.select_dtypes(include=np.number).shape[1],
                      df.select_dtypes(include="object").shape[1],
                      int(raw_df.isna().sum().sum()),
                      clean_report["duplicates_removed"],
                      int((df['Churn_Flag']==1).sum()),
                      int((df['Churn_Flag']==0).sum())]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        rate = df['Churn_Flag'].mean()
        if rate < 0.3 or rate > 0.7:
            st.warning(f"⚠️ Target **imbalanced** ({rate*100:.1f}% churn) — using class-weight balancing & F1/Recall.")
        else:
            st.success("✅ Target reasonably balanced.")

        st.markdown('<div class="section-title">🔑 Key Features</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Feature": FEATURES}), use_container_width=True,
                     hide_index=True, height=300)


# ============================================================
# PAGE 2: DATA CLEANING REPORT
# ============================================================
elif page == "🧹 Data Cleaning Report":
    st.markdown('<div class="main-header"><h1>🧹 Data Cleaning Report</h1>'
                '<p>All cleaning steps performed before analysis & modeling</p></div>',
                unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi("Raw Rows", f"{len(raw_df):,}"), unsafe_allow_html=True)
    c2.markdown(kpi("Duplicates Removed", f"{clean_report['duplicates_removed']}", "#ef4444"), unsafe_allow_html=True)
    c3.markdown(kpi("Missing Filled", f"{clean_report['missing_values_filled']}", "#f59e0b"), unsafe_allow_html=True)
    c4.markdown(kpi("Clean Rows", f"{clean_report['final_rows']:,}", "#10b981"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Missing Values BEFORE Cleaning</div>', unsafe_allow_html=True)
    miss_raw = raw_df.isna().sum().reset_index()
    miss_raw.columns = ["Column","MissingCount"]
    miss_raw = miss_raw[miss_raw["MissingCount"]>0]
    if len(miss_raw):
        fig = px.bar(miss_raw, x="Column", y="MissingCount",
                     color="MissingCount", color_continuous_scale="Reds")
        fig.update_layout(height=320, margin=dict(t=20,b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No missing values found in raw data.")

    st.markdown('<div class="section-title">📊 Cleaning Operations Performed</div>', unsafe_allow_html=True)
    st.markdown(f"""
    | Step | Operation | Result |
    |------|-----------|--------|
    | 1 | Remove duplicates on Customer_ID | **{clean_report['duplicates_removed']}** rows removed |
    | 2 | Strip whitespace from categorical fields | Done |
    | 3 | Standardize categorical values (gender, region, auto_pay) | Done |
    | 4 | Data type correction (age, charges, satisfaction → numeric) | Done |
    | 5 | Outlier capping in Monthly_Charges (99th pct) | **{clean_report['outliers_capped']}** capped |
    | 6 | Missing values filled (median / mode) | **{clean_report['missing_values_filled']}** cells |
    | 7 | Numerical range validation (clip to sane limits) | Done |
    | 8 | Target encoding: Yes/No → 1/0 | Done |
    """)

    st.markdown('<div class="section-title">⚙️ Engineered Features</div>', unsafe_allow_html=True)
    st.markdown("""
    | Feature | Description |
    |---------|-------------|
    | `Tenure_Group` | New / Developing / Established / Loyal |
    | `Monthly_Charge_Group` | Low / Medium / High / Very High |
    | `Engagement_Level` | Based on average monthly usage |
    | `Support_Issue_Level` | Combined tickets + complaints |
    | `Payment_Risk` | High if late payments ≥ 2 or no auto-pay |
    | `Customer_Value` | Monthly_Charges × Tenure |
    | `Customer_Value_Tier` | Bronze / Silver / Gold (qcut) |
    | `Satisfaction_Category` | Dissatisfied / Neutral / Satisfied |
    | `Activity_Status` | Active / Inactive (last interaction > 90d) |
    | `Complaint_Rate` | Complaints / Tenure |
    """)

    st.markdown('<div class="section-title">👀 Cleaned Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=400)


# ============================================================
# PAGE 3: CUSTOMER EXPLORER
# ============================================================
elif page == "🔍 Customer Explorer":
    st.markdown('<div class="main-header"><h1>🔍 Customer Explorer</h1>'
                '<p>Filter, segment, and export records</p></div>', unsafe_allow_html=True)

    f1,f2,f3,f4 = st.columns(4)
    with f1:
        regs = st.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
        segs = st.multiselect("Segment", sorted(df["Customer_Segment"].unique()), default=sorted(df["Customer_Segment"].unique()))
    with f2:
        subs = st.multiselect("Subscription", sorted(df["Subscription_Type"].unique()), default=sorted(df["Subscription_Type"].unique()))
        cons = st.multiselect("Contract", sorted(df["Contract_Type"].unique()), default=sorted(df["Contract_Type"].unique()))
    with f3:
        churn_sel = st.multiselect("Churn Status", ["Yes","No"], default=["Yes","No"])
        smin,smax = st.slider("Satisfaction", 1, 10, (1,10))
    with f4:
        tmin,tmax = st.slider("Tenure (mo)", int(df["Tenure_Months"].min()), int(df["Tenure_Months"].max()),
                              (int(df["Tenure_Months"].min()), int(df["Tenure_Months"].max())))
        cmin,cmax = st.slider("Monthly $", float(df["Monthly_Charges"].min()), float(df["Monthly_Charges"].max()),
                              (float(df["Monthly_Charges"].min()), float(df["Monthly_Charges"].max())))

    filt = df[
        df["Region"].isin(regs) & df["Customer_Segment"].isin(segs) &
        df["Subscription_Type"].isin(subs) & df["Contract_Type"].isin(cons) &
        df["Churned"].isin(churn_sel) &
        df["Satisfaction_Score"].between(smin,smax) &
        df["Tenure_Months"].between(tmin,tmax) &
        df["Monthly_Charges"].between(cmin,cmax)
    ]

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi("Matching", f"{len(filt):,}"), unsafe_allow_html=True)
    c2.markdown(kpi("Churn %", f"{filt['Churn_Flag'].mean()*100:.1f}%" if len(filt) else "0%", "#ef4444"), unsafe_allow_html=True)
    c3.markdown(kpi("Avg Tenure", f"{filt['Tenure_Months'].mean():.0f} mo" if len(filt) else "—", "#10b981"), unsafe_allow_html=True)
    c4.markdown(kpi("Avg Monthly", f"${filt['Monthly_Charges'].mean():.0f}" if len(filt) else "—", "#f59e0b"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📄 Filtered Records</div>', unsafe_allow_html=True)
    cols = ["Customer_ID","Customer_Age","Customer_Gender","Region","Customer_Segment",
            "Subscription_Type","Contract_Type","Tenure_Months","Monthly_Charges",
            "Payment_Method","Auto_Pay","Satisfaction_Score","NPS_Score","Churned"]
    st.dataframe(filt[cols].reset_index(drop=True), use_container_width=True, height=420)

    csv = filt[cols].to_csv(index=False).encode()
    st.download_button("⬇️ Download Filtered CSV", csv, "filtered_customers.csv", "text/csv",
                       use_container_width=True)


# ============================================================
# PAGE 4: CHURN DASHBOARD
# ============================================================
elif page == "📊 Churn Dashboard":
    st.markdown('<div class="main-header"><h1>📊 Churn Dashboard</h1>'
                '<p>Executive KPIs & revenue impact</p></div>', unsafe_allow_html=True)

    total=len(df); active=(df["Churn_Flag"]==0).sum(); churned=(df["Churn_Flag"]==1).sum()
    churn_rev = df[df["Churn_Flag"]==1]["Monthly_Charges"].sum()
    tot_rev   = df["Monthly_Charges"].sum()

    r1 = st.columns(3)
    r1[0].markdown(kpi("Total Customers", f"{total:,}"), unsafe_allow_html=True)
    r1[1].markdown(kpi("Active Customers", f"{active:,}", "#10b981"), unsafe_allow_html=True)
    r1[2].markdown(kpi("Churned Customers", f"{churned:,}", "#ef4444"), unsafe_allow_html=True)

    r2 = st.columns(3)
    r2[0].markdown(kpi("Churn Rate", f"{churned/total*100:.1f}%", "#ef4444"), unsafe_allow_html=True)
    r2[1].markdown(kpi("Avg Monthly Charge", f"${df['Monthly_Charges'].mean():.2f}", "#667eea"), unsafe_allow_html=True)
    r2[2].markdown(kpi("Avg Tenure", f"{df['Tenure_Months'].mean():.0f} mo", "#667eea"), unsafe_allow_html=True)

    r3 = st.columns(3)
    r3[0].markdown(kpi("Avg Satisfaction", f"{df['Satisfaction_Score'].mean():.1f}/10", "#10b981"), unsafe_allow_html=True)
    r3[1].markdown(kpi("Total Monthly Revenue", f"${tot_rev:,.0f}", "#10b981"), unsafe_allow_html=True)
    r3[2].markdown(kpi("Churned Revenue", f"${churn_rev:,.0f}", "#ef4444",
                       f"{churn_rev/tot_rev*100:.1f}% of total"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    a,b = st.columns(2)
    with a:
        st.markdown('<div class="section-title">Churn Distribution</div>', unsafe_allow_html=True)
        fig = px.pie(values=[active,churned], names=["Active","Churned"], hole=0.6,
                     color_discrete_sequence=["#10b981","#ef4444"])
        fig.update_layout(height=330, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.markdown('<div class="section-title">Revenue Impact</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=["Retained","Churned"], y=[tot_rev-churn_rev, churn_rev],
            marker_color=["#10b981","#ef4444"], text=[f"${tot_rev-churn_rev:,.0f}", f"${churn_rev:,.0f}"],
            textposition="outside"))
        fig.update_layout(height=330, margin=dict(t=10,b=10), yaxis_title="Monthly Revenue ($)")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 5: CHURN ANALYSIS
# ============================================================
elif page == "📈 Churn Analysis":
    st.markdown('<div class="main-header"><h1>📈 Churn Analysis</h1>'
                '<p>All churn drivers & 15+ visualizations</p></div>', unsafe_allow_html=True)

    tabs = st.tabs(["📊 Categorical Drivers","📉 Numeric Drivers","🔥 Correlation","💡 Insights"])

    with tabs[0]:
        def bar_by(col,title):
            g = df.groupby(col)["Churn_Flag"].mean().mul(100).reset_index()
            g.columns=[col,"Churn %"]
            fig = px.bar(g.sort_values("Churn %", ascending=False),
                         x=col, y="Churn %", color="Churn %",
                         color_continuous_scale="Reds", text="Churn %")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(title=title, height=360, margin=dict(t=50,b=10),
                              coloraxis_showscale=False)
            return fig
        c1,c2 = st.columns(2)
        c1.plotly_chart(bar_by("Contract_Type","Churn by Contract"), use_container_width=True)
        c2.plotly_chart(bar_by("Subscription_Type","Churn by Subscription"), use_container_width=True)
        c1,c2 = st.columns(2)
        c1.plotly_chart(bar_by("Payment_Method","Churn by Payment"), use_container_width=True)
        c2.plotly_chart(bar_by("Auto_Pay","Churn by Auto-Pay"), use_container_width=True)
        c1,c2 = st.columns(2)
        c1.plotly_chart(bar_by("Customer_Segment","Churn by Segment"), use_container_width=True)
        c2.plotly_chart(bar_by("Region","Churn by Region"), use_container_width=True)
        c1,c2 = st.columns(2)
        c1.plotly_chart(bar_by("Internet_Service","Churn by Internet"), use_container_width=True)
        c2.plotly_chart(bar_by("Tenure_Group","Churn by Tenure Group"), use_container_width=True)
        c1,c2 = st.columns(2)
        c1.plotly_chart(bar_by("Satisfaction_Category","Churn by Satisfaction"), use_container_width=True)
        c2.plotly_chart(bar_by("Support_Issue_Level","Churn by Support Issues"), use_container_width=True)

    with tabs[1]:
        def hist_by(col,title):
            fig = px.histogram(df, x=col, color="Churned", nbins=30, barmode="overlay",
                               color_discrete_map={"Yes":"#ef4444","No":"#10b981"}, opacity=0.75)
            fig.update_layout(title=title, height=350, margin=dict(t=50,b=10))
            return fig
        c1,c2 = st.columns(2)
        c1.plotly_chart(hist_by("Monthly_Charges","Monthly Charges vs Churn"), use_container_width=True)
        c2.plotly_chart(hist_by("Tenure_Months","Tenure vs Churn"), use_container_width=True)
        c1,c2 = st.columns(2)
        c1.plotly_chart(hist_by("Satisfaction_Score","Satisfaction vs Churn"), use_container_width=True)
        c2.plotly_chart(hist_by("NPS_Score","NPS vs Churn"), use_container_width=True)
        c1,c2 = st.columns(2)
        c1.plotly_chart(hist_by("Support_Tickets","Support Tickets vs Churn"), use_container_width=True)
        c2.plotly_chart(hist_by("Late_Payments","Late Payments vs Churn"), use_container_width=True)
        c1,c2 = st.columns(2)
        c1.plotly_chart(hist_by("Complaints","Complaints vs Churn"), use_container_width=True)
        c2.plotly_chart(hist_by("Referral_Count","Referrals vs Churn"), use_container_width=True)

    with tabs[2]:
        st.markdown('<div class="section-title">Correlation Heatmap (Numeric)</div>', unsafe_allow_html=True)
        num = df.select_dtypes(include=np.number)
        corr = num.corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(height=680, margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.markdown('<div class="section-title">💡 Key Business Insights</div>', unsafe_allow_html=True)

        churn_rate = df["Churn_Flag"].mean()*100
        mtm = df[df["Contract_Type"]=="Month-to-month"]["Churn_Flag"].mean()*100
        autopay_yes = df[df["Auto_Pay"]=="Yes"]["Churn_Flag"].mean()*100
        autopay_no  = df[df["Auto_Pay"]=="No"]["Churn_Flag"].mean()*100
        low_sat = df[df["Satisfaction_Score"]<=4]["Churn_Flag"].mean()*100
        high_sat= df[df["Satisfaction_Score"]>=8]["Churn_Flag"].mean()*100
        comp_yes= df[df["Complaints"]>0]["Churn_Flag"].mean()*100
        comp_no = df[df["Complaints"]==0]["Churn_Flag"].mean()*100

        insights = [
            ("Overall Churn Rate", f"{churn_rate:.1f}%"),
            ("Month-to-month contract churn", f"{mtm:.1f}%"),
            ("Auto-pay = Yes churn", f"{autopay_yes:.1f}%"),
            ("Auto-pay = No churn", f"{autopay_no:.1f}%"),
            ("Low satisfaction (≤4) churn", f"{low_sat:.1f}%"),
            ("High satisfaction (≥8) churn", f"{high_sat:.1f}%"),
            ("Complaint = Yes churn", f"{comp_yes:.1f}%"),
            ("Complaint = No churn", f"{comp_no:.1f}%"),
            ("Monthly revenue from churned", f"${df[df['Churn_Flag']==1]['Monthly_Charges'].sum():,.0f}"),
        ]
        st.dataframe(pd.DataFrame(insights, columns=["Metric","Value"]),
                     use_container_width=True, hide_index=True)


# ============================================================
# PAGE 6: MODEL COMPARISON
# ============================================================
elif page == "🤖 Model Comparison":
    st.markdown('<div class="main-header"><h1>🤖 Model Comparison</h1>'
                '<p>5 models evaluated — F1 & Recall prioritized (not just accuracy)</p></div>',
                unsafe_allow_html=True)

    rows = []
    for name,r in pipeline["results"].items():
        rows.append({"Model":name, "Accuracy":r["accuracy"], "Precision":r["precision"],
                     "Recall":r["recall"], "F1":r["f1"], "ROC-AUC":r["roc_auc"]})
    tbl = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    tbl_styled = tbl.style.format({c:"{:.3f}" for c in tbl.columns if c!="Model"}).highlight_max(
        subset=["F1","Recall","ROC-AUC"], color="#d1fae5")
    st.dataframe(tbl_styled, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div style="background:#eff6ff; border-left:4px solid #3b82f6; padding:16px 20px;
                border-radius:10px; margin:12px 0; color:#1e40af;">
    <b>🏆 Selected Best Model:</b> <b>{pipeline['best_name']}</b><br>
    Chosen using a composite score of <b>40% F1 + 40% Recall + 20% ROC-AUC</b> — 
    NOT accuracy alone. In churn, <b>recall matters most</b> because missing an at-risk 
    customer (false negative) means <b>losing revenue permanently</b>, while a false positive 
    only costs a retention offer.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Model Metrics Comparison</div>', unsafe_allow_html=True)
    melted = tbl.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=400, margin=dict(t=20,b=10), yaxis_range=[0,1.05])
    st.plotly_chart(fig, use_container_width=True)

    sel = st.selectbox("🔎 Inspect model:", list(pipeline["results"].keys()),
                       index=list(pipeline["results"].keys()).index(pipeline["best_name"]))
    r = pipeline["results"][sel]

    c1,c2 = st.columns([1,1.3])
    with c1:
        st.markdown(f'<div class="section-title">Confusion Matrix — {sel}</div>', unsafe_allow_html=True)
        cm = r["cm"]
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                        x=["Pred Active","Pred Churn"], y=["Actual Active","Actual Churn"],
                        color_continuous_scale="Blues")
        fig.update_layout(height=350, margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(f'<div class="section-title">Classification Report — {sel}</div>', unsafe_allow_html=True)
        rep = pd.DataFrame(r["report"]).T
        st.dataframe(rep.style.format("{:.3f}"), use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">ROC Curves</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for name,rr in pipeline["results"].items():
            fpr,tpr,_ = roc_curve(pipeline["y_test"], rr["y_proba"])
            fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",
                                     name=f"{name} (AUC={rr['roc_auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",
                                 line=dict(dash="dash",color="gray")))
        fig.update_layout(height=420, xaxis_title="FPR", yaxis_title="TPR",
                          margin=dict(t=20,b=10), legend=dict(font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="section-title">Precision-Recall Curves</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for name,rr in pipeline["results"].items():
            p,rc,_ = precision_recall_curve(pipeline["y_test"], rr["y_proba"])
            fig.add_trace(go.Scatter(x=rc,y=p,mode="lines",name=f"{name} (AUC={auc(rc,p):.3f})"))
        fig.update_layout(height=420, xaxis_title="Recall", yaxis_title="Precision",
                          margin=dict(t=20,b=10), legend=dict(font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🎯 Feature Importance</div>', unsafe_allow_html=True)
    imp_model_name = "Random Forest" if "Random Forest" in pipeline["results"] else "Decision Tree"
    imp_model = pipeline["results"][imp_model_name]["model"]
    if hasattr(imp_model, "feature_importances_"):
        imp = pd.DataFrame({"Feature": pipeline["feature_cols"],
                            "Importance": imp_model.feature_importances_}).sort_values(
                            "Importance", ascending=True).tail(15)
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Viridis")
        fig.update_layout(height=500, margin=dict(t=20,b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 7: THRESHOLD ANALYSIS
# ============================================================
elif page == "🎚️ Threshold Analysis":
    st.markdown('<div class="main-header"><h1>🎚️ Threshold Analysis</h1>'
                '<p>Adjust decision threshold to balance precision & recall</p></div>',
                unsafe_allow_html=True)

    sel = st.selectbox("Model to analyze:", list(pipeline["results"].keys()),
                       index=list(pipeline["results"].keys()).index(pipeline["best_name"]))
    r = pipeline["results"][sel]
    th = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)

    y_true = pipeline["y_test"]; y_prob = r["y_proba"]
    y_pred = (y_prob >= th).astype(int)

    tn,fp,fn,tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    c = st.columns(5)
    c[0].markdown(kpi("Precision", f"{prec:.3f}"), unsafe_allow_html=True)
    c[1].markdown(kpi("Recall", f"{rec:.3f}", "#10b981"), unsafe_allow_html=True)
    c[2].markdown(kpi("F1-Score", f"{f1:.3f}", "#667eea"), unsafe_allow_html=True)
    c[3].markdown(kpi("False Positives", f"{fp}", "#f59e0b"), unsafe_allow_html=True)
    c[4].markdown(kpi("False Negatives", f"{fn}", "#ef4444"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Threshold vs Metrics</div>', unsafe_allow_html=True)
        ths = np.arange(0.05, 0.96, 0.02)
        rows=[]
        for t in ths:
            yp = (y_prob>=t).astype(int)
            rows.append({"Threshold":t,
                         "Precision":precision_score(y_true,yp,zero_division=0),
                         "Recall":recall_score(y_true,yp,zero_division=0),
                         "F1":f1_score(y_true,yp,zero_division=0)})
        mdf = pd.DataFrame(rows).melt(id_vars="Threshold",var_name="Metric",value_name="Value")
        fig = px.line(mdf, x="Threshold", y="Value", color="Metric",
                      color_discrete_sequence=["#3b82f6","#10b981","#f59e0b"])
        fig.add_vline(x=th, line_dash="dash", line_color="red")
        fig.update_layout(height=380, margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="section-title">Confusion Matrix @ Current Threshold</div>', unsafe_allow_html=True)
        cm = np.array([[tn,fp],[fn,tp]])
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                        x=["Pred Active","Pred Churn"], y=["Actual Active","Actual Churn"],
                        color_continuous_scale="Blues")
        fig.update_layout(height=380, margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div style="background:#eff6ff; border-left:4px solid #3b82f6; padding:16px 20px;
                border-radius:10px; margin:12px 0; color:#1e40af;">
    <b>💡 Business Guidance:</b> Lower threshold (e.g. <b>0.35</b>) → catches more at-risk 
    customers (↑ recall) at the cost of more false alarms (↓ precision). For churn, err on 
    the side of <b>lower thresholds</b> — losing a customer costs more than a wasted retention offer.
    </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE 8: CHURN PREDICTION
# ============================================================
elif page == "🔮 Churn Prediction":
    st.markdown('<div class="main-header"><h1>🔮 Churn Prediction</h1>'
                '<p>Score any customer & get a retention recommendation</p></div>',
                unsafe_allow_html=True)

    with st.form("predict_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            gender   = st.selectbox("Gender", ["Male","Female"])
            age      = st.slider("Age", 18, 75, 35)
            region   = st.selectbox("Region", sorted(df["Region"].unique()))
            segment  = st.selectbox("Segment", sorted(df["Customer_Segment"].unique()))
            sub      = st.selectbox("Subscription", sorted(df["Subscription_Type"].unique()))
            contract = st.selectbox("Contract", sorted(df["Contract_Type"].unique()))
            tenure   = st.slider("Tenure (months)", 1, 72, 12)
        with c2:
            monthly  = st.number_input("Monthly Charges ($)", 20.0, 200.0, 70.0)
            payment  = st.selectbox("Payment", sorted(df["Payment_Method"].unique()))
            autopay  = st.selectbox("Auto-Pay", ["Yes","No"])
            internet = st.selectbox("Internet Service", sorted(df["Internet_Service"].unique()))
            devices  = st.slider("Device Count", 1, 6, 2)
            usage    = st.slider("Avg Monthly Usage (GB)", 1.0, 200.0, 60.0)
        with c3:
            tickets  = st.slider("Support Tickets", 0, 10, 1)
            complaints = st.slider("Complaints", 0, 10, 0)
            late     = st.slider("Late Payments", 0, 10, 0)
            interact = st.slider("Last Interaction (days ago)", 1, 180, 30)
            sat      = st.slider("Satisfaction (1-10)", 1, 10, 7)
            nps      = st.slider("NPS (0-10)", 0, 10, 7)
            discount = st.selectbox("Discount %", [0,5,10,15,20])
            referrals= st.slider("Referrals", 0, 10, 0)

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True, type="primary")

    if submitted:
        rec = {
            "Customer_Gender":gender, "Customer_Age":age, "Region":region,
            "Customer_Segment":segment, "Subscription_Type":sub, "Contract_Type":contract,
            "Tenure_Months":tenure, "Monthly_Charges":monthly, "Payment_Method":payment,
            "Auto_Pay":autopay, "Internet_Service":internet, "Device_Count":devices,
            "Avg_Monthly_Usage_GB":usage, "Support_Tickets":tickets, "Complaints":complaints,
            "Late_Payments":late, "Last_Interaction_Days":interact,
            "Satisfaction_Score":sat, "NPS_Score":nps,
            "Discount_Percent":discount, "Referral_Count":referrals,
        }
        prob = predict_one(pipeline, rec)
        pred = "Churn" if prob >= 0.5 else "Retain"

        if prob < 0.25: risk,cls = "Low Risk","risk-low"
        elif prob < 0.5: risk,cls = "Medium Risk","risk-medium"
        elif prob < 0.75: risk,cls = "High Risk","risk-high"
        else: risk,cls = "Critical Risk","risk-critical"

        c1,c2,c3 = st.columns(3)
        c1.markdown(kpi("Churn Probability", f"{prob*100:.1f}%",
                        "#ef4444" if prob>0.5 else "#10b981"), unsafe_allow_html=True)
        c2.markdown(kpi("Prediction", pred), unsafe_allow_html=True)
        c3.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Risk Level</div>
            <div style="margin-top:8px;"><span class="{cls}">{risk}</span></div></div>""",
            unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob*100,
            title={"text":"Churn Risk Score"},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":"#ef4444" if prob>0.5 else "#10b981"},
                   "steps":[{"range":[0,25],"color":"#d1fae5"},
                            {"range":[25,50],"color":"#fef3c7"},
                            {"range":[50,75],"color":"#fed7aa"},
                            {"range":[75,100],"color":"#fecaca"}]}))
        fig.update_layout(height=280, margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">🚩 Main Risk Indicators</div>', unsafe_allow_html=True)
        indicators = []
        if contract == "Month-to-month": indicators.append("📅 Month-to-month contract (no lock-in)")
        if sat <= 4: indicators.append(f"😞 Low satisfaction ({sat}/10)")
        if complaints >= 2: indicators.append(f"⚠️ Multiple complaints ({complaints})")
        if late >= 2: indicators.append(f"💰 {late} late payments")
        if autopay == "No": indicators.append("💳 No auto-pay enrolled")
        if tickets >= 3: indicators.append(f"🎫 High support tickets ({tickets})")
        if nps <= 4: indicators.append(f"📉 Low NPS ({nps})")
        if interact > 90: indicators.append(f"😴 Inactive ({interact} days)")
        if monthly > 100: indicators.append(f"💵 High monthly charge (${monthly})")
        if tenure < 6: indicators.append(f"🆕 New customer ({tenure} months)")
        if usage < 20: indicators.append(f"📉 Low engagement ({usage} GB/mo)")
        if referrals == 0: indicators.append("🙅 No referrals")

        if indicators:
            for i in indicators: st.markdown(f"- {i}")
        else:
            st.success("✅ No major risk factors detected.")

        st.markdown('<div class="section-title">🎯 Suggested Retention Actions</div>', unsafe_allow_html=True)
        actions = []
        if contract == "Month-to-month": actions.append(("📅 Offer annual contract discount", "10-20% discount for 1-year lock-in"))
        if monthly > 100: actions.append(("💵 Offer lower-cost plan", "Right-size the plan to usage"))
        if sat <= 4: actions.append(("🤝 Assign customer success manager", "Personal outreach"))
        if complaints >= 2: actions.append(("🚨 Escalate complaint", "Priority resolution"))
        if late >= 2 or autopay == "No": actions.append(("💳 Recommend auto-pay", "5% auto-pay discount"))
        if usage < 20: actions.append(("📣 Re-engagement campaign", "Feature education"))
        if nps <= 4: actions.append(("📞 Feedback call", "Understand & fix issues"))
        if referrals == 0: actions.append(("🎁 Referral incentive", "Loyalty reward"))

        if not actions: actions.append(("✅ Maintain engagement", "Standard care is sufficient"))
        for a,d in actions:
            st.markdown(f'<div class="rec-card"><b>{a}</b><br>'
                        f'<span style="color:#6b7280;">{d}</span></div>',
                        unsafe_allow_html=True)


# ============================================================
# PAGE 9: RETENTION DASHBOARD
# ============================================================
elif page == "🎯 Retention Dashboard":
    st.markdown('<div class="main-header"><h1>🎯 Retention Dashboard</h1>'
                '<p>High-risk customers, revenue at risk & campaigns</p></div>',
                unsafe_allow_html=True)

    @st.cache_data(show_spinner="Scoring all customers...")
    def score_all(df_clean, pipeline_hash):
        X = df_clean[FEATURES].copy()
        for c in pipeline["cat_cols"]:
            le = pipeline["encoders"][c]
            X[c] = X[c].astype(str).apply(lambda v: v if v in le.classes_ else le.classes_[0])
            X[c] = le.transform(X[c])
        m = pipeline["results"][pipeline["best_name"]]["model"]
        if pipeline["results"][pipeline["best_name"]]["uses_scaler"]:
            p = m.predict_proba(pipeline["scaler"].transform(X))[:,1]
        else:
            p = m.predict_proba(X.values)[:,1]
        return p

    df_s = df.copy()
    df_s["Churn_Probability"] = score_all(clean_df, pipeline["best_name"])
    df_s["Risk_Level"] = pd.cut(df_s["Churn_Probability"],
        bins=[-1,0.25,0.5,0.75,1.01],
        labels=["Low","Medium","High","Critical"]).astype(str)

    high = df_s[df_s["Risk_Level"].isin(["High","Critical"])]
    rev_at_risk = high["Monthly_Charges"].sum()

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi("High-Risk Customers", f"{len(high):,}", "#ef4444"), unsafe_allow_html=True)
    c2.markdown(kpi("Revenue at Risk", f"${rev_at_risk:,.0f}/mo", "#f59e0b"), unsafe_allow_html=True)
    c3.markdown(kpi("Critical Risk", f"{(df_s['Risk_Level']=='Critical').sum()}", "#991b1b"), unsafe_allow_html=True)
    c4.markdown(kpi("High Risk", f"{(df_s['Risk_Level']=='High').sum()}", "#9a3412"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">High-Risk by Region</div>', unsafe_allow_html=True)
        g = high.groupby("Region").size().reset_index(name="Count")
        fig = px.bar(g.sort_values("Count",ascending=False), x="Region", y="Count",
                     color="Count", color_continuous_scale="Reds")
        fig.update_layout(height=350, margin=dict(t=20,b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="section-title">High-Risk by Segment</div>', unsafe_allow_html=True)
        g = high.groupby("Customer_Segment").size().reset_index(name="Count")
        fig = px.pie(g, names="Customer_Segment", values="Count", hole=0.5,
                     color_discrete_sequence=px.colors.sequential.Reds_r)
        fig.update_layout(height=350, margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">📣 Recommended Campaigns</div>', unsafe_allow_html=True)
    campaigns = [
        ("Month-to-month lock-in campaign", "High", "Target contract upgrades with 15% annual discount"),
        ("Auto-pay enrollment drive", "High", "5% discount for auto-pay enrollment"),
        ("Satisfaction recovery program", "Critical", "CSM outreach for satisfaction ≤4"),
        ("Complaint escalation hotline", "Critical", "Dedicated resolution team"),
        ("Re-engagement for inactive", "Medium", "Personalized feature education"),
        ("Referral rewards program", "Medium", "Incentivize referrals from loyal base"),
    ]
    c1,c2 = st.columns(2)
    for i,(name,prio,desc) in enumerate(campaigns):
        cls = "critical" if prio=="Critical" else "high" if prio=="High" else "medium"
        with (c1 if i%2==0 else c2):
            st.markdown(f'<div class="rec-card"><b>{name}</b> '
                        f'<span class="risk-{cls}">{prio}</span>'
                        f'<br><span style="color:#6b7280;">{desc}</span></div>',
                        unsafe_allow_html=True)

    st.markdown('<div class="section-title">📋 High-Risk Customer List</div>', unsafe_allow_html=True)
    cols = ["Customer_ID","Region","Customer_Segment","Contract_Type","Tenure_Months",
            "Monthly_Charges","Satisfaction_Score","NPS_Score","Risk_Level","Churn_Probability"]
    st.dataframe(high[cols].sort_values("Churn_Probability", ascending=False).head(100),
                 use_container_width=True, height=420)

    csv = high[cols].to_csv(index=False).encode()
    st.download_button("⬇️ Download High-Risk Customer List", csv,
                       "high_risk_customers.csv", "text/csv", use_container_width=True)

    st.markdown('<div class="section-title">💡 Evidence-Based Retention Recommendations</div>', unsafe_allow_html=True)
    recs = [
        ("Month-to-month → offer annual contract",
         "Month-to-month contracts show 2-3x higher churn than annual contracts."),
        ("Auto-pay incentive",
         "Auto-pay customers churn significantly less — offer 5% discount."),
        ("Satisfaction recovery for score ≤4",
         "Low-satisfaction customers churn at very high rates — priority CSM outreach."),
        ("Complaint escalation & resolution SLA",
         "Customers with ≥2 complaints have dramatically higher churn risk."),
        ("Targeted discounts for high-charge customers",
         "High monthly charges correlate with churn — offer right-sized plans."),
    ]
    for t,d in recs:
        st.markdown(f'<div class="rec-card"><b>✅ {t}</b><br>'
                    f'<span style="color:#6b7280;">{d}</span></div>', unsafe_allow_html=True)