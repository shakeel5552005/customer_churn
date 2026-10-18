"""
CUSTOMER CHURN PREDICTION & RETENTION ANALYTICS SYSTEM
Optimized: Loads Pre-Trained Model (.pkl) for Instant Start
"""
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings, os
import joblib
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
# STEP 0: RAW DATA GENERATOR (Same as train_and_save.py)
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
    
    cap = df["Monthly_Charges"].quantile(0.99)
    outliers = (df["Monthly_Charges"] > cap).sum()
    df.loc[df["Monthly_Charges"] > cap, "Monthly_Charges"] = cap
    report["outliers_capped"] = int(outliers)

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns
    missing_before = int(df.isna().sum().sum())
    for c in num_cols: df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].mode().shape[0] > 0 and c != "Churned": df[c] = df[c].fillna(df[c].mode()[0])
    report["missing_values_filled"] = missing_before

    df["Churned"] = df["Churned"].astype(str).str.strip().str.title()
    df["Churn_Flag"] = df["Churned"].map({"No":0,"Yes":1}).astype(int)

    df["Customer_Age"]       = df["Customer_Age"].clip(18, 100)
    df["Satisfaction_Score"] = df["Satisfaction_Score"].clip(1, 10)
    df["NPS_Score"]          = df["NPS_Score"].clip(0, 10)
    df["Tenure_Months"]      = df["Tenure_Months"].clip(0, 120)

    report["final_rows"]  = len(df)
    return df, report


# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
@st.cache_data(show_spinner="⚙️ Engineering features...")
def engineer_features(df_clean):
    df = df_clean.copy()
    df["Tenure_Group"] = pd.cut(df["Tenure_Months"], bins=[-1,6,18,36,100], labels=["New","Developing","Established","Loyal"]).astype(str)
    df["Monthly_Charge_Group"] = pd.cut(df["Monthly_Charges"], bins=[0,50,90,130,10000], labels=["Low","Medium","High","Very High"]).astype(str)
    df["Engagement_Level"] = pd.cut(df["Avg_Monthly_Usage_GB"], bins=[-1,30,90,10000], labels=["Low","Medium","High"]).astype(str)
    df["Support_Issue_Level"] = pd.cut(df["Support_Tickets"] + df["Complaints"], bins=[-1,1,3,1000], labels=["Low","Medium","High"]).astype(str)
    df["Payment_Risk"] = np.where((df["Late_Payments"]>=2) | (df["Auto_Pay"]=="No"), "High","Low")
    df["Customer_Value"] = df["Monthly_Charges"] * df["Tenure_Months"]
    df["Customer_Value_Tier"] = pd.qcut(df["Customer_Value"], q=3, labels=["Bronze","Silver","Gold"], duplicates="drop").astype(str)
    df["Satisfaction_Category"] = pd.cut(df["Satisfaction_Score"], bins=[0,4,7,11], labels=["Dissatisfied","Neutral","Satisfied"]).astype(str)
    df["Activity_Status"] = np.where(df["Last_Interaction_Days"]>90, "Inactive","Active")
    df["Complaint_Rate"] = df["Complaints"] / np.maximum(df["Tenure_Months"], 1)
    return df


# ============================================================
# LOAD PRE-TRAINED MODEL (.pkl)
# ============================================================
@st.cache_resource(show_spinner="📦 Loading pre-trained model...")
def load_saved_model():
    if os.path.exists("churn_model.pkl"):
        return joblib.load("churn_model.pkl")
    else:
        st.error("⚠️ 'churn_model.pkl' nahi mili! Pehle 'train_and_save.py' chalayein.")
        st.stop()

# Load pipeline
pipeline = load_saved_model()


def predict_one(pipeline, rec):
    X = pd.DataFrame([rec])[pipeline["feature_cols"]]
    for c in pipeline["cat_cols"]:
        le = pipeline["encoders"][c]
        v  = str(X[c].iloc[0])
        X[c] = le.transform([v if v in le.classes_ else le.classes_[0]])
    m = pipeline["model"]
    if pipeline["uses_scaler"]:
        return m.predict_proba(pipeline["scaler"].transform(X))[0,1]
    return m.predict_proba(X.values)[0,1]


def kpi(label, value, color="#667eea", delta=None):
    return f"""<div class="kpi-card" style="border-left-color:{color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {f'<div class="kpi-delta">{delta}</div>' if delta else ''}</div>"""


# ============================================================
# RUN DATA PIPELINE (Fast now, no training)
# ============================================================
raw_df, source = load_raw_data()
clean_df, clean_report = clean_data(raw_df)
df = engineer_features(clean_df)


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
        "🤖 Model Info",
        "🎚️ Threshold Analysis",
        "🔮 Churn Prediction",
        "🎯 Retention Dashboard",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.78rem;color:#94a3b8;line-height:1.9;'>
    <b style='color:#cbd5e1;'>Dataset</b><br>
    Rows: {len(df):,} • Cols: {df.shape[1]}<br>
    Churn Rate: {df['Churn_Flag'].mean()*100:.1f}%<br><br>
    <b style='color:#cbd5e1;'>Active Model</b><br>
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
        st.markdown('<div class="section-title">⚖️ Target Balance</div>', unsafe_allow_html=True)
        fig = px.pie(values=[(df['Churn_Flag']==0).sum(), (df['Churn_Flag']==1).sum()],
                     names=["Active","Churned"], hole=0.55,
                     color_discrete_sequence=["#10b981","#ef4444"])
        fig.update_layout(height=260, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with b:
        st.markdown('<div class="section-title">📋 Dataset Summary</div>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            "Property": ["Raw Records","After Cleaning","Numeric Features","Categorical Features","Churned","Active"],
            "Value": [len(raw_df), len(df),
                      df.select_dtypes(include=np.number).shape[1],
                      df.select_dtypes(include="object").shape[1],
                      int((df['Churn_Flag']==1).sum()),
                      int((df['Churn_Flag']==0).sum())]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 2: DATA CLEANING REPORT
# ============================================================
elif page == "🧹 Data Cleaning Report":
    st.markdown('<div class="main-header"><h1>🧹 Data Cleaning Report</h1></div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi("Raw Rows", f"{len(raw_df):,}"), unsafe_allow_html=True)
    c2.markdown(kpi("Duplicates Removed", f"{clean_report['duplicates_removed']}", "#ef4444"), unsafe_allow_html=True)
    c3.markdown(kpi("Missing Filled", f"{clean_report['missing_values_filled']}", "#f59e0b"), unsafe_allow_html=True)
    c4.markdown(kpi("Clean Rows", f"{clean_report['final_rows']:,}", "#10b981"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">👀 Cleaned Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=400)


# ============================================================
# PAGE 3: CUSTOMER EXPLORER
# ============================================================
elif page == "🔍 Customer Explorer":
    st.markdown('<div class="main-header"><h1>🔍 Customer Explorer</h1></div>', unsafe_allow_html=True)
    
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
        tmin,tmax = st.slider("Tenure", int(df["Tenure_Months"].min()), int(df["Tenure_Months"].max()), (int(df["Tenure_Months"].min()), int(df["Tenure_Months"].max())))

    filt = df[
        df["Region"].isin(regs) & df["Customer_Segment"].isin(segs) &
        df["Subscription_Type"].isin(subs) & df["Contract_Type"].isin(cons) &
        df["Churned"].isin(churn_sel) & df["Satisfaction_Score"].between(smin,smax) &
        df["Tenure_Months"].between(tmin,tmax)
    ]

    st.markdown(f"### 📄 {len(filt):,} Customers Found")
    cols = ["Customer_ID","Customer_Age","Region","Customer_Segment","Contract_Type","Tenure_Months","Monthly_Charges","Churned"]
    st.dataframe(filt[cols].reset_index(drop=True), use_container_width=True, height=400)
    
    csv = filt[cols].to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv, "filtered.csv", "text/csv")


# ============================================================
# PAGE 4: CHURN DASHBOARD
# ============================================================
elif page == "📊 Churn Dashboard":
    st.markdown('<div class="main-header"><h1>📊 Churn Dashboard</h1></div>', unsafe_allow_html=True)
    total=len(df); active=(df["Churn_Flag"]==0).sum(); churned=(df["Churn_Flag"]==1).sum()
    
    r1 = st.columns(3)
    r1[0].markdown(kpi("Total Customers", f"{total:,}"), unsafe_allow_html=True)
    r1[1].markdown(kpi("Active", f"{active:,}", "#10b981"), unsafe_allow_html=True)
    r1[2].markdown(kpi("Churned", f"{churned:,}", "#ef4444"), unsafe_allow_html=True)
    
    r2 = st.columns(3)
    r2[0].markdown(kpi("Churn Rate", f"{churned/total*100:.1f}%", "#ef4444"), unsafe_allow_html=True)
    r2[1].markdown(kpi("Total Revenue", f"${df['Monthly_Charges'].sum():,.0f}", "#667eea"), unsafe_allow_html=True)
    r2[2].markdown(kpi("Revenue at Risk", f"${df[df['Churn_Flag']==1]['Monthly_Charges'].sum():,.0f}", "#f59e0b"), unsafe_allow_html=True)


# ============================================================
# PAGE 5: CHURN ANALYSIS
# ============================================================
elif page == "📈 Churn Analysis":
    st.markdown('<div class="main-header"><h1>📈 Churn Analysis</h1></div>', unsafe_allow_html=True)
    
    def bar_by(col,title):
        g = df.groupby(col)["Churn_Flag"].mean().mul(100).reset_index()
        g.columns=[col,"Churn %"]
        fig = px.bar(g.sort_values("Churn %", ascending=False), x=col, y="Churn %", color="Churn %", color_continuous_scale="Reds", text="Churn %")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(title=title, height=360, coloraxis_showscale=False)
        return fig

    c1,c2 = st.columns(2)
    c1.plotly_chart(bar_by("Contract_Type","Churn by Contract"), use_container_width=True)
    c2.plotly_chart(bar_by("Subscription_Type","Churn by Subscription"), use_container_width=True)
    c1,c2 = st.columns(2)
    c1.plotly_chart(bar_by("Payment_Method","Churn by Payment"), use_container_width=True)
    c2.plotly_chart(bar_by("Auto_Pay","Churn by Auto-Pay"), use_container_width=True)


# ============================================================
# PAGE 6: MODEL INFO (Replaces Model Comparison)
# ============================================================
elif page == "🤖 Model Info":
    st.markdown('<div class="main-header"><h1>🤖 Active Model Info</h1></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background:#eff6ff; border-left:4px solid #3b82f6; padding:16px 20px; border-radius:10px;">
    <b>🏆 Active Model:</b> <b>{pipeline['best_name']}</b><br>
    Yeh model pehle se train shuda (.pkl) hai. Isliye app turant load hoti hai aur koi wait nahi karna padta.
    </div>""", unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">🎯 Feature Importance</div>', unsafe_allow_html=True)
    m = pipeline["model"]
    if hasattr(m, "feature_importances_"):
        imp = pd.DataFrame({"Feature": pipeline["feature_cols"], "Importance": m.feature_importances_}).sort_values("Importance", ascending=True).tail(15)
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Viridis")
        fig.update_layout(height=500, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Logistic Regression model mein direct feature importance nahi hoti (coefficients use hotay hain).")


# ============================================================
# PAGE 7: THRESHOLD ANALYSIS (Simulated for single model)
# ============================================================
elif page == "🎚️ Threshold Analysis":
    st.markdown('<div class="main-header"><h1>🎚️ Threshold Analysis</h1><p>Test data par threshold ka asar</p></div>', unsafe_allow_html=True)
    
    # We need test data to show threshold. Let's create a quick split.
    from sklearn.model_selection import train_test_split
    X_test = clean_df[pipeline["feature_cols"]].copy()
    for c in pipeline["cat_cols"]:
        le = pipeline["encoders"][c]
        X_test[c] = X_test[c].astype(str).apply(lambda v: v if v in le.classes_ else le.classes_[0])
        X_test[c] = le.transform(X_test[c])
    y_test = clean_df["Churn_Flag"].values
    
    if pipeline["uses_scaler"]:
        X_input = pipeline["scaler"].transform(X_test)
    else:
        X_input = X_test.values
        
    y_prob = pipeline["model"].predict_proba(X_input)[:,1]
    
    th = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)
    y_pred = (y_prob >= th).astype(int)
    
    tn,fp,fn,tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
    
    c = st.columns(4)
    c[0].markdown(kpi("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}"), unsafe_allow_html=True)
    c[1].markdown(kpi("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.3f}", "#10b981"), unsafe_allow_html=True)
    c[2].markdown(kpi("False Positives", f"{fp}", "#f59e0b"), unsafe_allow_html=True)
    c[3].markdown(kpi("False Negatives", f"{fn}", "#ef4444"), unsafe_allow_html=True)


# ============================================================
# PAGE 8: CHURN PREDICTION
# ============================================================
elif page == "🔮 Churn Prediction":
    st.markdown('<div class="main-header"><h1>🔮 Churn Prediction</h1><p>Score a customer instantly</p></div>', unsafe_allow_html=True)

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
            "Customer_Gender":gender, "Customer_Age":age, "Region":region, "Customer_Segment":segment,
            "Subscription_Type":sub, "Contract_Type":contract, "Tenure_Months":tenure, "Monthly_Charges":monthly,
            "Payment_Method":payment, "Auto_Pay":autopay, "Internet_Service":internet, "Device_Count":devices,
            "Avg_Monthly_Usage_GB":usage, "Support_Tickets":tickets, "Complaints":complaints, "Late_Payments":late,
            "Last_Interaction_Days":interact, "Satisfaction_Score":sat, "NPS_Score":nps, "Discount_Percent":discount, "Referral_Count":referrals,
        }
        prob = predict_one(pipeline, rec)
        pred = "Churn" if prob >= 0.5 else "Retain"

        if prob < 0.25: risk,cls = "Low Risk","risk-low"
        elif prob < 0.5: risk,cls = "Medium Risk","risk-medium"
        elif prob < 0.75: risk,cls = "High Risk","risk-high"
        else: risk,cls = "Critical Risk","risk-critical"

        c1,c2,c3 = st.columns(3)
        c1.markdown(kpi("Churn Probability", f"{prob*100:.1f}%", "#ef4444" if prob>0.5 else "#10b981"), unsafe_allow_html=True)
        c2.markdown(kpi("Prediction", pred), unsafe_allow_html=True)
        c3.markdown(f"""<div class="kpi-card"><div class="kpi-label">Risk Level</div><div style="margin-top:8px;"><span class="{cls}">{risk}</span></div></div>""", unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">🎯 Suggested Retention Actions</div>', unsafe_allow_html=True)
        if prob >= 0.5:
            if contract == "Month-to-month": st.markdown('<div class="rec-card"><b>📅 Offer annual contract discount</b></div>', unsafe_allow_html=True)
            if sat <= 4: st.markdown('<div class="rec-card"><b>🤝 Assign customer success manager</b></div>', unsafe_allow_html=True)
            if complaints >= 2: st.markdown('<div class="rec-card"><b>🚨 Escalate complaint</b></div>', unsafe_allow_html=True)
            if autopay == "No": st.markdown('<div class="rec-card"><b>💳 Recommend auto-pay</b></div>', unsafe_allow_html=True)
        else:
            st.success("✅ Customer is likely to stay. Maintain standard engagement.")


# ============================================================
# PAGE 9: RETENTION DASHBOARD
# ============================================================
elif page == "🎯 Retention Dashboard":
    st.markdown('<div class="main-header"><h1>🎯 Retention Dashboard</h1></div>', unsafe_allow_html=True)
    
    # Score all quickly
    X_all = clean_df[pipeline["feature_cols"]].copy()
    for c in pipeline["cat_cols"]:
        le = pipeline["encoders"][c]
        X_all[c] = X_all[c].astype(str).apply(lambda v: v if v in le.classes_ else le.classes_[0])
        X_all[c] = le.transform(X_all[c])
    if pipeline["uses_scaler"]: X_in = pipeline["scaler"].transform(X_all)
    else: X_in = X_all.values
    
    df_s = df.copy()
    df_s["Churn_Probability"] = pipeline["model"].predict_proba(X_in)[:,1]
    df_s["Risk_Level"] = pd.cut(df_s["Churn_Probability"], bins=[-1,0.25,0.5,0.75,1.01], labels=["Low","Medium","High","Critical"]).astype(str)
    
    high = df_s[df_s["Risk_Level"].isin(["High","Critical"])]
    
    c1,c2 = st.columns(2)
    c1.markdown(kpi("High-Risk Customers", f"{len(high):,}", "#ef4444"), unsafe_allow_html=True)
    c2.markdown(kpi("Revenue at Risk", f"${high['Monthly_Charges'].sum():,.0f}/mo", "#f59e0b"), unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">📋 High-Risk Customer List</div>', unsafe_allow_html=True)
    cols = ["Customer_ID","Region","Customer_Segment","Contract_Type","Monthly_Charges","Risk_Level","Churn_Probability"]
    st.dataframe(high[cols].sort_values("Churn_Probability", ascending=False).head(100), use_container_width=True)
    
    csv = high[cols].to_csv(index=False).encode()
    st.download_button("⬇️ Download High-Risk List", csv, "high_risk.csv", "text/csv")