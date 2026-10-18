
# train_and_save.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score

# 1. Aapka data generate karne wala function (app.py se copy karein)
def generate_raw_dataset(n=1200, seed=42):
    np.random.seed(seed)
    regions = ["North","South","East","West","Central"]
    segments = ["Basic","Standard","Premium","Enterprise"]
    subs = ["Basic","Standard","Premium"]
    contracts = ["Month-to-month","One year","Two year"]
    payments = ["Credit Card","Bank Transfer","E-Wallet","Cash"]
    internet = ["Fiber","DSL","Cable","None"]

    df = pd.DataFrame({
        "Customer_ID": [f"CUST{str(i).zfill(5)}" for i in range(1,n+1)],
        "Customer_Gender": np.random.choice(["Male","Female"," male","FEMALE",""], n, p=[0.45,0.4,0.05,0.05,0.05]),
        "Customer_Age": np.random.randint(18,75,n),
        "Region": np.random.choice(regions+["north","EAST"], n, p=[0.19,0.19,0.19,0.19,0.19,0.03,0.02]),
        "Customer_Segment": np.random.choice(segments, n, p=[0.35,0.3,0.25,0.1]),
        "Subscription_Type": np.random.choice(subs, n, p=[0.45,0.35,0.2]),
        "Contract_Type": np.random.choice(contracts, n, p=[0.5,0.3,0.2]),
        "Tenure_Months": np.random.randint(1,72,n),
        "Monthly_Charges": np.round(np.random.uniform(20,150,n), 2),
        "Payment_Method": np.random.choice(payments, n),
        "Auto_Pay": np.random.choice(["Yes","No","yes","NO"], n, p=[0.5,0.35,0.1,0.05]),
        "Internet_Service": np.random.choice(internet, n, p=[0.4,0.3,0.2,0.1]),
        "Device_Count": np.random.randint(1,6,n),
        "Avg_Monthly_Usage_GB": np.round(np.random.uniform(1,200,n), 1),
        "Support_Tickets": np.random.poisson(1.5,n),
        "Complaints": np.random.poisson(0.8,n),
        "Late_Payments": np.random.poisson(0.6,n),
        "Last_Interaction_Days": np.random.randint(1,180,n),
        "Satisfaction_Score": np.random.randint(1,11,n),
        "NPS_Score": np.random.randint(0,11,n),
        "Discount_Percent": np.random.choice([0,5,10,15,20], n, p=[0.5,0.15,0.15,0.1,0.1]),
        "Referral_Count": np.random.poisson(0.9,n),
    })
    # Churn logic...
    score = (-1.2 + (df["Contract_Type"]=="Month-to-month").astype(int)*1.5 + (df["Contract_Type"]=="One year").astype(int)*0.3
             - (df["Tenure_Months"]/24)*0.9 - (df["Satisfaction_Score"]-5)*0.35 - (df["NPS_Score"]-5)*0.12
             + df["Complaints"]*0.55 + df["Late_Payments"]*0.45 + df["Support_Tickets"]*0.28
             - (df["Auto_Pay"].str.lower()=="yes").astype(int)*0.5 + (df["Monthly_Charges"]/100)*0.6
             - df["Referral_Count"]*0.2 + (df["Internet_Service"]=="Fiber").astype(int)*0.4
             - (df["Discount_Percent"]/100)*1.5 + (df["Last_Interaction_Days"]/100)*0.3 + np.random.normal(0,0.9,n))
    prob = 1/(1+np.exp(-score))
    df["Churned"] = np.where(np.random.random(n)<prob, "Yes","No")
    return df

FEATURES = [
    "Customer_Gender","Customer_Age","Region","Customer_Segment","Subscription_Type","Contract_Type",
    "Tenure_Months","Monthly_Charges","Payment_Method","Auto_Pay","Internet_Service","Device_Count",
    "Avg_Monthly_Usage_GB","Support_Tickets","Complaints","Late_Payments","Last_Interaction_Days",
    "Satisfaction_Score","NPS_Score","Discount_Percent","Referral_Count",
]

print("⏳ Data generate aur process ho raha hai...")
raw_df = generate_raw_dataset()
df = raw_df.copy()
df["Churn_Flag"] = df["Churned"].map({"No":0, "Yes":1})

# Encoding
X = df[FEATURES].copy()
y = df["Churn_Flag"].values
cat_cols = X.select_dtypes(include="object").columns.tolist()
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c].astype(str))
    encoders[c] = le

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# 5 Models Train karein
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, random_state=42)
}

results = {}
best_name, best_score = None, -1

print("🤖 Models train ho rahe hain (Please wait)...")
for name, mdl in models.items():
    if name in ["Logistic Regression", "K-Nearest Neighbors"]:
        mdl.fit(X_tr_s, y_tr)
        y_pred = mdl.predict(X_te_s)
        y_proba = mdl.predict_proba(X_te_s)[:,1]
    else:
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)
        y_proba = mdl.predict_proba(X_te)[:,1]
        
    f1 = f1_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba)
    composite = f1*0.4 + rec*0.4 + auc*0.2
    
    results[name] = {"model": mdl, "composite": composite}
    if composite > best_score:
        best_score = composite
        best_name = name

# Sirf Best Model save karein (RAM bachane ke liye)
print(f"🏆 Best Model: {best_name}. Ise save kiya ja raha hai...")
final_pipeline = {
    "best_name": best_name,
    "model": results[best_name]["model"],
    "uses_scaler": best_name in ["Logistic Regression", "K-Nearest Neighbors"],
    "encoders": encoders,
    "scaler": scaler,
    "feature_cols": X.columns.tolist(),
    "cat_cols": cat_cols
}

# Model ko .pkl file mein save karein
joblib.dump(final_pipeline, "churn_model.pkl")
print("✅ Model 'churn_model.pkl' mein save ho gaya!")