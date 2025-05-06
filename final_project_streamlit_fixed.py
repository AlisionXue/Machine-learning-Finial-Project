
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

st.set_page_config(page_title="Diabetes Readmission Prediction", layout="wide")
st.title("📊 Diabetes Readmission Prediction App")

@st.cache_data
def load_data():
    df = pd.read_csv("diabetic_data.csv")
    df['readmitted'] = df['readmitted'].map({'<30': 1, '>30': 0, 'NO': 0})
    df = df.replace('?', pd.NA).fillna(-1)
    return df

df = load_data()
st.success("✅ Data loaded successfully!")

if st.checkbox("🔍 Show raw data"):
    st.write(df.head())

features = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'age', 'insulin',
    'diabetesMed', 'change', 'gender', 'A1Cresult'
]
target = 'readmitted'
X = df[features].copy()
y = df[target]
categorical_features = ['age', 'insulin', 'diabetesMed', 'change', 'gender', 'A1Cresult']
X[categorical_features] = X[categorical_features].astype(str)

numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_medications']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

neg, pos = sum(y == 0), sum(y == 1)
scale = neg / pos if pos != 0 else 1  # Prevent ZeroDivisionError

model_choice = st.selectbox("📌 Select a model to train", ("Logistic Regression", "Random Forest", "XGBoost"))
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(
        scale_pos_weight=scale,
        eval_metric='logloss',
        use_label_encoder=False
    )
}
model = models[model_choice]

if st.button("🚀 Train Model"):
    st.info("Training in progress...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    st.subheader("📈 Evaluation Metrics")
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred)
    }
    st.write(pd.DataFrame(metrics, index=["Score"]).T)

    st.subheader("📊 Visualizations")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax[0], cmap='Blues')
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax[1])
    st.pyplot(fig)
