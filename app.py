import os
import joblib
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix)

st.title("Dry Bean Classifier")
st.markdown(
    "Upload the Dry Bean test data (CSV) to see how well each trained model "
    "identifies the bean variety. Select a model, upload the file, and view the "
    "metrics and confusion matrix below."
)

model_files = sorted(
    f for f in os.listdir('model')
    if f.endswith('.joblib') and f != 'scaler.joblib'
)
model_names = [f.replace('.joblib', '') for f in model_files]

selected = st.selectbox("Select a model", model_names)


@st.cache_resource
def load(path):
    return joblib.load(path)


model = load(f'model/{selected}.joblib')
scaler = load('model/scaler.joblib')

uploaded = st.file_uploader("Upload test data (CSV)", type='csv')
if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded)

required = list(scaler.feature_names_in_) + ['Class']
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"The uploaded file is missing the following required columns: {', '.join(missing)}")
    st.stop()

X = df[scaler.feature_names_in_]
y = df['Class']

X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)
probs = model.predict_proba(X_scaled)

metrics = {
    'Accuracy': accuracy_score(y, preds),
    'AUC': roc_auc_score(y, probs, multi_class='ovr', average='weighted'),
    'Precision': precision_score(y, preds, average='weighted', zero_division=0),
    'Recall': recall_score(y, preds, average='weighted', zero_division=0),
    'F1': f1_score(y, preds, average='weighted', zero_division=0),
    'MCC': matthews_corrcoef(y, preds),
}

cols = st.columns(len(metrics))
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.3f}")

cm = confusion_matrix(y, preds, labels=model.classes_)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
