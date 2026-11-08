# 1. Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

# 2. Data load
df = pd.read_excel('Dry_Bean_Dataset.xlsx')

# 3. CSV banao
df.to_csv('dry_bean.csv', index=False)

# 4. Data check
print("Shape:", df.shape)
print("\nClass counts:\n", df['Class'].value_counts())
print("\nTotal missing values:", df.isna().sum().sum())

# 5. X aur y
X = df.drop(columns=['Class'])
y = df['Class']

# 6. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 7. Scaling (train pe fit_transform, test pe sirf transform)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Check
print("\nX_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

models = {
    'Logistic_Regression': LogisticRegression(max_iter=1000),
    'Decision_Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive_Bayes': GaussianNB(),
    'SVM': SVC(probability=True, random_state=42),
}

os.makedirs('model', exist_ok=True)

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)

    results[name] = {
        'Accuracy': accuracy_score(y_test, preds),
        'AUC': roc_auc_score(y_test, probs, multi_class='ovr', average='weighted'),
        'Precision': precision_score(y_test, preds, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, preds, average='weighted', zero_division=0),
        'F1': f1_score(y_test, preds, average='weighted', zero_division=0),
        'MCC': matthews_corrcoef(y_test, preds),
    }

    joblib.dump(model, f'model/{name}.joblib')

joblib.dump(scaler, 'model/scaler.joblib')

test_data = X_test.copy()
test_data['Class'] = y_test
test_data.to_csv('test_data.csv', index=False)

results_df = pd.DataFrame(results).T.round(3)
print(results_df)
results_df.to_csv('model/metrics_summary.csv')
