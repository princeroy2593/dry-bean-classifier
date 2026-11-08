# 1. Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
