import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)


# 1. Load the data
file_path = "../data/raw/crx.data"
columns = [f"A{i}" for i in range(1, 17)]

df = pd.read_csv(
    file_path,
    header=None,
    names=columns,
    na_values="?",
)

# 2. Define features and target
numeric_features = ["A2", "A3", "A8", "A11", "A14", "A15"]
categorical_features = [
    col for col in columns[:15]
    if col not in numeric_features
]

# Ensure numerical features are converted to numeric values
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# + = approved (1), - = rejected (0)
df = df[df["A16"].isin(["+", "-"])].copy()
X = df[columns[:15]]
y = (df["A16"] == "+").astype(int)

print("Number of observations:", len(df))
print("\nTarget distribution:")
print(y.value_counts().sort_index().rename(index={
    0: "Rejected (-)",
    1: "Approved (+)",
}))

# 3. Split the data before preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 4. Numerical feature pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# 5. Categorical feature pipeline
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# 6. Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 7. Build the baseline model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000)),
])

# 8. Train the baseline model
model.fit(X_train, y_train)

# 9. Evaluate on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nNumber of training samples:", len(X_train))
print("Number of test samples:", len(X_test))

print("\nAccuracy:")
print(round(accuracy_score(y_test, y_pred), 4))

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Rejected (-)", "Approved (+)"],
    zero_division=0,
))

print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nROC-AUC:")
print(round(roc_auc_score(y_test, y_prob), 4))