from pathlib import Path

import polars as pl

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


# 1. Load the data
file_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "crx.data"
)

columns = [f"A{i}" for i in range(1, 17)]

df = pl.read_csv(
    file_path,
    has_header=False,
    new_columns=columns,
    null_values="?",
)

# 2. Define numerical and categorical features
numeric_features = ["A2", "A3", "A8", "A11", "A14", "A15"]
categorical_features = [
    col for col in columns[:15]
    if col not in numeric_features
]

# Convert numerical features to numeric values.
# Invalid values become nulls and will be imputed later.
df = df.with_columns([
    pl.col(col).cast(pl.Float64, strict=False)
    for col in numeric_features
])

# Keep only valid target labels.
df = df.filter(pl.col("A16").is_in(["+", "-"]))

# 3. Prepare features and target
feature_columns = columns[:15]

X_df = df.select(feature_columns)

y = (
    df.select(
        (pl.col("A16") == "+").cast(pl.Int64)
    )
    .to_series()
    .to_numpy()
)

# Convert Polars features to a NumPy array for scikit-learn.
X = X_df.to_numpy()

# Map feature names to column positions.
numeric_indices = [
    feature_columns.index(col)
    for col in numeric_features
]

categorical_indices = [
    feature_columns.index(col)
    for col in categorical_features
]

# 4. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 5. Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# 6. Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# 7. Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_indices),
    ("categorical", categorical_pipeline, categorical_indices),
])

# 8. Build the Random Forest model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )),
])

# 9. Train the model
model.fit(X_train, y_train)

# 10. Evaluate using ROC-AUC
probabilities = model.predict_proba(X_test)[:, 1]

print(
    "Test ROC-AUC:",
    round(roc_auc_score(y_test, probabilities), 4),
)

# 11. Calculate permutation feature importance
importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=30,
    random_state=42,
    n_jobs=-1,
)

# Build a Polars DataFrame for the results
importance_df = pl.DataFrame({
    "feature": feature_columns,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std,
}).sort(
    "importance_mean",
    descending=True,
)

print("\nPermutation importance:")
print(importance_df)
