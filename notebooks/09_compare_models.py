from pathlib import Path

import polars as pl

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# 1. Load the data
file_path = (
    Path(_file_).resolve().parent.parent
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

# Identify feature positions for ColumnTransformer.
numeric_indices = [
    feature_columns.index(col)
    for col in numeric_features
]

categorical_indices = [
    feature_columns.index(col)
    for col in categorical_features
]

# 4. Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# 5. Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# 6. Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_indices),
    ("categorical", categorical_pipeline, categorical_indices),
])

# 7. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        class_weight=None,
        n_jobs=-1,
    ),
}

# 8. Configure five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

# 9. Evaluate and compare models
for name, classifier in models.items():
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    print(f"\n=== {name} ===")

    for metric in scoring:
        scores = results[f"test_{metric}"]

        print(
            f"{metric}: {scores.mean():.4f} "
            f"+/- {scores.std():.4f}"
        )
