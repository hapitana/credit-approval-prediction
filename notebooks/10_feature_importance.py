from pathlib import Path

import polars as pl

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# 1. Define project paths
project_dir = Path(__file__).resolve().parent.parent

file_path = (
    project_dir
    / "data"
    / "raw"
    / "crx.data"
)

output_dir = project_dir / "reports" / "feature_importance"
output_dir.mkdir(parents=True, exist_ok=True)


# 2. Load the data
columns = [f"A{i}" for i in range(1, 17)]

df = pl.read_csv(
    file_path,
    has_header=False,
    new_columns=columns,
    null_values="?",
)

print("Dataset shape before filtering:", df.shape)


# 3. Define numerical and categorical features
numeric_features = [
    "A2",
    "A3",
    "A8",
    "A11",
    "A14",
    "A15",
]

categorical_features = [
    col
    for col in columns[:15]
    if col not in numeric_features
]


# 4. Convert numerical columns
df = df.with_columns([
    pl.col(col).cast(pl.Float64, strict=False)
    for col in numeric_features
])


# 5. Keep valid target labels
df = df.filter(
    pl.col("A16").is_in(["+", "-"])
)

print("Dataset shape after filtering:", df.shape)


# 6. Prepare features and target
feature_columns = columns[:15]

X_df = df.select(feature_columns)

y = (
    df.select(
        (pl.col("A16") == "+").cast(pl.Int64)
    )
    .to_series()
    .to_numpy()
)

X = X_df.to_numpy()


# 7. Map feature names to column positions
numeric_indices = [
    feature_columns.index(col)
    for col in numeric_features
]

categorical_indices = [
    feature_columns.index(col)
    for col in categorical_features
]


# 8. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# 9. Numerical preprocessing
numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median"),
    ),
    (
        "scaler",
        StandardScaler(),
    ),
])


# 10. Categorical preprocessing
categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent"),
    ),
    (
        "onehot",
        OneHotEncoder(handle_unknown="ignore"),
    ),
])


# 11. Combine preprocessing
preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_indices,
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_indices,
    ),
])


# 12. Build the model pipeline
model = Pipeline([
    (
        "preprocessor",
        preprocessor,
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])


# 13. Train the model
model.fit(X_train, y_train)


# 14. Evaluate the model
predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

metrics = {
    "roc_auc": roc_auc_score(
        y_test,
        probabilities,
    ),
    "accuracy": accuracy_score(
        y_test,
        predictions,
    ),
    "precision": precision_score(
        y_test,
        predictions,
        zero_division=0,
    ),
    "recall": recall_score(
        y_test,
        predictions,
        zero_division=0,
    ),
    "f1_score": f1_score(
        y_test,
        predictions,
        zero_division=0,
    ),
}


print("\nModel evaluation on the test set")

for metric_name, metric_value in metrics.items():
    print(
        f"{metric_name}: {metric_value:.4f}"
    )


# 15. Calculate permutation importance
importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=30,
    random_state=42,
    n_jobs=-1,
)


# 16. Build feature importance results
importance_df = pl.DataFrame({
    "feature": feature_columns,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std,
}).sort(
    "importance_mean",
    descending=True,
)


print("\nPermutation importance")
print(importance_df)


# 17. Save feature importance results
importance_path = (
    output_dir
    / "permutation_feature_importance.csv"
)

importance_df.write_csv(importance_path)


# 18. Save model evaluation results
metrics_df = pl.DataFrame({
    "metric": list(metrics.keys()),
    "value": list(metrics.values()),
})

metrics_path = (
    output_dir
    / "model_evaluation.csv"
)

metrics_df.write_csv(metrics_path)


# 19. Display the five most important features
print("\nTop 5 features")

print(
    importance_df.head(5)
)


# 20. Report output locations
print("\nFiles saved successfully:")

print(importance_path)
print(metrics_path)
