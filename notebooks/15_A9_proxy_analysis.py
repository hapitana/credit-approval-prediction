from pathlib import Path

import numpy as np
import polars as pl

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. paths and configuration

project_root = Path(__file__).resolve().parent.parent

data_path = project_root / "data" / "raw" / "crx.data"

output_dir = project_root / "reports" / "a9_proxy_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

all_features = [f"a{i}" for i in range(1, 16)]

numeric_features = [
    "a2",
    "a3",
    "a8",
    "a11",
    "a14",
    "a15",
]

categorical_features = [
    feature
    for feature in all_features
    if feature not in numeric_features
]

proxy_features = [
    feature
    for feature in all_features
    if feature not in ["a9"]
]

random_state = 42
n_splits = 5


# 2. load and prepare data with polars

columns = all_features + ["a16"]

df = pl.read_csv(
    data_path,
    has_header=False,
    new_columns=columns,
    null_values="?",
    infer_schema_length=1000,
)

df = df.with_columns(
    [
        pl.col(feature).cast(pl.Float64, strict=False)
        for feature in numeric_features
    ]
)

df = df.filter(pl.col("a9").is_not_null())

df = df.with_columns(
    pl.when(
        pl.col("a9")
        .cast(pl.String)
        .str.strip_chars()
        == "t"
    )
    .then(1)
    .otherwise(0)
    .cast(pl.Int8)
    .alias("a9_target")
)

x = df.select(proxy_features).to_pandas()

y = df["a9_target"].to_numpy()


# 3. define preprocessing

numeric_columns = [
    feature
    for feature in numeric_features
    if feature in proxy_features
]

categorical_columns = [
    feature
    for feature in categorical_features
    if feature in proxy_features
]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessing = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ),
    ]
)


# 4. define models

models = {
    "logistic_regression": LogisticRegression(
        max_iter=2000,
        random_state=random_state,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_iter=150,
        l2_regularization=1.0,
        random_state=random_state,
    ),
}


# 5. stratified cross-validation

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state,
)

fold_rows = []
oof_rows = []

print("A9 PROXY ANALYSIS")
print(f"Dataset shape: {df.select(all_features).shape}")
print(f"A9 = t: {int(y.sum())}")
print(f"A9 = f: {int((y == 0).sum())}")
print(f"Number of proxy features: {len(proxy_features)}")
print(f"Cross-validation folds: {n_splits}")


for model_name, classifier in models.items():

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    oof_probabilities = np.full(len(y), np.nan)

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(x, y),
        start=1,
    ):

        x_train = x.iloc[train_idx]
        x_valid = x.iloc[valid_idx]

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        model = Pipeline(
            steps=[
                ("preprocessing", preprocessing),
                ("classifier", classifier),
            ]
        )

        model.fit(x_train, y_train)

        probabilities = model.predict_proba(x_valid)[:, 1]

        predictions = (probabilities >= 0.50).astype(int)

        oof_probabilities[valid_idx] = probabilities

        fold_metrics = {
            "model": model_name,
            "fold": fold,
            "roc_auc": roc_auc_score(
                y_valid,
                probabilities,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_valid,
                predictions,
            ),
            "accuracy": accuracy_score(
                y_valid,
                predictions,
            ),
            "precision": precision_score(
                y_valid,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_valid,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_valid,
                predictions,
                zero_division=0,
            ),
        }

        fold_rows.append(fold_metrics)

        print(
            f"Fold {fold}: "
            f"ROC-AUC = {fold_metrics['roc_auc']:.4f}, "
            f"Balanced Accuracy = "
            f"{fold_metrics['balanced_accuracy']:.4f}"
        )

    oof_predictions = (
        oof_probabilities >= 0.50
    ).astype(int)

    oof_metrics = {
        "model": model_name,
        "roc_auc": roc_auc_score(
            y,
            oof_probabilities,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y,
            oof_predictions,
        ),
        "accuracy": accuracy_score(
            y,
            oof_predictions,
        ),
        "precision": precision_score(
            y,
            oof_predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            oof_predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            oof_predictions,
            zero_division=0,
        ),
    }

    print("\nOut-of-fold metrics:")

    for metric_name, metric_value in oof_metrics.items():
        if metric_name != "model":
            print(f"{metric_name}: {metric_value:.4f}")

    for index in range(len(y)):
        oof_rows.append(
            {
                "model": model_name,
                "actual_a9": int(y[index]),
                "probability_a9_t": float(
                    oof_probabilities[index]
                ),
                "predicted_a9": int(
                    oof_predictions[index]
                ),
            }
        )


# 6. summarize cross-validation results

fold_results = pl.DataFrame(fold_rows)

summary_rows = []

for model_name in models:

    model_folds = fold_results.filter(
        pl.col("model") == model_name
    )

    summary_row = {"model": model_name}

    for metric_name in [
        "roc_auc",
        "balanced_accuracy",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:

        metric_values = model_folds[metric_name].to_numpy()

        summary_row[f"{metric_name}_mean"] = float(
            np.mean(metric_values)
        )

        summary_row[f"{metric_name}_std"] = float(
            np.std(metric_values, ddof=1)
        )

    summary_rows.append(summary_row)

summary = pl.DataFrame(summary_rows)

oof_predictions_df = pl.DataFrame(oof_rows)


# 7. save results

fold_path = output_dir / "a9_proxy_fold_results.csv"
summary_path = output_dir / "a9_proxy_summary.csv"
predictions_path = output_dir / "a9_proxy_oof_predictions.csv"

fold_results.write_csv(fold_path)
summary.write_csv(summary_path)
oof_predictions_df.write_csv(predictions_path)


# 8. display results

print("\n" + "=" * 60)
print("CROSS-VALIDATION SUMMARY")
print("=" * 60)

print(summary)

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(fold_path)
print(summary_path)
print(predictions_path)

print("\nA9 proxy analysis completed.")
