from pathlib import Path

import numpy as np
import polars as pl

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
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

output_dir = project_root / "reports" / "a9_decision_impact"
output_dir.mkdir(parents=True, exist_ok=True)

features = [f"a{i}" for i in range(1, 16)]

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
    for feature in features
    if feature not in numeric_features
]

random_state = 42
n_splits = 5
threshold = 0.50


# 2. load and prepare data with polars

columns = features + ["a16"]

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

df = df.filter(pl.col("a16").is_not_null())

df = df.with_columns(
    (
        pl.col("a16")
        .cast(pl.String)
        .str.strip_chars()
        == "+"
    )
    .cast(pl.Int8)
    .alias("target")
)

df = df.with_columns(
    pl.col("a9")
    .cast(pl.String)
    .fill_null("Missing")
    .str.strip_chars()
    .alias("a9_group")
)

x_all = df.select(features).to_pandas()

x_without_a9 = df.select(
    [
        feature
        for feature in features
        if feature != "a9"
    ]
).to_pandas()

y = df["target"].to_numpy()
a9_groups = df["a9_group"].to_numpy()


# 3. build model pipeline

def build_model(feature_columns):
    numeric_columns = [
        column
        for column in numeric_features
        if column in feature_columns
    ]

    categorical_columns = [
        column
        for column in categorical_features
        if column in feature_columns
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

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return model


# 4. out-of-fold predictions

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state,
)

scenarios = {
    "all_features": x_all,
    "without_a9": x_without_a9,
}

oof_results = {}
overall_rows = []

print("=" * 65)
print("A9 DECISION IMPACT AUDIT")
print("=" * 65)

print(f"Dataset shape: {df.select(features).shape}")
print(f"Approved: {int(y.sum())}")
print(f"Rejected: {int((y == 0).sum())}")
print(f"Cross-validation folds: {n_splits}")
print(f"Decision threshold: {threshold:.2f}")


for scenario_name, x in scenarios.items():

    print("\n" + "=" * 65)
    print(f"SCENARIO: {scenario_name}")
    print("=" * 65)

    oof_probabilities = np.full(len(y), np.nan)

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(x, y),
        start=1,
    ):
        x_train = x.iloc[train_idx]
        x_valid = x.iloc[valid_idx]

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        model = build_model(x.columns.tolist())

        model.fit(x_train, y_train)

        probabilities = model.predict_proba(x_valid)[:, 1]

        oof_probabilities[valid_idx] = probabilities

        fold_auc = roc_auc_score(
            y_valid,
            probabilities,
        )

        print(
            f"Fold {fold}: ROC-AUC = {fold_auc:.4f}"
        )

    predicted_labels = (
        oof_probabilities >= threshold
    ).astype(int)

    oof_results[scenario_name] = pl.DataFrame(
        {
            "actual": y,
            "probability_approved": oof_probabilities,
            "predicted_approved": predicted_labels,
            "a9_group": a9_groups,
        }
    )

    overall_row = {
        "scenario": scenario_name,
        "roc_auc": roc_auc_score(y, oof_probabilities),
        "accuracy": accuracy_score(y, predicted_labels),
        "precision": precision_score(
            y,
            predicted_labels,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predicted_labels,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            predicted_labels,
            zero_division=0,
        ),
        "predicted_approval_rate": float(
            predicted_labels.mean()
        ),
    }

    overall_rows.append(overall_row)

    for metric_name, metric_value in overall_row.items():
        if metric_name != "scenario":
            print(f"{metric_name}: {metric_value:.4f}")


# 5. calculate metrics by a9 group

def calculate_group_metrics(group_df):
    actual = group_df["actual"].to_numpy()
    predicted = group_df["predicted_approved"].to_numpy()

    true_positive = int(
        np.sum((actual == 1) & (predicted == 1))
    )

    false_positive = int(
        np.sum((actual == 0) & (predicted == 1))
    )

    false_negative = int(
        np.sum((actual == 1) & (predicted == 0))
    )

    true_negative = int(
        np.sum((actual == 0) & (predicted == 0))
    )

    actual_approved = int(np.sum(actual == 1))
    actual_rejected = int(np.sum(actual == 0))

    tpr = (
        true_positive / actual_approved
        if actual_approved > 0
        else None
    )

    fpr = (
        false_positive / actual_rejected
        if actual_rejected > 0
        else None
    )

    fnr = (
        false_negative / actual_approved
        if actual_approved > 0
        else None
    )

    return {
        "n": len(group_df),
        "historical_approval_rate": float(actual.mean()),
        "predicted_approval_rate": float(predicted.mean()),
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
    }


group_rows = []

for scenario_name, results in oof_results.items():

    for group_df in results.partition_by("a9_group"):

        group_name = group_df["a9_group"][0]

        metrics = calculate_group_metrics(group_df)

        group_rows.append(
            {
                "scenario": scenario_name,
                "a9_group": group_name,
                **metrics,
            }
        )

group_summary = pl.DataFrame(group_rows)


# 6. compare group gaps

gap_rows = []

gap_metrics = [
    "historical_approval_rate",
    "predicted_approval_rate",
    "tpr",
    "fpr",
    "fnr",
]

for scenario_name in scenarios:

    subset = group_summary.filter(
        pl.col("scenario") == scenario_name
    )

    group_f = subset.filter(
        pl.col("a9_group") == "f"
    )

    group_t = subset.filter(
        pl.col("a9_group") == "t"
    )

    row = {"scenario": scenario_name}

    if group_f.height > 0 and group_t.height > 0:

        for metric_name in gap_metrics:

            value_f = group_f[metric_name][0]
            value_t = group_t[metric_name][0]

            if value_f is not None and value_t is not None:
                row[f"{metric_name}_gap_abs"] = abs(
                    value_t - value_f
                )

                row[f"{metric_name}_gap_t_minus_f"] = (
                    value_t - value_f
                )
            else:
                row[f"{metric_name}_gap_abs"] = None
                row[f"{metric_name}_gap_t_minus_f"] = None

    else:

        for metric_name in gap_metrics:
            row[f"{metric_name}_gap_abs"] = None
            row[f"{metric_name}_gap_t_minus_f"] = None

    gap_rows.append(row)

gap_summary = pl.DataFrame(gap_rows)


# 7. save results with polars

overall_summary = pl.DataFrame(overall_rows)

overall_path = output_dir / "overall_model_metrics.csv"
group_path = output_dir / "a9_group_metrics.csv"
gap_path = output_dir / "a9_group_gaps.csv"

overall_summary.write_csv(overall_path)
group_summary.write_csv(group_path)
gap_summary.write_csv(gap_path)


# 8. display results

print("\n" + "=" * 65)
print("OVERALL MODEL METRICS")
print("=" * 65)

print(overall_summary)

print("\n" + "=" * 65)
print("METRICS BY A9 GROUP")
print("=" * 65)

print(group_summary)

print("\n" + "=" * 65)
print("ABSOLUTE DIFFERENCES BETWEEN A9 GROUPS")
print("=" * 65)

print(gap_summary)

print("\n" + "=" * 65)
print("FILES SAVED")
print("=" * 65)

print(overall_path)
print(group_path)
print(gap_path)

print("\nAudit completed.")