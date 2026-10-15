from pathlib import Path

import numpy as np
import polars as pl

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. paths and configuration

project_root = Path(__file__).resolve().parent.parent

data_path = project_root / "data" / "raw" / "crx.data"

output_dir = (
    project_root
    / "reports"
    / "a9_proxy_ablation_validation"
)

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

random_state = 42
n_splits = 5
threshold = 0.50


# 2. load and prepare data

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

df = df.with_columns(
    [
        pl.col(feature)
        .cast(pl.String)
        .str.strip_chars()
        .alias(feature)
        for feature in categorical_features
    ]
)

df = df.with_columns(
    pl.col("a16")
    .cast(pl.String)
    .str.strip_chars()
    .alias("a16")
)

df = df.filter(
    pl.col("a16").is_in(["+", "-"])
)

df = df.with_columns(
    (pl.col("a16") == "+")
    .cast(pl.Int8)
    .alias("credit_target")
)

df = df.with_columns(
    pl.col("a9").alias("a9_group")
)

x = df.select(all_features).to_pandas()

y = df["credit_target"].to_numpy()

a9_groups = df["a9_group"].to_numpy()


# 3. define experimental scenarios

scenarios = {
    "all_features": all_features,

    "without_a9": [
        feature
        for feature in all_features
        if feature != "a9"
    ],

    "without_a9_a8": [
        feature
        for feature in all_features
        if feature not in ["a9", "a8"]
    ],

    "without_a9_a10": [
        feature
        for feature in all_features
        if feature not in ["a9", "a10"]
    ],

    "without_a9_a11": [
        feature
        for feature in all_features
        if feature not in ["a9", "a11"]
    ],

    "without_a9_top_proxy": [
        feature
        for feature in all_features
        if feature not in [
            "a9",
            "a8",
            "a10",
            "a11",
        ]
    ],
}


# 4. define preprocessing and model

def build_model(feature_columns):

    numeric_columns = [
        feature
        for feature in numeric_features
        if feature in feature_columns
    ]

    categorical_columns = [
        feature
        for feature in categorical_features
        if feature in feature_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:
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

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    preprocessing = ColumnTransformer(
        transformers=transformers
    )

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


# 5. run paired cross-validation

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state,
)

fold_rows = []
oof_rows = []

print("A9 PROXY ABLATION VALIDATION")
print(f"Dataset shape: {df.shape}")
print(f"Approved: {int(y.sum())}")
print(f"Rejected: {int((y == 0).sum())}")
print(f"Cross-validation folds: {n_splits}")


for scenario_name, feature_columns in scenarios.items():

    print("\n" + "=" * 60)
    print(f"SCENARIO: {scenario_name}")
    print(f"Features: {feature_columns}")
    print("=" * 60)

    x_scenario = x[feature_columns]

    oof_probabilities = np.full(
        len(y),
        np.nan,
    )

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(x_scenario, y),
        start=1,
    ):

        x_train = x_scenario.iloc[train_idx]
        x_valid = x_scenario.iloc[valid_idx]

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        model = build_model(feature_columns)

        model.fit(
            x_train,
            y_train,
        )

        probabilities = model.predict_proba(
            x_valid
        )[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        oof_probabilities[valid_idx] = probabilities

        fold_auc = roc_auc_score(
            y_valid,
            probabilities,
        )

        fold_accuracy = accuracy_score(
            y_valid,
            predictions,
        )

        fold_balanced_accuracy = balanced_accuracy_score(
            y_valid,
            predictions,
        )

        fold_rows.append(
            {
                "scenario": scenario_name,
                "fold": fold,
                "roc_auc": fold_auc,
                "accuracy": fold_accuracy,
                "balanced_accuracy": fold_balanced_accuracy,
            }
        )

        print(
            f"Fold {fold}: "
            f"ROC-AUC = {fold_auc:.4f}, "
            f"Accuracy = {fold_accuracy:.4f}, "
            f"Balanced Accuracy = "
            f"{fold_balanced_accuracy:.4f}"
        )

    oof_predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    print(
        f"OOF ROC-AUC: "
        f"{roc_auc_score(y, oof_probabilities):.4f}"
    )

    for index in range(len(y)):
        oof_rows.append(
            {
                "scenario": scenario_name,
                "actual": int(y[index]),
                "probability_approved": float(
                    oof_probabilities[index]
                ),
                "predicted_approved": int(
                    oof_predictions[index]
                ),
                "a9_group": a9_groups[index],
            }
        )


# 6. summarize model performance

fold_df = pl.DataFrame(fold_rows)

performance_rows = []

for scenario_name in scenarios:

    scenario_df = fold_df.filter(
        pl.col("scenario") == scenario_name
    )

    performance_rows.append(
        {
            "scenario": scenario_name,
            "roc_auc_mean": scenario_df["roc_auc"].mean(),
            "roc_auc_std": scenario_df["roc_auc"].std(ddof=1),
            "accuracy_mean": scenario_df["accuracy"].mean(),
            "accuracy_std": scenario_df["accuracy"].std(ddof=1),
            "balanced_accuracy_mean": (
                scenario_df["balanced_accuracy"].mean()
            ),
            "balanced_accuracy_std": (
                scenario_df["balanced_accuracy"].std(ddof=1)
            ),
        }
    )

performance_summary = pl.DataFrame(
    performance_rows
)


# 7. calculate group-level metrics

oof_df = pl.DataFrame(oof_rows)

group_rows = []

for scenario_name in scenarios:

    scenario_df = oof_df.filter(
        pl.col("scenario") == scenario_name
    )

    for group_name in ["t", "f"]:

        group_df = scenario_df.filter(
            pl.col("a9_group") == group_name
        )

        actual = group_df["actual"].to_numpy()

        predicted = group_df[
            "predicted_approved"
        ].to_numpy()

        true_positive = int(
            np.sum(
                (actual == 1)
                & (predicted == 1)
            )
        )

        false_positive = int(
            np.sum(
                (actual == 0)
                & (predicted == 1)
            )
        )

        actual_approved = int(
            np.sum(actual == 1)
        )

        actual_rejected = int(
            np.sum(actual == 0)
        )

        group_rows.append(
            {
                "scenario": scenario_name,
                "a9_group": group_name,
                "n": len(actual),
                "historical_approval_rate": float(
                    actual.mean()
                ),
                "predicted_approval_rate": float(
                    predicted.mean()
                ),
                "tpr": (
                    true_positive / actual_approved
                    if actual_approved > 0
                    else None
                ),
                "fpr": (
                    false_positive / actual_rejected
                    if actual_rejected > 0
                    else None
                ),
            }
        )

group_summary = pl.DataFrame(group_rows)


# 8. calculate absolute gaps between groups

gap_rows = []

for scenario_name in scenarios:

    scenario_df = group_summary.filter(
        pl.col("scenario") == scenario_name
    )

    group_t = scenario_df.filter(
        pl.col("a9_group") == "t"
    )

    group_f = scenario_df.filter(
        pl.col("a9_group") == "f"
    )

    row = {
        "scenario": scenario_name,
    }

    for metric in [
        "historical_approval_rate",
        "predicted_approval_rate",
        "tpr",
        "fpr",
    ]:

        value_t = group_t[metric][0]
        value_f = group_f[metric][0]

        row[f"{metric}_gap_abs"] = abs(
            value_t - value_f
        )

        row[f"{metric}_gap_t_minus_f"] = (
            value_t - value_f
        )

    gap_rows.append(row)

gap_summary = pl.DataFrame(gap_rows)


# 9. calculate changes relative to the no-a9 baseline

baseline = performance_summary.filter(
    pl.col("scenario") == "without_a9"
)

baseline_auc = baseline["roc_auc_mean"][0]
baseline_balanced_accuracy = (
    baseline["balanced_accuracy_mean"][0]
)

comparison_rows = []

for scenario_name in scenarios:

    scenario_df = performance_summary.filter(
        pl.col("scenario") == scenario_name
    )

    comparison_rows.append(
        {
            "scenario": scenario_name,
            "roc_auc_mean": scenario_df["roc_auc_mean"][0],
            "roc_auc_change_vs_without_a9": (
                scenario_df["roc_auc_mean"][0]
                - baseline_auc
            ),
            "balanced_accuracy_mean": (
                scenario_df["balanced_accuracy_mean"][0]
            ),
            "balanced_accuracy_change_vs_without_a9": (
                scenario_df["balanced_accuracy_mean"][0]
                - baseline_balanced_accuracy
            ),
        }
    )

comparison_summary = pl.DataFrame(
    comparison_rows
)


# 10. save results

performance_summary.write_csv(
    output_dir / "ablation_performance_summary.csv"
)

group_summary.write_csv(
    output_dir / "ablation_group_metrics.csv"
)

gap_summary.write_csv(
    output_dir / "ablation_group_gaps.csv"
)

comparison_summary.write_csv(
    output_dir / "ablation_comparison_vs_without_a9.csv"
)


# 11. display results

print("\nPERFORMANCE SUMMARY")
print(performance_summary)

print("\nGROUP METRICS")
print(group_summary)

print("\nGROUP GAPS")
print(gap_summary)

print("\nCOMPARISON VS WITHOUT A9")
print(comparison_summary)

print("\nRESULTS SAVED TO")
print(output_dir)

print("\nAnalysis completed.")