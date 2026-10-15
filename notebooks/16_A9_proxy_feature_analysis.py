from pathlib import Path

import numpy as np
import polars as pl

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
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

output_dir = project_root / "reports" / "a9_proxy_feature_analysis"
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
top_proxy_count = 3


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
    pl.col("a9")
    .cast(pl.String)
    .str.strip_chars()
    .alias("a9_group")
)

df = df.filter(
    pl.col("a9_group").is_in(["t", "f"])
)

df = df.with_columns(
    (pl.col("a9_group") == "t")
    .cast(pl.Int8)
    .alias("a9_target")
)

df = df.filter(
    pl.col("a16").is_not_null()
)

df = df.with_columns(
    (
        pl.col("a16")
        .cast(pl.String)
        .str.strip_chars()
        == "+"
    )
    .cast(pl.Int8)
    .alias("credit_target")
)

x_proxy = df.select(
    [
        feature
        for feature in all_features
        if feature != "a9"
    ]
).to_pandas()

y_proxy = df["a9_target"].to_numpy()

x_credit = df.select(all_features).to_pandas()

y_credit = df["credit_target"].to_numpy()

a9_groups = df["a9_group"].to_numpy()


# 3. define preprocessing

def build_preprocessing(feature_columns):
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

    return ColumnTransformer(
        transformers=transformers
    )


def build_model(feature_columns, model_type="random_forest"):
    preprocessing = build_preprocessing(
        feature_columns
    )

    if model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )

    elif model_type == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2000,
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}"
        )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


# 4. identify proxy features using out-of-fold permutation importance

proxy_features = [
    feature
    for feature in all_features
    if feature != "a9"
]

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state,
)

importance_rows = []
proxy_fold_rows = []

print("A9 PROXY FEATURE ANALYSIS")
print(f"Dataset shape: {df.select(all_features).shape}")
print(f"A9 = t: {int(y_proxy.sum())}")
print(f"A9 = f: {int((y_proxy == 0).sum())}")
print(f"Proxy features: {len(proxy_features)}")
print(f"Cross-validation folds: {n_splits}")


for fold, (train_idx, valid_idx) in enumerate(
    cv.split(x_proxy, y_proxy),
    start=1,
):
    x_train = x_proxy.iloc[train_idx]
    x_valid = x_proxy.iloc[valid_idx]

    y_train = y_proxy[train_idx]
    y_valid = y_proxy[valid_idx]

    model = build_model(
        proxy_features,
        model_type="random_forest",
    )

    model.fit(x_train, y_train)

    probabilities = model.predict_proba(
        x_valid
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    fold_auc = roc_auc_score(
        y_valid,
        probabilities,
    )

    proxy_fold_rows.append(
        {
            "fold": fold,
            "roc_auc": fold_auc,
            "balanced_accuracy": balanced_accuracy_score(
                y_valid,
                predictions,
            ),
            "accuracy": accuracy_score(
                y_valid,
                predictions,
            ),
        }
    )

    importance = permutation_importance(
        model,
        x_valid,
        y_valid,
        scoring="roc_auc",
        n_repeats=10,
        random_state=random_state + fold,
        n_jobs=1,
    )

    for feature, mean_importance, std_importance in zip(
        proxy_features,
        importance.importances_mean,
        importance.importances_std,
    ):
        importance_rows.append(
            {
                "fold": fold,
                "feature": feature,
                "importance_mean": float(
                    mean_importance
                ),
                "importance_std": float(
                    std_importance
                ),
            }
        )

    print(
        f"Fold {fold}: "
        f"ROC-AUC = {fold_auc:.4f}, "
        f"Balanced Accuracy = "
        f"{balanced_accuracy_score(y_valid, predictions):.4f}"
    )


# 5. summarize proxy feature importance

importance_df = pl.DataFrame(importance_rows)

importance_summary = (
    importance_df
    .group_by("feature")
    .agg(
        pl.col("importance_mean")
        .mean()
        .alias("importance_mean"),
        pl.col("importance_mean")
        .std()
        .alias("importance_fold_std"),
        pl.col("importance_std")
        .mean()
        .alias("permutation_noise_std_mean"),
    )
    .sort(
        "importance_mean",
        descending=True,
    )
)

proxy_cv_df = pl.DataFrame(proxy_fold_rows)

proxy_cv_summary = pl.DataFrame(
    [
        {
            "roc_auc_mean": proxy_cv_df["roc_auc"].mean(),
            "roc_auc_std": proxy_cv_df["roc_auc"].std(ddof=1),
            "balanced_accuracy_mean": (
                proxy_cv_df["balanced_accuracy"].mean()
            ),
            "balanced_accuracy_std": (
                proxy_cv_df["balanced_accuracy"].std(ddof=1)
            ),
            "accuracy_mean": proxy_cv_df["accuracy"].mean(),
            "accuracy_std": proxy_cv_df["accuracy"].std(ddof=1),
        }
    ]
)

top_proxy_features = (
    importance_summary
    .filter(pl.col("importance_mean") > 0)
    .head(top_proxy_count)["feature"]
    .to_list()
)

print("\nPROXY MODEL CROSS-VALIDATION SUMMARY")
print(proxy_cv_summary)

print("\nPROXY FEATURE IMPORTANCE")
print(importance_summary)

print("\nTOP PROXY FEATURES")
print(top_proxy_features)


# 6. define credit approval scenarios

without_a9_features = [
    feature
    for feature in all_features
    if feature != "a9"
]

without_top_proxy_features = [
    feature
    for feature in without_a9_features
    if feature not in top_proxy_features
]

scenarios = {
    "all_features": all_features,
    "without_a9": without_a9_features,
    "without_a9_top_proxy": without_top_proxy_features,
    "a9_only": ["a9"],
}


# 7. evaluate credit approval models using out-of-fold predictions

credit_fold_rows = []
credit_oof_rows = []

for scenario_name, feature_columns in scenarios.items():

    print("\n" + "=" * 60)
    print(f"CREDIT SCENARIO: {scenario_name}")
    print(f"Features: {feature_columns}")
    print("=" * 60)

    x_scenario = x_credit[feature_columns]

    oof_probabilities = np.full(
        len(y_credit),
        np.nan,
    )

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(x_scenario, y_credit),
        start=1,
    ):
        x_train = x_scenario.iloc[train_idx]
        x_valid = x_scenario.iloc[valid_idx]

        y_train = y_credit[train_idx]
        y_valid = y_credit[valid_idx]

        model = build_model(
            feature_columns,
            model_type="logistic_regression",
        )

        model.fit(x_train, y_train)

        probabilities = model.predict_proba(
            x_valid
        )[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        oof_probabilities[valid_idx] = probabilities

        credit_fold_rows.append(
            {
                "scenario": scenario_name,
                "fold": fold,
                "roc_auc": roc_auc_score(
                    y_valid,
                    probabilities,
                ),
                "accuracy": accuracy_score(
                    y_valid,
                    predictions,
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y_valid,
                    predictions,
                ),
            }
        )

        print(
            f"Fold {fold}: "
            f"ROC-AUC = "
            f"{roc_auc_score(y_valid, probabilities):.4f}"
        )

    oof_predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    print(
        f"OOF ROC-AUC: "
        f"{roc_auc_score(y_credit, oof_probabilities):.4f}"
    )

    for index in range(len(y_credit)):
        credit_oof_rows.append(
            {
                "scenario": scenario_name,
                "actual": int(y_credit[index]),
                "probability_approved": float(
                    oof_probabilities[index]
                ),
                "predicted_approved": int(
                    oof_predictions[index]
                ),
                "a9_group": a9_groups[index],
            }
        )


# 8. summarize credit approval metrics

credit_fold_df = pl.DataFrame(credit_fold_rows)

credit_summary_rows = []

for scenario_name in scenarios:

    scenario_df = credit_fold_df.filter(
        pl.col("scenario") == scenario_name
    )

    credit_summary_rows.append(
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

credit_summary = pl.DataFrame(
    credit_summary_rows
)

credit_oof_df = pl.DataFrame(
    credit_oof_rows
)


# 9. calculate group-level decision metrics

group_rows = []

for scenario_name in scenarios:

    scenario_df = credit_oof_df.filter(
        pl.col("scenario") == scenario_name
    )

    for group_name in ["t", "f"]:

        group_df = scenario_df.filter(
            pl.col("a9_group") == group_name
        )

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
                "fnr": (
                    false_negative / actual_approved
                    if actual_approved > 0
                    else None
                ),
            }
        )

group_summary = pl.DataFrame(group_rows)


# 10. calculate group-level gaps

gap_rows = []

gap_metrics = [
    "historical_approval_rate",
    "predicted_approval_rate",
    "tpr",
    "fpr",
    "fnr",
]

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

    row = {"scenario": scenario_name}

    for metric_name in gap_metrics:

        value_t = group_t[metric_name][0]
        value_f = group_f[metric_name][0]

        if value_t is not None and value_f is not None:
            row[f"{metric_name}_gap_abs"] = abs(
                value_t - value_f
            )

            row[f"{metric_name}_gap_t_minus_f"] = (
                value_t - value_f
            )
        else:
            row[f"{metric_name}_gap_abs"] = None
            row[f"{metric_name}_gap_t_minus_f"] = None

    gap_rows.append(row)

gap_summary = pl.DataFrame(gap_rows)


# 11. save results

importance_path = (
    output_dir / "a9_proxy_feature_importance.csv"
)

proxy_cv_path = (
    output_dir / "a9_proxy_cv_summary.csv"
)

credit_summary_path = (
    output_dir / "credit_scenario_summary.csv"
)

group_path = (
    output_dir / "credit_group_metrics.csv"
)

gap_path = (
    output_dir / "credit_group_gaps.csv"
)

importance_summary.write_csv(importance_path)
proxy_cv_summary.write_csv(proxy_cv_path)
credit_summary.write_csv(credit_summary_path)
group_summary.write_csv(group_path)
gap_summary.write_csv(gap_path)


# 12. display results

print("\nPROXY FEATURE IMPORTANCE")
print(importance_summary)

print("\nPROXY MODEL SUMMARY")
print(proxy_cv_summary)

print("\nCREDIT SCENARIO SUMMARY")
print(credit_summary)

print("\nCREDIT GROUP METRICS")
print(group_summary)

print("\nCREDIT GROUP GAPS")
print(gap_summary)

print("\nFILES SAVED")
print(importance_path)
print(proxy_cv_path)
print(credit_summary_path)
print(group_path)
print(gap_path)

print("\nAnalysis completed.")