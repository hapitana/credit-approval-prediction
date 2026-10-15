from pathlib import Path


import numpy as np
import polars as pl


from sklearn.compose import ColumnTransformer
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
reports_path = project_root / "reports"


reports_path.mkdir(parents=True, exist_ok=True)


random_state = 42
n_splits = 5


thresholds = np.round(np.arange(0.20, 0.801, 0.05), 2)


target_column = "A16"
group_column = "A9"


feature_columns = [
    f"A{i}" for i in range(1, 16)
]


scenarios = {
    "all_features": feature_columns,
    "without_a9": [
        column
        for column in feature_columns
        if column != "A9"
    ],
    "without_a9_top_proxy": [
        column
        for column in feature_columns
        if column not in ["A9", "A8", "A10", "A11"]
    ],
}




# 2. load dataset


column_names = [
    f"A{i}" for i in range(1, 16)
] + ["A16"]


df = pl.read_csv(
    data_path,
    has_header=False,
    new_columns=column_names,
    infer_schema_length=0,
    null_values=["?"],
)


df = df.with_columns(
    pl.when(pl.col(target_column) == "+")
    .then(1)
    .otherwise(0)
    .cast(pl.Int64)
    .alias(target_column)
)


data = df.to_pandas()


y = data[target_column].astype(int)
group_values = data[group_column].astype(str)


print("A9 THRESHOLD SENSITIVITY ANALYSIS")
print(f"Dataset shape: {data.shape}")
print(f"Target approval rate: {y.mean():.4f}")
print(f"Thresholds: {thresholds.tolist()}")
print(f"Cross-validation folds: {n_splits}")




# 3. define preprocessing


def build_model(selected_features, frame):
    categorical_columns = []
    numerical_columns = []

    for column in selected_features:
        values = frame[column].dropna().astype(str)

        numeric_values = values.str.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
        )

        if len(values) > 0 and numeric_values.all():
            numerical_columns.append(column)
        else:
            categorical_columns.append(column)

    numerical_pipeline = Pipeline(
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
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
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
            ("preprocessor", preprocessor),
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


    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )


    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
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
            ("preprocessor", preprocessor),
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




# 4. generate out-of-fold probabilities


cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state,
)


scenario_predictions = {}
scenario_auc_scores = {}


for scenario_name, selected_features in scenarios.items():
    x = data[selected_features]


    oof_probabilities = np.zeros(len(data))
    fold_auc_scores = []


    for fold_number, (train_index, test_index) in enumerate(
        cv.split(x, y),
        start=1,
    ):
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]


        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]


        model = build_model(selected_features, data)


        model.fit(x_train, y_train)


        probabilities = model.predict_proba(x_test)[:, 1]


        oof_probabilities[test_index] = probabilities


        fold_auc = roc_auc_score(
            y_test,
            probabilities,
        )


        fold_auc_scores.append(fold_auc)


        print(
            f"{scenario_name} | "
            f"Fold {fold_number}: "
            f"ROC-AUC = {fold_auc:.4f}"
        )


    scenario_predictions[scenario_name] = oof_probabilities


    scenario_auc_scores[scenario_name] = {
        "roc_auc_mean": float(np.mean(fold_auc_scores)),
        "roc_auc_std": float(np.std(fold_auc_scores, ddof=1)),
        "roc_auc_oof": float(
            roc_auc_score(y, oof_probabilities)
        ),
    }




# 5. calculate metrics for each threshold


results = []


for scenario_name, probabilities in scenario_predictions.items():
    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)


        row = {
            "scenario": scenario_name,
            "threshold": float(threshold),
            "roc_auc_oof": scenario_auc_scores[
                scenario_name
            ]["roc_auc_oof"],
            "roc_auc_fold_mean": scenario_auc_scores[
                scenario_name
            ]["roc_auc_mean"],
            "roc_auc_fold_std": scenario_auc_scores[
                scenario_name
            ]["roc_auc_std"],
            "accuracy": accuracy_score(y, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y,
                predictions,
            ),
            "precision": precision_score(
                y,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y,
                predictions,
                zero_division=0,
            ),
            "overall_predicted_approval_rate": predictions.mean(),
        }


        group_metrics = {}


        for group_value in ["t", "f"]:
            group_mask = (
                group_values.to_numpy() == group_value
            )


            group_y = y.to_numpy()[group_mask]
            group_predictions = predictions[group_mask]


            true_positive = int(
                ((group_y == 1) & (group_predictions == 1)).sum()
            )


            false_positive = int(
                ((group_y == 0) & (group_predictions == 1)).sum()
            )


            false_negative = int(
                ((group_y == 1) & (group_predictions == 0)).sum()
            )


            true_negative = int(
                ((group_y == 0) & (group_predictions == 0)).sum()
            )


            approval_rate = float(
                group_predictions.mean()
            )


            tpr = (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative > 0
                else np.nan
            )


            fpr = (
                false_positive / (false_positive + true_negative)
                if false_positive + true_negative > 0
                else np.nan
            )


            group_metrics[group_value] = {
                "approval_rate": approval_rate,
                "tpr": tpr,
                "fpr": fpr,
            }


            row[f"a9_{group_value}_approval_rate"] = approval_rate
            row[f"a9_{group_value}_tpr"] = tpr
            row[f"a9_{group_value}_fpr"] = fpr


        row["approval_rate_gap"] = abs(
            group_metrics["t"]["approval_rate"]
            - group_metrics["f"]["approval_rate"]
        )


        row["tpr_gap"] = abs(
            group_metrics["t"]["tpr"]
            - group_metrics["f"]["tpr"]
        )


        row["fpr_gap"] = abs(
            group_metrics["t"]["fpr"]
            - group_metrics["f"]["fpr"]
        )


        results.append(row)




# 6. save detailed results


results_df = pl.DataFrame(results)


output_path = (
    reports_path
    / "a9_threshold_sensitivity_results.csv"
)


results_df.write_csv(output_path)


print("\nTHRESHOLD SENSITIVITY RESULTS")
print(
    results_df.select(
        [
            "scenario",
            "threshold",
            "roc_auc_oof",
            "balanced_accuracy",
            "overall_predicted_approval_rate",
            "a9_t_approval_rate",
            "a9_f_approval_rate",
            "approval_rate_gap",
            "tpr_gap",
            "fpr_gap",
        ]
    )
)


print(f"\nDetailed results saved to: {output_path}")




# 7. summarize the best balanced accuracy per scenario


summary_rows = []


for scenario_name in scenarios:
    scenario_df = results_df.filter(
        pl.col("scenario") == scenario_name
    )


    best_row = scenario_df.sort(
        "balanced_accuracy",
        descending=True,
    ).row(0, named=True)


    summary_rows.append(
        {
            "scenario": scenario_name,
            "best_threshold_by_balanced_accuracy": best_row[
                "threshold"
            ],
            "balanced_accuracy_at_that_threshold": best_row[
                "balanced_accuracy"
            ],
            "approval_rate_gap_at_that_threshold": best_row[
                "approval_rate_gap"
            ],
            "tpr_gap_at_that_threshold": best_row[
                "tpr_gap"
            ],
            "fpr_gap_at_that_threshold": best_row[
                "fpr_gap"
            ],
            "roc_auc_oof": best_row["roc_auc_oof"],
        }
    )


summary_df = pl.DataFrame(summary_rows)


summary_path = (
    reports_path
    / "a9_threshold_sensitivity_summary.csv"
)


summary_df.write_csv(summary_path)


print("\nSUMMARY AT BEST BALANCED ACCURACY THRESHOLD")
print(summary_df)


print(f"\nSummary saved to: {summary_path}")
print("\nAnalysis completed.")