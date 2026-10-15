from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


# 1. Project paths
project_root = Path(__file__).resolve().parent.parent

file_path = project_root / "data" / "raw" / "crx.data"

output_dir = project_root / "reports" / "a9_stability"
output_dir.mkdir(parents=True, exist_ok=True)


# 2. Load dataset
columns = [f"A{i}" for i in range(1, 17)]

df = pl.read_csv(
    file_path,
    has_header=False,
    new_columns=columns,
    null_values="?",
)

numeric_features = [
    "A2", "A3", "A8", "A11", "A14", "A15"
]

categorical_features = [
    col for col in columns[:15]
    if col not in numeric_features
]

df = df.with_columns([
    pl.col(col).cast(pl.Float64, strict=False)
    for col in numeric_features
])

df = df.filter(pl.col("A16").is_in(["+", "-"]))

feature_columns = columns[:15]

X = df.select(feature_columns).to_numpy()

y = (
    df.select((pl.col("A16") == "+").cast(pl.Int64))
    .to_series()
    .to_numpy()
)

print("Dataset shape:", X.shape)
print("Approved:", int(y.sum()))
print("Rejected:", int(len(y) - y.sum()))


# 3. Define scenarios
scenarios = {
    "All features": feature_columns,
    "Without A9": [
        col for col in feature_columns if col != "A9"
    ],
    "A9 only": ["A9"],
}


# 4. Configure the same five folds for every scenario
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


# 5. Build a model for each feature scenario
def build_model(selected_features):
    selected_numeric = [
        col for col in selected_features
        if col in numeric_features
    ]

    selected_categorical = [
        col for col in selected_features
        if col in categorical_features
    ]

    transformers = []

    if selected_numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append((
            "numeric",
            numeric_pipeline,
            [
                selected_features.index(col)
                for col in selected_numeric
            ],
        ))

    if selected_categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            [
                selected_features.index(col)
                for col in selected_categorical
            ],
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])


# 6. Evaluate every scenario
fold_records = []
summary_records = []
fold_scores = {}

for scenario_name, selected_features in scenarios.items():
    print(f"\n{'=' * 55}")
    print("SCENARIO:", scenario_name)
    print("=" * 55)

    indices = [
        feature_columns.index(col)
        for col in selected_features
    ]

    X_selected = X[:, indices]
    model = build_model(selected_features)

    results = cross_validate(
        model,
        X_selected,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        error_score="raise",
    )

    roc_scores = results["test_roc_auc"]
    fold_scores[scenario_name] = roc_scores

    for fold_number in range(len(roc_scores)):
        record = {
            "scenario": scenario_name,
            "fold": fold_number + 1,
        }

        for metric in scoring:
            record[metric] = results[f"test_{metric}"][fold_number]

        fold_records.append(record)

        print(
            f"Fold {fold_number + 1}: "
            f"ROC-AUC = {roc_scores[fold_number]:.4f}"
        )

    summary = {
        "scenario": scenario_name,
        "number_of_features": len(selected_features),
    }

    for metric in scoring:
        values = results[f"test_{metric}"]
        summary[f"{metric}_mean"] = values.mean()
        summary[f"{metric}_std"] = values.std(ddof=1)

    summary_records.append(summary)

    print(
        f"Mean ROC-AUC: {roc_scores.mean():.4f}"
    )
    print(
        f"Fold standard deviation: {roc_scores.std(ddof=1):.4f}"
    )
    print(
        f"Minimum ROC-AUC: {roc_scores.min():.4f}"
    )
    print(
        f"Maximum ROC-AUC: {roc_scores.max():.4f}"
    )


# 7. Save fold-level results
fold_df = pd.DataFrame(fold_records)

fold_file = output_dir / "a9_fold_level_results.csv"
fold_df.to_csv(fold_file, index=False)


# 8. Save summary results
summary_df = pd.DataFrame(summary_records)

summary_file = output_dir / "a9_stability_summary.csv"
summary_df.to_csv(summary_file, index=False)


# 9. Compare ROC-AUC scores fold by fold
comparisons = [
    ("All features", "Without A9"),
    ("All features", "A9 only"),
    ("A9 only", "Without A9"),
]

comparison_records = []

print("\n" + "=" * 55)
print("PAIRED FOLD-LEVEL ROC-AUC COMPARISONS")
print("=" * 55)

for scenario_1, scenario_2 in comparisons:
    scores_1 = fold_scores[scenario_1]
    scores_2 = fold_scores[scenario_2]

    differences = scores_1 - scores_2

    record = {
        "comparison": f"{scenario_1} minus {scenario_2}",
        "mean_difference": differences.mean(),
        "std_difference": differences.std(ddof=1),
        "folds_positive": int((differences > 0).sum()),
        "folds_negative": int((differences < 0).sum()),
        "folds_equal": int((differences == 0).sum()),
    }

    comparison_records.append(record)

    print(f"\n{record['comparison']}")
    print(
        "Fold differences:",
        np.round(differences, 4).tolist(),
    )
    print(
        f"Mean difference: {record['mean_difference']:.4f}"
    )
    print(
        f"Sample standard deviation: "
        f"{record['std_difference']:.4f}"
    )
    print(
        f"Folds where first scenario performed better: "
        f"{record['folds_positive']}/5"
    )


comparison_df = pd.DataFrame(comparison_records)

comparison_file = output_dir / "a9_paired_comparisons.csv"
comparison_df.to_csv(comparison_file, index=False)


# 10. Final report
print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)

for _, row in summary_df.iterrows():
    print(
        f"{row['scenario']}: "
        f"ROC-AUC = {row['roc_auc_mean']:.4f} "
        f"+/- {row['roc_auc_std']:.4f} "
        f"(sample SD)"
    )

print("\nFiles saved:")
print(fold_file)
print(summary_file)
print(comparison_file)
