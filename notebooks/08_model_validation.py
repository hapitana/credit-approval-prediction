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
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


# ============================================================
# 1. PROJECT PATHS
# ============================================================

project_root = Path(__file__).resolve().parent.parent

file_path = project_root / "data" / "raw" / "crx.data"

output_dir = project_root / "reports" / "a9_robustness"
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

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

df = df.filter(
    pl.col("A16").is_in(["+", "-"])
)

feature_columns = columns[:15]

X = df.select(feature_columns).to_numpy()

y = (
    df.select(
        (pl.col("A16") == "+").cast(pl.Int64)
    )
    .to_series()
    .to_numpy()
)

print("Dataset shape:", X.shape)
print("Approved:", int(y.sum()))
print("Rejected:", int(len(y) - y.sum()))


# ============================================================
# 3. FEATURE SCENARIOS
# ============================================================

scenarios = {
    "All features": feature_columns,

    "Without A9": [
        col for col in feature_columns
        if col != "A9"
    ],

    "A9 only": ["A9"],
}


# ============================================================
# 4. CROSS-VALIDATION
# ============================================================

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


# ============================================================
# 5. MODEL DEFINITIONS
# ============================================================

model_names = [
    "Logistic Regression",
    "Random Forest",
    "HistGradientBoosting",
]


# ============================================================
# 6. BUILD PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor(selected_features):

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
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
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
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            [
                selected_features.index(col)
                for col in selected_categorical
            ],
        ))

    return ColumnTransformer(
        transformers=transformers
    )


# ============================================================
# 7. BUILD MODELS
# ============================================================

def build_model(model_name, selected_features):

    preprocessor = build_preprocessor(
        selected_features
    )

    if model_name == "Logistic Regression":

        classifier = LogisticRegression(
            max_iter=2000,
        )

    elif model_name == "Random Forest":

        classifier = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    elif model_name == "HistGradientBoosting":

        # HistGradientBoosting requires numeric input.
        # The preprocessing pipeline below uses dense
        # one-hot encoded features for categorical variables.
        classifier = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.1,
            l2_regularization=1.0,
            random_state=42,
        )

        # Use dense output for compatibility with boosting.
        for transformer in preprocessor.transformers:
            if transformer[0] == "categorical":
                transformer[1].named_steps[
                    "onehot"
                ].set_params(
                    sparse_output=False
                )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# ============================================================
# 8. RUN ALL EXPERIMENTS
# ============================================================

fold_records = []
summary_records = []

for model_name in model_names:

    for scenario_name, selected_features in scenarios.items():

        print("\n" + "=" * 65)
        print("MODEL:", model_name)
        print("SCENARIO:", scenario_name)
        print("NUMBER OF FEATURES:", len(selected_features))
        print("=" * 65)

        selected_indices = [
            feature_columns.index(col)
            for col in selected_features
        ]

        X_selected = X[:, selected_indices]

        model = build_model(
            model_name,
            selected_features,
        )

        results = cross_validate(
            model,
            X_selected,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            error_score="raise",
        )

        for fold_index in range(5):

            record = {
                "model": model_name,
                "scenario": scenario_name,
                "fold": fold_index + 1,
            }

            for metric in scoring:
                record[metric] = results[
                    f"test_{metric}"
                ][fold_index]

            fold_records.append(record)

            print(
                f"Fold {fold_index + 1}: "
                f"ROC-AUC = "
                f"{results['test_roc_auc'][fold_index]:.4f}"
            )

        summary = {
            "model": model_name,
            "scenario": scenario_name,
            "number_of_features": len(selected_features),
        }

        for metric in scoring:

            values = results[f"test_{metric}"]

            summary[f"{metric}_mean"] = values.mean()

            summary[f"{metric}_std"] = values.std(
                ddof=1
            )

        summary_records.append(summary)

        print(
            f"Mean ROC-AUC: "
            f"{summary['roc_auc_mean']:.4f}"
        )

        print(
            f"Sample SD: "
            f"{summary['roc_auc_std']:.4f}"
        )


# ============================================================
# 9. SAVE RESULTS
# ============================================================

fold_df = pd.DataFrame(fold_records)

summary_df = pd.DataFrame(summary_records)

fold_file = (
    output_dir
    / "a9_robustness_fold_results.csv"
)

summary_file = (
    output_dir
    / "a9_robustness_summary.csv"
)

fold_df.to_csv(
    fold_file,
    index=False,
)

summary_df.to_csv(
    summary_file,
    index=False,
)


# ============================================================
# 10. DISPLAY ROC-AUC COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("FINAL ROC-AUC COMPARISON")
print("=" * 75)

for model_name in model_names:

    print(f"\n{model_name}")

    model_results = summary_df[
        summary_df["model"] == model_name
    ]

    for _, row in model_results.iterrows():

        print(
            f"{row['scenario']}: "
            f"{row['roc_auc_mean']:.4f} "
            f"+/- {row['roc_auc_std']:.4f}"
        )


print("\nFiles saved:")
print(fold_file)
print(summary_file)
