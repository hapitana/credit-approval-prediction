from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


# 1. Define project paths
project_root = Path(__file__).resolve().parent.parent

file_path = (
    project_root
    / "data"
    / "raw"
    / "crx.data"
)

output_dir = (
    project_root
    / "reports"
    / "model_validation"
)

output_dir.mkdir(parents=True, exist_ok=True)


# 2. Load the dataset
columns = [f"A{i}" for i in range(1, 17)]

df = pl.read_csv(
    file_path,
    has_header=False,
    new_columns=columns,
    null_values="?",
)


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
    col for col in columns[:15]
    if col not in numeric_features
]


# 4. Convert numerical columns to numeric values
df = df.with_columns([
    pl.col(col).cast(pl.Float64, strict=False)
    for col in numeric_features
])


# 5. Keep valid target labels
df = df.filter(
    pl.col("A16").is_in(["+", "-"])
)


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

print("Dataset shape:", X.shape)
print("Approved applications:", int(y.sum()))
print("Rejected applications:", int(len(y) - y.sum()))


# 7. Configure cross-validation
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


# 8. Define feature scenarios
scenarios = {
    "All features": feature_columns,

    "Without A9": [
        col for col in feature_columns
        if col != "A9"
    ],

    "A9 only": ["A9"],
}


# 9. Evaluate each scenario
results_list = []

for scenario_name, selected_features in scenarios.items():

    print(f"\n{'=' * 50}")
    print(f"SCENARIO: {scenario_name}")
    print(f"Features: {selected_features}")
    print(f"Number of features: {len(selected_features)}")
    print("=" * 50)

    # Identify selected numeric and categorical features
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

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                [
                    selected_features.index(col)
                    for col in selected_numeric
                ],
            )
        )

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

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                [
                    selected_features.index(col)
                    for col in selected_categorical
                ],
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    logistic_model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=2000),
        ),
    ])

    # Select the requested columns
    selected_indices = [
        feature_columns.index(col)
        for col in selected_features
    ]

    X_selected = X[:, selected_indices]

    # Cross-validation
    cv_results = cross_validate(
        logistic_model,
        X_selected,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        error_score="raise",
    )

    row = {
        "scenario": scenario_name,
        "number_of_features": len(selected_features),
    }

    for metric in scoring:
        scores = cv_results[f"test_{metric}"]

        row[f"{metric}_mean"] = scores.mean()
        row[f"{metric}_std"] = scores.std()

        print(
            f"{metric}: "
            f"{scores.mean():.4f} "
            f"+/- {scores.std():.4f}"
        )

    results_list.append(row)


# 10. Evaluate the majority-class baseline
print(f"\n{'=' * 50}")
print("SCENARIO: Majority-class baseline")
print("=" * 50)

dummy_model = DummyClassifier(
    strategy="most_frequent"
)

dummy_results = cross_validate(
    dummy_model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
    error_score="raise",
)

baseline_row = {
    "scenario": "Majority-class baseline",
    "number_of_features": 0,
}

for metric in scoring:
    scores = dummy_results[f"test_{metric}"]

    baseline_row[f"{metric}_mean"] = scores.mean()
    baseline_row[f"{metric}_std"] = scores.std()

    print(
        f"{metric}: "
        f"{scores.mean():.4f} "
        f"+/- {scores.std():.4f}"
    )

results_list.append(baseline_row)


# 11. Save results to CSV
results_df = pd.DataFrame(results_list)

output_file = (
    output_dir
    / "cross_validation_feature_comparison.csv"
)

results_df.to_csv(
    output_file,
    index=False,
)

print("\nResults saved to:")
print(output_file)


# 12. Display a concise comparison
print("\nFINAL ROC-AUC COMPARISON")

for _, row in results_df.iterrows():
    print(
        f"{row['scenario']}: "
        f"{row['roc_auc_mean']:.4f} "
        f"+/- {row['roc_auc_std']:.4f}"
    )
