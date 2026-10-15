from pathlib import Path

import polars as pl

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


# 1. Define paths
project_dir = Path(__file__).resolve().parent.parent

file_path = (
    project_dir
    / "data"
    / "raw"
    / "crx.data"
)

output_dir = project_dir / "reports" / "a9_sensitivity"
output_dir.mkdir(parents=True, exist_ok=True)


# 2. Load data
columns = [f"A{i}" for i in range(1, 17)]

df = pl.read_csv(
    file_path,
    has_header=False,
    new_columns=columns,
    null_values="?",
)

df = df.filter(
    pl.col("A16").is_in(["+", "-"])
)

print("Dataset shape:", df.shape)


# 3. Prepare target
y = (
    df.select(
        (pl.col("A16") == "+").cast(pl.Int64)
    )
    .to_series()
    .to_numpy()
)


# 4. Define features
numeric_features = [
    "A2",
    "A3",
    "A8",
    "A11",
    "A14",
    "A15",
]

all_features = columns[:15]

experiments = {
    "All features": all_features,
    "Without A9": [
        col for col in all_features
        if col != "A9"
    ],
    "A9 only": ["A9"],
}


# 5. Convert numerical columns
df = df.with_columns([
    pl.col(col).cast(pl.Float64, strict=False)
    for col in numeric_features
])


# 6. Split once, using the same rows for all experiments
X_df = df.select(all_features)
X = X_df.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# 7. Run experiments
results = []

for experiment_name, selected_features in experiments.items():

    # Select the features for this experiment
    X_train_selected = X_train[:, [
        all_features.index(col)
        for col in selected_features
    ]]

    X_test_selected = X_test[:, [
        all_features.index(col)
        for col in selected_features
    ]]

    numeric_selected = [
        col for col in selected_features
        if col in numeric_features
    ]

    categorical_selected = [
        col for col in selected_features
        if col not in numeric_features
    ]

    # IMPORTANT:
    # Indices must refer to the columns in the selected input,
    # not the original 15-column dataset.
    numeric_indices = [
        selected_features.index(col)
        for col in numeric_selected
    ]

    categorical_indices = [
        selected_features.index(col)
        for col in categorical_selected
    ]

    transformers = []

    if numeric_indices:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_indices)
        )

    if categorical_indices:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_indices)
        )

    preprocessor = ColumnTransformer(transformers)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train_selected, y_train)

    predictions = model.predict(X_test_selected)
    probabilities = model.predict_proba(X_test_selected)[:, 1]

    roc_auc = roc_auc_score(y_test, probabilities)
    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "experiment": experiment_name,
        "number_of_features": len(selected_features),
        "roc_auc": roc_auc,
        "accuracy": accuracy,
    })

    print(f"\n{experiment_name}")
    print("Features:", len(selected_features))
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


# 8. Save results
results_df = pl.DataFrame(results)

output_path = output_dir / "a9_sensitivity_results.csv"
results_df.write_csv(output_path)

print("\nComparison of experiments:")
print(results_df)

print("\nResults saved to:")
print(output_path)