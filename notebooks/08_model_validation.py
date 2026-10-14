import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the data
columns = [f"A{i}" for i in range(1, 17)]

df = pd.read_csv(
    "../data/raw/crx.data",
    header=None,
    names=columns,
    na_values="?",
)

numeric_features = ["A2", "A3", "A8", "A11", "A14", "A15"]
categorical_features = [
    col for col in columns[:15]
    if col not in numeric_features
]

for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["A16"].isin(["+", "-"])].copy()

X = df[columns[:15]]
y = (df["A16"] == "+").astype(int)

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# Logistic regression
logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000)),
])

# Baseline: always predict the majority class in the training data
dummy_model = DummyClassifier(strategy="most_frequent")

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

for name, model in [
    ("Majority baseline", dummy_model),
    ("Logistic regression", logistic_model),
]:
    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    print(f"\n=== {name} ===")

    for metric in scoring:
        scores = results[f"test_{metric}"]
        print(
            f"{metric}: {scores.mean():.4f} "
            f"+/- {scores.std():.4f}"
        )