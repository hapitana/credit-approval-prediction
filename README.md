# Credit Approval Prediction — UCI Credit Approval Dataset

## Project Overview

This project explores whether historical credit application outcomes can be predicted from the anonymized attributes in the **UCI Credit Approval dataset**. It is an applied machine-learning project focused on data preparation, exploratory data analysis (EDA), model evaluation, and responsible interpretation.

**Important:** the target represents whether an application was historically approved or rejected. This is **not** a model of loan default, repayment behavior, or credit risk after a loan has been issued.

## Research Questions

- How do the dataset's anonymized attributes relate to historical approval outcomes?
- How well can baseline classification models distinguish approved from rejected applications?
- Which features are most influential to model predictions, and what checks are needed before interpreting them?

## Dataset

- **Source:** UCI Machine Learning Repository — Credit Approval dataset.
- **Observations:** 690 applications.
- **Input features:** 15 anonymized attributes, named `A1`–`A15`.
- **Target:** `A16`, where `+` means approved and `-` means rejected.
- **Target distribution:** 307 approved (44.49%) and 383 rejected (55.51%).

The attributes are anonymized. This README therefore does not assign business meanings such as income, debt, or employment to individual columns.

### Data location

Place the raw dataset at:

```text
data/raw/crx.data
```

The notebook workflow described in this project assumes the data file is available at that location relative to the project root. If your notebook uses a different relative path, update the read path accordingly.

## Repository Structure

```text
01_credit_risk_uci/
├── data/
│   ├── raw/
│   │   └── crx.data
│   └── processed/
├── notebooks/
├── sql/
├── src/
├── outputs/
└── README.md
```

- `data/raw/`: original dataset; keep raw data unchanged.
- `data/processed/`: cleaned or transformed datasets, if saved.
- `notebooks/`: EDA, experiments, and evaluation.
- `sql/`: optional SQL queries for data inspection and aggregation.
- `src/`: reusable preprocessing and modeling code.
- `outputs/`: figures, tables, and other project artifacts.

## Workflow

1. Load the raw data and interpret `?` as a missing value.
2. Check dimensions, target distribution, missingness, and feature types.
3. Explore numeric features and compare their distributions across target classes.
4. Build preprocessing pipelines:
   - Numeric features: `A2`, `A3`, `A8`, `A11`, `A14`, `A15`.
   - Categorical features: the remaining attributes among `A1`–`A15`.
   - Numeric missing values: median imputation.
   - Categorical missing values: most-frequent imputation.
   - Numeric scaling: `StandardScaler`.
   - Categorical encoding: `OneHotEncoder(handle_unknown="ignore")`.
5. Train and evaluate a majority-class baseline, Logistic Regression, and Random Forest.
6. Use stratified five-fold cross-validation for model comparison.
7. Inspect Random Forest permutation importance and investigate whether highly influential features may reflect the historical decision process.

## Model Evaluation

The following results are from **stratified five-fold cross-validation**. Values are mean ± standard deviation across folds.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority-class baseline | 0.5551 ± 0.0035 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| Logistic Regression | 0.8609 ± 0.0365 | 0.8302 ± 0.0524 | 0.8698 ± 0.0272 | 0.8486 ± 0.0340 | 0.9344 ± 0.0196 |
| Random Forest | 0.8739 ± 0.0417 | 0.8574 ± 0.0592 | 0.8665 ± 0.0529 | 0.8601 ± 0.0426 | 0.9365 ± 0.0258 |

Precision, recall, and F1 are reported for the **approved (`+`) class**, consistent with the model evaluation used in this project.

### Interpretation

- Both trained models outperform the majority-class baseline on the reported cross-validation metrics.
- Random Forest has slightly higher mean accuracy, precision, F1, and ROC-AUC than Logistic Regression in these runs.
- Logistic Regression has slightly higher mean recall for the approved class.
- The differences are modest relative to the fold-to-fold variation; these results do not establish that one model is definitively superior.
- The Random Forest achieved a holdout ROC-AUC of **0.9683** on one 80/20 train-test split. This is a single-split result and should not be treated as a guarantee of performance on new data.

## Key Finding: Feature `A9` Needs Investigation

Random Forest permutation importance on the holdout set identified `A9` as the dominant feature:

| Feature | Mean ROC-AUC decrease | Standard deviation |
|---|---:|---:|
| `A9` | 0.274360 | 0.036090 |
| `A14` | 0.006061 | 0.002463 |
| `A15` | 0.005649 | 0.003003 |
| `A10` | 0.004492 | 0.005731 |
| `A5` | 0.003130 | 0.001600 |

The descriptive approval rates by `A9` were:

| `A9` value | Approved | Rejected | Total | Approval rate |
|---|---:|---:|---:|---:|
| `f` | 23 | 306 | 329 | 6.99% |
| `t` | 284 | 77 | 361 | 78.67% |

A chi-square test of independence reported a statistic of **355.2038** with **1 degree of freedom**, **p ≈ 3.12 × 10⁻⁷⁹**, and **Cramér's V = 0.7175**. This indicates a strong association between `A9` and the historical target in this dataset.

These results are **associational, not causal**. Since the feature names are anonymized, the substantive meaning of `A9` cannot be inferred from the column label alone. Its unusually high predictive influence could reflect a legitimate signal, a feature closely related to the historical decision rule, or possible target leakage. Its provenance and availability at the time an application would be scored must be investigated before any practical interpretation.

## Limitations

- **Anonymized features:** the business meaning of `A1`–`A15` is not established in this project.
- **Historical target:** the model predicts historical approval decisions, not repayment or default.
- **Potential leakage or decision-rule replication:** the influence of `A9` requires investigation before drawing operational conclusions.
- **Validation scope:** the reported results use cross-validation and a single holdout split; there is no external, temporal, or prospective validation.
- **Fairness not assessed:** the analysis does not establish that the data or models are fair across demographic or other relevant groups.
- **Limited model set:** only a majority-class baseline, Logistic Regression, and Random Forest have been compared.
- **No deployment assessment:** calibration, decision thresholds, business costs, monitoring, and governance have not been fully evaluated.

## Next Steps

1. Investigate the dataset documentation and the provenance and timing of `A9`.
2. Confirm that every input feature would genuinely be available at the intended prediction time.
3. Evaluate threshold-dependent trade-offs and the costs of false approvals and false rejections.
4. Assess probability calibration and robustness across alternative splits or resampling strategies.
5. Where valid group information is available, assess subgroup performance and fairness carefully.
6. Extend the portfolio toward research on MSME access to finance and FinTech, with a clear distinction between application approval and post-loan outcomes.

## Tools

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib / Seaborn (for exploratory plots, if used)
- Jupyter Notebook

## Responsible Use

This project is an educational portfolio exercise. The findings should not be used to make real credit decisions. Strong predictive performance on this historical dataset does not demonstrate causal validity, fairness, regulatory compliance, or suitability for deployment.

## Reproducibility Note

The metrics above summarize the experiments performed during development. For a reproducible portfolio, keep the preprocessing and model pipelines in code, fix random seeds where appropriate, and record package versions and the exact evaluation procedure. Do not commit private or restricted data; check the dataset's terms and attribution requirements before redistributing it.
