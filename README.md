# Credit Approval Prediction


**An end-to-end data analysis and machine learning project exploring
historical credit approval decisions using Python.**


## Project Overview


Financial institutions need consistent processes for evaluating credit
applications. Historical application data can help analysts understand
patterns associated with previous approval decisions and evaluate
classification models.


This project explores the UCI Credit Approval dataset to investigate
application characteristics, compare machine learning models, and
identify features strongly associated with historical approval outcomes.


The project demonstrates a structured analytical workflow, from
exploratory data analysis and data quality checks to model evaluation
and statistical hypothesis testing.


> **Important:** This project predicts historical credit approval
> decisions. It does not directly predict loan default or repayment
> behavior and is not intended for direct use in real lending decisions.


## Business Objectives


The project addresses five key questions:


1. What patterns can be identified in historical credit application data?
2. Which features are associated with approval and rejection decisions?
3. How effectively can machine learning models classify applications?
4. Which features contribute most to predictive performance?
5. What unusual values or strong feature relationships require further
   investigation?


## Dataset


**Source:** UCI Machine Learning Repository — Credit Approval Dataset


Dataset: https://archive.ics.uci.edu/dataset/27/credit+approval


| Property | Description |
|---|---|
| Observations | 690 applications |
| Input features | 15 anonymized features, A1–A15 |
| Target variable | A16 |
| Target classes | `+` (approved), `-` (rejected) |
| Data types | Numeric and categorical |


### Target Distribution


| Historical decision | Applications | Percentage |
|---|---:|---:|
| Approved (`+`) | 307 | 44.5% |
| Rejected (`-`) | 383 | 55.5% |
| **Total** | **690** | **100%** |


The input features are anonymized. Their real-world meanings should not
be inferred from the feature labels alone.


## Tools and Technologies


- **Python** — Data analysis and model development
- **pandas and NumPy** — Data manipulation
- **Matplotlib and Seaborn** — Data visualization
- **scikit-learn** — Classification models and evaluation
- **SciPy** — Statistical hypothesis testing
- **SQL** — Data exploration and querying
- **Git and GitHub** — Version control and project documentation


## Analytical Workflow


The project is organized into 12 Python scripts.


| Stage | Script | Purpose |
|---|---|---|
| 1 | `01_first_exploration.py` | Inspect dataset structure and data quality |
| 2 | `02_sql_exploration.py` | Explore data using SQL |
| 3 | `03_visual_exploration.py` | Visualize feature distributions and patterns |
| 4 | `04_feature_analysis.py` | Analyze features and target relationships |
| 5 | `05_outlier_analysis.py` | Investigate potential outliers |
| 6 | `06_inspect_extreme_values.py` | Examine extreme values, especially A15 |
| 7 | `07_baseline_model.py` | Establish a Logistic Regression baseline |
| 8 | `08_model_validation.py` | Evaluate model performance |
| 9 | `09_compare_models.py` | Compare Logistic Regression and Random Forest |
| 10 | `10_feature_importance.py` | Evaluate permutation feature importance |
| 11 | `11_analyze_A9.py` | Investigate feature A9 |
| 12 | `12_test_A9_association.py` | Statistically test the association between A9 and approval |


Detailed methodology and findings are documented in
[`notebooks/README.md`](notebooks/README.md).


## Machine Learning Models


Two classification algorithms were evaluated:


- **Logistic Regression** — Baseline classification model
- **Random Forest** — Nonlinear, tree-based classification model


The models were evaluated using five-fold cross-validation.


### Cross-Validation Results


| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.8609 ± 0.0365 | 0.8739 ± 0.0417 |
| Precision | 0.8302 ± 0.0524 | 0.8574 ± 0.0592 |
| Recall | 0.8698 ± 0.0272 | 0.8665 ± 0.0529 |
| F1-score | 0.8486 ± 0.0340 | 0.8601 ± 0.0426 |
| ROC-AUC | 0.9344 ± 0.0196 | 0.9365 ± 0.0258 |


Random Forest achieved somewhat higher mean accuracy, precision,
F1-score, and ROC-AUC in these experiments. However, the difference
in mean ROC-AUC was small.


Further validation is needed before concluding that either model is
more suitable for practical deployment.


*Note: Precision and recall depend on which target class is defined as
positive. Consult the modeling scripts for the exact class convention.*


## Key Findings


### 1. Strong Predictive Performance


Both models achieved strong cross-validation results on this dataset.
These results demonstrate that the historical decisions contain
patterns that the evaluated models can learn.


They do not guarantee similar performance on future applications.


### 2. Extreme Values in A15


Feature A15 contains extreme values and substantially different
distributions across the approval classes.


The maximum observed value among approved applications was 100,000,
compared with 5,552 among rejected applications.


These observations warrant investigation. They should not automatically
be removed because the available data do not establish whether they
are errors or valid extreme values.


### 3. Feature A9 Has a Strong Association with Approval


Feature A9 showed a strong association with the historical target.


| A9 category | Approved | Rejected | Approval rate |
|---|---:|---:|---:|
| `f` | 23 | 306 | 6.99% |
| `t` | 284 | 77 | 78.67% |


The chi-square test produced a statistic of 355.2038, with 1 degree
of freedom and a p-value of approximately 3.12 × 10⁻⁷⁹.
Cramér's V was 0.7175.


These results indicate a strong statistical association, not causation.
The meaning and availability of A9 should be investigated before
drawing business conclusions or considering its use in a decision system.


### 4. Feature Importance Requires Careful Interpretation


Permutation importance identified A9 as the most influential feature
under the evaluation procedure used.


This finding warrants additional investigation into feature meaning,
potential information leakage, and whether the feature would be
available at the appropriate point in an actual application process.


## Limitations and Responsible Use


- **Anonymized features:** The real-world meanings of A1–A15 are not
  established by the feature labels.
- **Small dataset:** The dataset contains only 690 observations.
- **Historical target:** Approval decisions are not equivalent to
  borrower repayment or default outcomes.
- **Potential leakage:** Strong feature-target relationships require
  investigation of feature availability and the evaluation pipeline.
- **Generalization:** Cross-validation and holdout metrics do not
  guarantee performance on future data.
- **Fairness and compliance:** The analysis does not establish that
  the features or models are fair, legally compliant, or appropriate
  for lending decisions.


This project is intended for educational and portfolio demonstration
purposes. It should not be used directly to approve or reject real
credit applications.


## Repository Structure


```text
credit-approval-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── README.md
│   ├── 01_first_exploration.py
│   ├── 02_sql_exploration.py
│   ├── 03_visual_exploration.py
│   ├── 04_feature_analysis.py
│   ├── 05_outlier_analysis.py
│   ├── 06_inspect_extreme_values.py
│   ├── 07_baseline_model.py
│   ├── 08_model_validation.py
│   ├── 09_compare_models.py
│   ├── 10_feature_importance.py
│   ├── 11_analyze_A9.py
│   └── 12_test_A9_association.py
├── outputs/
├── src/
├── sql/
└── README.md
