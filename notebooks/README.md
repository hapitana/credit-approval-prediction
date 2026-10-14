# Credit Approval Prediction — Exploratory Data Analysis & Modeling


## 1. Overview


This folder contains 12 Python scripts documenting the analytical workflow
for the Credit Approval Prediction project.


The project explores historical credit application data to understand
which application characteristics are associated with approval decisions,
evaluate predictive model performance, and investigate the importance
of individual features.


The workflow covers:


- Initial data exploration
- SQL-based data analysis
- Exploratory data visualization
- Feature analysis
- Outlier and extreme-value investigation
- Baseline model development
- Model validation and comparison
- Feature importance analysis
- Statistical analysis of a highly influential feature


The goal is to demonstrate a structured, reproducible approach to
data analysis and machine learning for a credit approval classification
problem.


---


## 2. Business Problem


Financial institutions need consistent and efficient processes for
evaluating credit applications.


Historical application data can help analysts identify patterns
associated with previous approval decisions and develop models that
estimate whether an application resembles those previously approved
or rejected.


This project addresses the following questions:


1. What patterns and distributions can be observed in the application data?
2. Which features are associated with historical approval decisions?
3. How well can machine learning models classify applications as approved
   or rejected?
4. Which features contribute most to predictive performance?
5. Are there extreme values or unusually strong feature relationships
   that require further investigation?


### Business objective


Develop and evaluate classification models that predict historical
credit approval decisions and identify patterns that may support
further investigation of the application review process.


**Important distinction:** This project predicts historical approval
decisions. It does not directly predict whether a borrower will default
on a loan or repay a debt.


The results should therefore not be interpreted as a complete credit
risk assessment or as a ready-to-deploy automated lending system.


---


## 3. Dataset


### Source


The project uses the Credit Approval dataset from the UCI Machine
Learning Repository.


The dataset is available from:


- UCI Machine Learning Repository:
  https://archive.ics.uci.edu/dataset/27/credit+approval


### Dataset characteristics


| Property | Description |
|---|---|
| Dataset | UCI Credit Approval |
| Number of observations | 690 |
| Number of columns | 16 |
| Input features | 15 columns, A1–A15 |
| Target variable | A16 |
| Target classes | `+` and `-` |
| Data types | Numeric and categorical |
| Missing values | Represented by `?` in the raw dataset |


The features are anonymized and identified by the labels A1 through A15.
Their real-world meanings are not provided in the dataset columns used
in this analysis.


Consequently, the analysis refers to features by their original labels
rather than assigning them unsupported business interpretations.


### Target variable


The target column, A16, represents the historical application decision:


- `+`: Approved
- `-`: Rejected


The dataset contains:


| Historical decision | Number of applications |
|---|---:|
| Approved (`+`) | 307 |
| Rejected (`-`) | 383 |
| Total | 690 |


The target distribution is moderately imbalanced, with more rejected
applications than approved applications.


This imbalance should be considered when interpreting model performance,
particularly precision, recall, and the consequences of classification
errors.


---


## 4. Analytical Workflow


The project is organized into 12 Python scripts. Each script focuses
on a specific stage of the analysis.


| Script | Purpose |
|---|---|
| `01_first_exploration.py` | Inspect the dataset structure, dimensions, data types, target distribution, and initial data quality. |
| `02_sql_exploration.py` | Explore the dataset using SQL-style queries and summarize relevant patterns. |
| `03_visual_exploration.py` | Create visualizations to examine distributions and relationships in the data. |
| `04_feature_analysis.py` | Investigate individual features and their relationships with the target variable. |
| `05_outlier_analysis.py` | Examine potential outliers and unusual observations. |
| `06_inspect_extreme_values.py` | Investigate extreme values, particularly in feature A15, and compare their distributions across target classes. |
| `07_baseline_model.py` | Establish a baseline classification model using Logistic Regression. |
| `08_model_validation.py` | Evaluate model performance using validation procedures and classification metrics. |
| `09_compare_models.py` | Compare Logistic Regression and Random Forest performance. |
| `10_feature_importance.py` | Investigate feature contributions using permutation importance. |
| `11_analyze_A9.py` | Examine the relationship between feature A9 and historical approval decisions. |
| `12_test_A9_association.py` | Use statistical testing to assess the association between A9 and the target variable. |


Together, these scripts document the progression from data understanding
to predictive modeling and statistical investigation.


---


## 5. Analytical Methods


### 5.1 Data Quality and Exploration


The initial analysis examines:


- Dataset dimensions and column types
- Missing or unknown values
- Target class distribution
- Numeric feature distributions
- Categorical feature frequencies
- Potential anomalies and extreme observations


Missing or unknown values require careful treatment before modeling.
Any preprocessing decisions should be fitted using training data
to avoid information leakage.


### 5.2 Exploratory Data Analysis


Exploratory analysis is used to understand the structure of the data
before drawing conclusions from predictive models.


The analysis examines feature distributions, differences between
approved and rejected applications, and potentially unusual patterns.


Because the feature names are anonymized, observed relationships
are interpreted statistically rather than assigned unsupported
business meanings.


### 5.3 Outlier and Extreme-Value Analysis


Potential outliers are investigated to determine whether unusually
large values could influence descriptive statistics or predictive
performance.


Feature A15 receives particular attention because its distribution
contains substantial extreme values.


Extreme observations are not automatically treated as errors.
Without additional documentation, it is not possible to determine
whether a large value is invalid, unusual but legitimate, or meaningful
for the classification task.


Therefore, removing or transforming these observations should be
treated as a modeling decision requiring further validation.


### 5.4 Predictive Modeling


Two classification approaches are evaluated:


**Logistic Regression**


Used as a baseline model because it is a widely used classification
method and provides a useful benchmark for comparison.


**Random Forest**


Used as a nonlinear, tree-based alternative that can capture more
complex relationships between features.


The models are compared using classification metrics and ROC-AUC.


### 5.5 Model Evaluation


The evaluation uses five-fold cross-validation to estimate how
consistently the models perform across different data partitions.


The metrics include:


- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted approvals that are actually
  approved in the historical labels.
- **Recall:** Proportion of historically approved applications correctly
  identified by the model.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ability to distinguish between the two target classes
  across different classification thresholds.


Metric interpretation depends on which class is treated as the positive
class. In this project, the positive class should be explicitly checked
in the modeling scripts before interpreting precision and recall.


Cross-validation results describe performance on this dataset and do
not guarantee similar performance on future applications.


### 5.6 Feature Importance


Permutation importance is used to investigate how much model performance
changes when the values of an individual feature are shuffled.


A larger decrease in ROC-AUC indicates that the fitted model relies
more heavily on that feature for its predictions under the evaluation
procedure used.


Permutation importance indicates predictive usefulness, not causation.
It can also be affected by correlated features, dataset composition,
and the evaluation design.


### 5.7 Statistical Association


Feature A9 is examined in more detail because it shows a particularly
strong relationship with the target variable.


The analysis includes:


- A cross-tabulation of A9 against A16
- Approval rates for each observed A9 category
- A chi-square test of independence
- Cramér's V as a measure of association strength


The chi-square test evaluates whether the observed categorical
variables are statistically associated.


Cramér's V summarizes the strength of that association on a scale
from 0 to 1, where larger values indicate a stronger relationship.


Neither statistical significance nor a strong association establishes
a causal relationship.


---


## 6. Key Findings


### 6.1 Baseline Model Performance


The Logistic Regression baseline achieved the following five-fold
cross-validation results:


| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 0.8609 | 0.0365 |
| Precision | 0.8302 | 0.0524 |
| Recall | 0.8698 | 0.0272 |
| F1-score | 0.8486 | 0.0340 |
| ROC-AUC | 0.9344 | 0.0196 |


These results indicate that Logistic Regression provides a useful
baseline on this dataset.


The results should be interpreted in the context of the validation
design, preprocessing pipeline, and definition of the positive class.


### 6.2 Random Forest Performance


The Random Forest model, configured with 300 estimators and
`min_samples_leaf=2`, achieved the following five-fold cross-validation
results:


| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 0.8739 | 0.0417 |
| Precision | 0.8574 | 0.0592 |
| Recall | 0.8665 | 0.0529 |
| F1-score | 0.8601 | 0.0426 |
| ROC-AUC | 0.9365 | 0.0258 |


Random Forest performed somewhat better than Logistic Regression
on accuracy, precision, F1-score, and mean ROC-AUC in these results.


However, the difference in mean ROC-AUC is small. Further validation
would be necessary before concluding that Random Forest is meaningfully
better or more suitable for a real-world application.


### 6.3 Feature A15 Contains Extreme Values


Feature A15 has a strongly skewed distribution, with some observations
far larger than the median.


Among approved applications, the median A15 value is 221 and the
maximum is 100,000.


Among rejected applications, the median is 1 and the maximum is 5,552.


These differences indicate that A15 has very different distributions
across the two historical decision classes.


However, because the feature is anonymized, its real-world meaning
cannot be established from the feature label alone.


The extreme values should be investigated carefully rather than
automatically removed.


### 6.4 Feature A9 Has a Strong Association with Historical Decisions


The observed distribution of A9 is:


| A9 category | Approved (`+`) | Rejected (`-`) | Approval rate |
|---|---:|---:|---:|
| `f` | 23 | 306 | 6.99% |
| `t` | 284 | 77 | 78.67% |


The chi-square test produced:


| Statistic | Result |
|---|---:|
| Chi-square | 355.2038 |
| Degrees of freedom | 1 |
| p-value | Approximately 3.12 × 10⁻⁷⁹ |
| Cramér's V | 0.7175 |


The observed data show a strong statistical association between A9
and historical approval decisions.


This finding also aligns with the feature-importance analysis, in
which A9 was the most influential feature under the permutation
importance procedure used.


Nevertheless, this association does not prove that A9 causes an
approval or rejection decision.


The meaning of A9 is not established by the anonymized dataset, so
its business interpretation and appropriateness for decision-making
cannot be determined from this analysis alone.


### 6.5 Holdout Evaluation


The Random Forest model achieved a holdout ROC-AUC of 0.9683 in the
reported evaluation.


This result suggests strong discrimination on the evaluated holdout
set. However, a single holdout score is not sufficient to establish
generalization to future applications.


The result should be interpreted alongside cross-validation results
and a review of the full preprocessing and evaluation procedure.


---


## 7. Limitations and Considerations


### 7.1 Anonymized Features


The feature columns A1–A15 are anonymized. Their underlying business
meanings are not available in the analysis.


This limits the ability to explain findings in terms of specific
financial characteristics or to assess whether individual features
are appropriate for credit decisions.


### 7.2 Limited Dataset Size


The dataset contains 690 observations.


Although it is suitable for demonstrating exploratory analysis and
classification techniques, it is relatively small for developing
a production-grade credit decision system.


Larger and more representative datasets would be needed for stronger
evidence of generalization.


### 7.3 Historical Decisions Are Not Loan Repayment Outcomes


The target variable records historical approval or rejection decisions.
It does not indicate whether approved applicants later defaulted,
repaid their debts, or experienced financial hardship.


Consequently, this project evaluates the prediction of historical
decisions, not the prediction of borrower default or credit losses.


Historical decisions may also reflect the policies, practices, and
biases in place when the data were collected.


### 7.4 Potential Data Leakage and Feature Availability


The strong association between A9 and the target warrants further
investigation.


Before any real-world use, it would be necessary to establish what
A9 represents, when it becomes available, and whether it could
directly or indirectly encode information related to the historical
decision.


Preprocessing, feature selection, and hyperparameter tuning must also
be performed without using information from validation or test data.


The high holdout ROC-AUC should not be treated as evidence that leakage
has been ruled out.


### 7.5 Extreme Values


Some feature A15 observations are unusually large.


The analysis does not establish whether these observations are data
errors or valid values.


Further investigation and sensitivity analysis would be needed to
compare model performance with appropriate alternative treatments
of extreme observations.


### 7.6 Model Evaluation and Generalization


Cross-validation and holdout evaluation provide useful estimates
of performance, but neither guarantees performance on future data.


Additional evaluation should consider alternative splits, calibration,
threshold selection, subgroup performance, and the operational
consequences of false approvals and false rejections.


Any reported metrics also depend on the exact preprocessing,
positive-class definition, model configuration, and evaluation code.


### 7.7 Fairness, Privacy, and Responsible Use


A strong predictive relationship does not establish that a feature
is fair, legally permissible, or appropriate for lending decisions.


Before real-world deployment, a credit decision system would require
a review of applicable regulations, data privacy, explainability,
fairness across relevant groups, and appropriate human oversight.


This project is an educational analytical exercise and should not
be used directly to approve or reject real credit applications.


---


## 8. How to Run the Scripts


The scripts are numbered to reflect the analytical workflow.


Run them in sequence, starting with:


`01_first_exploration.py`


and continuing through:


`12_test_A9_association.py`


Before running the scripts, make sure that:


1. Python and the required libraries are installed.
2. The dataset is downloaded from the UCI Machine Learning Repository.
3. The dataset file is placed in the location expected by the scripts.
4. Any file paths in the scripts match your local project structure.
5. The scripts are executed in an environment where their dependencies
   are available.


Depending on the scripts, required libraries may include:


- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- SciPy
- SQLite or a compatible SQL engine


Check the imports in each script to confirm the actual dependencies.


For reproducibility, consider documenting the Python version,
library versions, and any random seeds used in the experiments.


---


## 9. Project Structure


The scripts in this folder are part of the larger project:


```text
credit-approval-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
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
│
├── outputs/
├── src/
└── sql/