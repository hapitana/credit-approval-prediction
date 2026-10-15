# Credit Approval Prediction
## Python Analysis Scripts: Documentation and Research Workflow


This directory contains 20 Python scripts documenting the analytical workflow of the Credit Approval Prediction project.


The project investigates historical credit approval decisions using the UCI Credit Approval dataset. It combines exploratory data analysis, SQL-based analysis, statistical testing, machine learning, model validation, feature importance analysis, proxy-feature investigation, threshold sensitivity analysis, and exploratory fairness-performance comparisons.


The objective is to understand how applicant characteristics relate to historical approval decisions, evaluate the predictive performance of different models, and examine the potential implications of using particular features in credit decision systems.


**Important:** The dataset contains historical approval decisions, not actual loan repayment or default outcomes. The analysis therefore evaluates the prediction of historical decisions rather than borrowers' ability to repay loans.


---


## 1. Research Objectives


The analytical workflow addresses five main objectives.


### Objective 1: Understand the dataset


- Inspect the dataset structure and feature types.
- Identify missing values and data-quality issues.
- Examine the distribution of the historical approval target.
- Explore relationships between applicant characteristics and approval outcomes.


### Objective 2: Evaluate predictive performance


- Establish baseline classification performance.
- Validate models using cross-validation.
- Compare different classification algorithms.
- Evaluate discrimination and classification metrics.


### Objective 3: Investigate feature associations


- Examine the relationship between feature A9 and historical approval decisions.
- Apply statistical association testing.
- Quantify the strength of the observed association.
- Investigate whether other features can predict A9.


### Objective 4: Examine model sensitivity


- Compare models trained with and without A9.
- Investigate the effect of removing selected proxy candidates.
- Evaluate changes in predictive performance.
- Explore the relationship between classification thresholds and group-level metrics.


### Objective 5: Explore fairness-performance trade-offs


- Compare predictive performance with differences in historical approval predictions across A9 groups.
- Examine approval-rate, true-positive-rate, and false-positive-rate gaps.
- Identify exploratory Pareto-efficient configurations.
- Document limitations and questions requiring further research.


---


## 2. Dataset and Target Definition


The project uses the UCI Credit Approval dataset.


Official dataset page:


https://archive.ics.uci.edu/dataset/27/credit+approval


The dataset contains 690 observations and 15 input features, represented as A1 through A15, along with a binary target A16.


The original data file is expected at:


```text
data/raw/crx.data
```


### Target interpretation


The original target uses two values:


| Target | Interpretation |
|---|---|
| `+` | Historical application approval |
| `-` | Historical application rejection |


The target represents the decisions recorded in the dataset. It must not be interpreted as a measure of actual repayment performance, creditworthiness, or default risk.


### Feature interpretation


The input variables are anonymized. In particular, this project does not assume that A9 has a known real-world meaning.


The observed association between A9 and historical approval decisions is investigated empirically. Without reliable documentation identifying the underlying meaning of A9, its relationship to protected characteristics, or the decision process that generated the data, the results cannot establish discrimination or unlawful decision-making.


Similarly, a feature that predicts A9 is treated as a potential proxy candidate for further investigation, not as proof that the feature is an unfair proxy.


---


## 3. Software and Libraries


The scripts use Python and common data-analysis and machine-learning libraries.


| Tool or library | Purpose |
|---|---|
| Python | Main programming language |
| Polars | Data loading, transformation, and tabular analysis |
| pandas | Tabular data
