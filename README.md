# Credit Approval Prediction


### Machine Learning, Feature Ablation, Proxy Analysis, and Performance–Disparity Trade-offs


An end-to-end data analysis and machine learning project investigating historical credit approval decisions using Python, statistical testing, cross-validation, feature ablation, and threshold sensitivity analysis.


The project explores how predictive performance changes when selected features are removed and examines how those changes relate to differences in model predictions across groups defined by an anonymized feature.


> **Important:** This project predicts historical credit approval decisions. It does not directly predict loan default, repayment behavior, or true creditworthiness. The results are exploratory and are not intended for direct use in real lending decisions.


---


## 1. Project Overview


Credit decision systems can learn patterns from historical application data. However, reproducing historical decisions accurately does not necessarily establish that a model is appropriate, reliable, or fair.


This project investigates the relationship between predictive performance, feature selection, and group-level differences in predictions.


The analysis begins with exploratory data analysis and baseline classification models. It then examines feature A9, evaluates how well other features predict A9, tests alternative feature sets, and compares predictive performance with several group-disparity metrics.


The objective is to demonstrate a reproducible analytical workflow while distinguishing statistical findings from causal or fairness claims that the available data cannot establish.


### Research Questions


1. How accurately can machine learning models reproduce historical credit approval decisions?
2. How strongly is feature A9 associated with the historical approval outcome?
3. How much information about A9 can be predicted from the remaining features?
4. How does removing A9 affect predictive performance and group-level prediction differences?
5. What happens when A9 and selected proxy-feature candidates are removed together?
6. How sensitive are performance and disparity metrics to the classification threshold?
7. What trade-offs emerge when predictive performance and approval-rate gaps are evaluated together?


---


## 2. Dataset


**Source:** UCI Machine Learning Repository — Credit Approval Dataset


Dataset documentation:
https://archive.ics.uci.edu/dataset/27/credit+approval


| Property | Description |
|---|---|
| Observations | 690 applications |
| Input features | 15 anonymized features, A1–A15 |
| Target variable | A16 |
| Approved class | `+` |
| Rejected class | `-` |
| Data types | Numeric and categorical |
| Missing values | Represented using `?` in the original dataset |


### Target Distribution


| Historical decision | Applications | Percentage |
|---|---:|---:|
| Approved (`+`) | 307 | 44.5% |
| Rejected (`-`) | 383 | 55.5% |
| **Total** | **690** | **100%** |


The features are anonymized. Their real-world meanings must not be inferred from the feature labels alone.


In particular, the meaning of A9 is unknown in this analysis. The project therefore treats A9 as an anonymized categorical feature rather than assigning it an assumed demographic, financial, or personal meaning.


### Important Target Limitation


The target records historical approval decisions, not actual repayment or default outcomes.


Consequently, model performance measures how well the models reproduce historical decisions. High predictive performance does not establish that the models accurately assess a borrower's underlying ability to repay a loan.


---


## 3. Tools and Technologies


- **Python** — Data analysis and machine learning
- **Polars** — Data loading and tabular data operations
- **pandas and NumPy** — Data manipulation and model integration
- **scikit-learn** — Classification, cross-validation, and permutation importance
- **SciPy** — Statistical hypothesis testing
- **Matplotlib** — Data visualization
- **SQL** — Exploratory data analysis
- **Git and GitHub** — Version control and documentation


---


## 4. Analytical Workflow


The project contains 20 numbered Python scripts covering exploratory analysis, modeling, statistical testing, feature ablation, and performance–disparity evaluation.


| No. | Script | Purpose |
|---:|---|---|
| 01 | `01_first_exploration.py` | Inspect dataset structure and data quality |
| 02 | `02_sql_exploration.py` | Explore data using SQL |
| 03 | `03_visual_exploration.py` | Visualize feature distributions and patterns |
| 04 | `04_feature_analysis.py` | Investigate features and their relationships with the target |
| 05 | `05_outlier_analysis.py` | Identify potential outliers |
| 06 | `06_inspect_extreme_values.py` | Examine extreme observations |
| 07 | `07_baseline_model.py` | Establish a Logistic Regression baseline |
| 08 | `08_model_validation.py` | Evaluate model performance |
| 09 | `09_compare_models.py` | Compare classification algorithms |
| 10 | `10_feature_importance.py` | Evaluate permutation feature importance |
| 11 | `11_analyze_A9.py` | Examine A9 and historical approval patterns |
| 12 | `12_test_A9_association.py` | Statistically test the association between A9 and A16 |
| 13 | `13_A9_sensitivity_analysis.py` | Investigate sensitivity to A9 |
| 14 | `14_a9_decision_impact_audit.py` | Compare group-level prediction and decision metrics |
| 15 | `15_a9_proxy_analysis.py` | Test whether other features predict A9 |
| 16 | `16_a9_proxy_feature_analysis.py` | Analyze proxy-feature information and feature ablation |
| 17 | `17_a9_proxy_ablation_validation.py` | Validate alternative feature sets using cross-validation |
| 18 | `18_A9_threshold_sensivity_analysis.py` | Examine threshold sensitivity across feature scenarios |
| 19 | `19_a9_fairness_performance_tradeoff.py` | Compare predictive performance and group disparities |
| 20 | `20_a9_final_results_dashboard.py` | Generate the final Pareto analysis, comparison tables, and plots |


**Filename note:** Script 18 retains the existing filename spelling `sensivity`.


The scripts document the progression from basic data exploration to more advanced model evaluation and feature-sensitivity experiments.


---


## 5. Modeling Strategy


The project uses classification models to predict the historical approval target.


### Algorithms Evaluated


- Logistic Regression
- Random Forest
- HistGradientBoosting


Five-fold cross-validation is used in the cross-model robustness experiment.


The analysis also uses out-of-fold predictions to evaluate selected feature scenarios and examine group-level prediction differences.


### Evaluation Metrics


| Metric | Purpose |
|---|---|
| ROC-AUC | Measures the model's ability to rank positive and negative historical outcomes |
| Accuracy | Measures the proportion of correctly classified observations |
| Precision | Measures the proportion of predicted positive outcomes that are positive |
| Recall / TPR | Measures the proportion of actual positive labels correctly identified |
| F1-score | Combines precision and recall |
| Balanced accuracy | Averages recall across the two classes |
| Approval-rate gap | Measures the absolute difference in predicted approval rates between A9 groups |
| TPR gap | Measures the absolute difference in true-positive rates between A9 groups |
| FPR gap | Measures the absolute difference in false-positive rates between A9 groups |


The positive class and threshold must be considered when interpreting classification metrics.


The group-disparity metrics compare model predictions against historical labels. They are not direct measures of fairness against an independently established ground truth.


---


## 6. Cross-Model Robustness


The following results summarize the five-fold cross-validation experiment comparing different models and feature scenarios.


### ROC-AUC Results


| Model | Feature scenario | Mean ROC-AUC | Standard deviation |
|---|---|---:|---:|
| Logistic Regression | All features | 0.9336 | 0.0229 |
| Logistic Regression | Without A9 | 0.8568 | 0.0250 |
| Logistic Regression | A9 only | 0.8620 | 0.0386 |
| Random Forest | All features | 0.9344 | 0.0287 |
| Random Forest | Without A9 | 0.8596 | 0.0445 |
| Random Forest | A9 only | 0.8620 | 0.0386 |
| HistGradientBoosting | All features | 0.9293 | 0.0285 |
| HistGradientBoosting | Without A9 | 0.8615 | 0.0417 |
| HistGradientBoosting | A9 only | 0.8620 | 0.0386 |


### Interpretation


All three models achieved relatively high ROC-AUC values when all features were included.


Removing A9 reduced mean ROC-AUC by approximately 0.068–0.077 across the three models.


The A9-only scenario also achieved substantial predictive performance against the historical target.


These results indicate that A9 contains strong predictive information about historical approval decisions.


However, the small differences between the all-feature models do not establish that one algorithm is statistically superior to another. Additional validation would be necessary to support such a conclusion.


---


## 7. Statistical Association Between A9 and Historical Approval


A9 has a strong observed association with the historical target A16.


### Contingency Table


| A9 category | Approved | Rejected | Total | Historical approval rate |
|---|---:|---:|---:|---:|
| `f` | 23 | 306 | 329 | 6.99% |
| `t` | 284 | 77 | 361 | 78.67% |
| **Total** | **307** | **383** | **690** | **44.49%** |


### Statistical Test Results


| Statistic | Result |
|---|---:|
| Chi-square statistic | 355.2038 |
| Degrees of freedom | 1 |
| p-value | Approximately 3.12 × 10⁻⁷⁹ |
| Cramér's V | 0.7175 |


The results indicate a strong statistical association between A9 and historical approval decisions.


The observed approval rates differ by approximately 71.68 percentage points between the two A9 categories.


### Interpretation and Limitations


The association is statistically strong, but several questions remain unanswered:


- What does A9 represent in the original application process?
- Was A9 available at the appropriate point in the decision process?
- Does A9 contain legitimate predictive information, problematic information, or both?
- How much of its predictive contribution overlaps with other features?
- Are the observed differences justified by relevant information about credit risk?


The dataset alone cannot answer these questions.


Statistical association does not establish causation or demonstrate that the historical decisions were fair or unfair.


---


## 8. Proxy-Feature Analysis


The project investigates whether the remaining features contain information that can predict A9.


A Random Forest model was trained to predict A9 from the other 14 features using five-fold cross-validation.


### Proxy Prediction Results


| Metric | Mean | Standard deviation |
|---|---:|---:|
| ROC-AUC | 0.8541 | 0.0209 |
| Balanced accuracy | 0.7795 | 0.0222 |
| Accuracy | 0.7768 | 0.0231 |


The model achieved meaningful predictive performance without directly using A9.


This suggests that information associated with A9 is also present in other features.


### Leading Proxy Candidates


Permutation importance identified the following features as the leading candidates in this experiment.


| Feature | Mean permutation importance | Standard deviation |
|---|---:|---:|
| A8 | 0.0435 | 0.0048 |
| A10 | 0.0393 | 0.0210 |
| A11 | 0.0247 | 0.0093 |
| A7 | 0.0150 | 0.0057 |
| A3 | 0.0146 | 0.0048 |


A8, A10, and A11 were selected as the top proxy candidates for the subsequent feature-ablation experiments.


### Important Distinction


These features are described as **proxy candidates** because they help predict A9.


Their predictive relationship does not establish that they are sensitive attributes, that they cause disparate outcomes, or that their use constitutes improper discrimination.


Further analysis of feature meaning, data provenance, and the decision process would be required.


---


## 9. Feature Ablation and Model Performance


Feature ablation examines how model performance changes when selected features are removed.


The project compares six scenarios.


### Cross-Validation Results


| Feature scenario | Mean ROC-AUC | ROC-AUC standard deviation | Mean accuracy | Mean balanced accuracy |
|---|---:|---:|---:|---:|
| All features | 0.9344 | 0.0219 | 0.8609 | 0.8616 |
| Without A9 | 0.8571 | 0.0231 | 0.7768 | 0.7676 |
| Without A9 and A8 | 0.8483 | 0.0264 | 0.7812 | 0.7728 |
| Without A9 and A10 | 0.8554 | 0.0220 | 0.7870 | 0.7741 |
| Without A9 and A11 | 0.8510 | 0.0218 | 0.7652 | 0.7591 |
| Without A9, A8, A10, and A11 | 0.7931 | 0.0161 | 0.7420 | 0.7297 |


### Interpretation


Removing A9 reduced predictive performance substantially.


Removing individual proxy candidates produced smaller and non-uniform changes.


Removing A9 together with A8, A10, and A11 resulted in a further reduction in ROC-AUC.


These findings indicate that the selected features collectively contain information useful for reproducing historical decisions.


However, feature ablation does not establish why a feature is predictive. It also cannot determine whether the information is legitimate, causally relevant, or appropriate for use in lending.


---


## 10. Performance–Disparity Trade-off


The project evaluates three principal scenarios:


1. **All features**
2. **Without A9**
3. **Without A9 and the top proxy candidates A8, A10, and A11**


The following table reports out-of-fold results at a classification threshold of 0.50.


### Results at Threshold 0.50


| Metric | All features | Without A9 | Without A9 + top proxies |
|---|---:|---:|---:|
| ROC-AUC | 0.9296 | 0.8555 | 0.7912 |
| Balanced accuracy | 0.8617 | 0.7676 | 0.7298 |
| Approval-rate gap | 0.8889 | 0.5130 | 0.4048 |
| TPR gap | 0.8931 | 0.5514 | 0.2930 |
| FPR gap | 0.7273 | 0.1713 | 0.1095 |


### Metric Definitions


**Approval-rate gap**


The absolute difference between the predicted approval rates for A9=`t` and A9=`f`.


**TPR gap**


The absolute difference between the groups' true-positive rates, where positive labels represent historical approvals.


**FPR gap**


The absolute difference between the groups' false-positive rates, where a false positive means a historically rejected application is predicted as approved.


For these metrics, a gap of 0.10 represents a difference of 10 percentage points.


### Relative Changes Compared With All Features


| Change | Without A9 | Without A9 + top proxies |
|---|---:|---:|
| ROC-AUC change | -0.0742 | -0.1384 |
| Approval-rate gap reduction | 42.3% | 54.5% |
| TPR-gap reduction | 38.2% | 67.2% |
| FPR-gap reduction | 76.5% | 84.9% |


### Interpretation


Removing A9 reduced predictive performance against historical approval labels while also reducing the measured group-level gaps.


Removing A9 together with the selected proxy candidates reduced the gaps further, with an additional loss in predictive performance.


This illustrates a measurable trade-off in the current experiment.


It does not establish that the model with the smallest gaps is the fairest model, or that the model with the highest ROC-AUC is the most appropriate for lending.


The results must be interpreted in the context of the dataset's historical labels, unknown feature meanings, small sample size, and evaluation design.


---


## 11. Threshold Sensitivity Analysis


A classification threshold determines how predicted probabilities are converted into binary decisions.


Changing the threshold can alter:


- The number of applications predicted as approved
- Balanced accuracy
- TPR and FPR
- Group-level gaps in prediction outcomes


The project evaluates thresholds from 0.20 to 0.80 in increments of 0.05.


### Best Balanced Accuracy Within the Tested Threshold Range


| Scenario | Selected threshold | Balanced accuracy | Approval-rate gap | TPR gap | FPR gap |
|---|---:|---:|---:|---:|---:|
| All features | 0.45 | 0.8689 | 0.9111 | 0.9143 | 0.7532 |
| Without A9 | 0.35 | 0.7841 | 0.5274 | 0.4867 | 0.1676 |
| Without A9 + top proxies | 0.50 | 0.7298 | 0.4048 | 0.2930 | 0.1095 |


The selected thresholds maximize balanced accuracy among the tested thresholds for each scenario.


These are exploratory selections, not independently validated optimal thresholds.


Selecting a threshold based on the same out-of-fold results used to report its performance may lead to optimistic estimates. A separate validation procedure should be used for threshold selection before final evaluation.


---


## 12. Pareto Analysis


Script 20 produces a two-objective Pareto analysis using:


1. **Maximize ROC-AUC**
2. **Minimize approval-rate gap**


A candidate is Pareto-dominated if another candidate has an equal or higher ROC-AUC and an equal or lower approval-rate gap, with at least one strict improvement.


### Current Pareto Frontier


| Scenario | Threshold | ROC-AUC | Approval-rate gap | Balanced accuracy |
|---|---:|---:|---:|---:|
| Without A9 + top proxies | 0.80 | 0.7912 | 0.1887 | 0.6124 |
| Without A9 | 0.80 | 0.8555 | 0.3779 | 0.7133 |
| All features | 0.80 | 0.9296 | 0.5676 | 0.7990 |


The analysis identifies three non-dominated scenario-threshold combinations among the 39 tested candidates.


### Interpretation


The frontier illustrates the trade-off between predictive performance and approval-rate gap.


The all-feature scenario has the highest ROC-AUC among these candidates, while the scenario without A9 and the selected proxy candidates has the smallest approval-rate gap.


Neither result alone determines which scenario should be preferred in practice.


The frontier is conditional on the tested scenarios, thresholds, dataset, and evaluation method. It does not establish a universally optimal decision rule.


In addition, ROC-AUC remains constant across thresholds within each feature scenario because it is calculated from predicted scores rather than thresholded decisions. This helps explain why the frontier retains the lowest-gap threshold from each scenario.


---


## 13. Generated Reports and Figures


The analysis scripts generate CSV summaries and PNG figures in the `reports/` directory.


### Principal CSV Outputs


| File | Purpose |
|---|---|
| `a9_threshold_sensitivity_results.csv` | Threshold-level performance and disparity results |
| `a9_fairness_performance_comparison.csv` | Performance–disparity comparison |
| `a9_fairness_performance_pareto.csv` | Pareto analysis from Script 19 |
| `a9_fairness_performance_summary.csv` | Summary of the performance–disparity analysis |
| `a9_final_pareto_frontier.csv` | Results with Pareto-optimal flags from Script 20 |
| `a9_final_scenario_comparison.csv` | Comparison of scenarios at threshold 0.50 |
| `a9_final_results_dashboard.csv` | Final summary of metrics and changes |


### Principal Figures


Located in `reports/figures/`:


- `a9_roc_auc_vs_approval_gap.png`
- `a9_balanced_accuracy_approval_gap.png`
- `a9_final_pareto_frontier.png`
- `a9_final_metrics_comparison.png`


These files provide visual summaries of the trade-offs observed in the experiments.


---


## 14. Limitations and Responsible Interpretation


Several limitations must be considered when interpreting the findings.


### 14.1 Small Dataset


The dataset contains only 690 observations. Estimates may be sensitive to the sample composition and cross-validation splits.


### 14.2 Anonymized Features


The meanings of A1–A15 are not established by the feature labels. In particular, the meaning of A9 must not be guessed.


### 14.3 Historical Approval Target


The target records previous approval decisions, not actual repayment or default. A model that predicts historical decisions well may reproduce existing patterns without accurately measuring creditworthiness.


### 14.4 Proxy Analysis Is Not Proof of Discrimination


The ability of other features to predict A9 shows that they contain associated information. It does not prove that the model uses improper proxies or that a causal pathway exists.


### 14.5 Group Gaps Are Not a Complete Fairness Assessment


Approval-rate, TPR, and FPR gaps describe differences in predictions relative to historical labels.


They do not independently establish discrimination, legal non-compliance, or fairness. Different fairness criteria may conflict, and their appropriateness depends on the context and the quality of the target labels.


### 14.6 Threshold Selection


Thresholds selected after examining the same evaluation results can produce optimistic performance estimates. Independent validation is needed before relying on a selected threshold.


### 14.7 Generalization and Statistical Uncertainty


Cross-validation estimates do not guarantee future performance. Mean differences between models do not, on their own, demonstrate statistical significance.


More rigorous evaluation should account for uncertainty and avoid using final evaluation data for feature selection, model tuning, or threshold selection.


### 14.8 Not for Direct Lending Decisions


This project is intended for educational, research, and portfolio demonstration purposes. It should not be used directly to approve or reject real credit applications.


---


## 15. Future Research


Potential extensions include:


1. **Independent validation:** Establish a clearly separated evaluation set or use nested cross-validation where appropriate.
2. **Uncertainty estimation:** Report confidence intervals and investigate the stability of performance and group-gap estimates.
3. **Feature provenance:** Determine what the anonymized features represent before making substantive claims about their use.
4. **Threshold selection:** Select thresholds on validation data and evaluate them on separate data.
5. **Additional models:** Compare alternative classifiers using consistent validation procedures.
6. **Calibration:** Evaluate whether predicted probabilities correspond to observed historical outcomes.
7. **Fairness evaluation:** Assess multiple relevant criteria with careful attention to label quality, group definitions, and application context.
8. **Real repayment outcomes:** If appropriate and lawfully available, investigate whether models predict actual repayment outcomes rather than only historical approval decisions.


These extensions would strengthen the methodological rigor of the project without assuming that the current results establish fairness or causality.


---


## 16. Repository Structure


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
│   ├── 12_test_A9_association.py
│   ├── 13_A9_sensitivity_analysis.py
│   ├── 14_a9_decision_impact_audit.py
│   ├── 15_a9_proxy_analysis.py
│   ├── 16_a9_proxy_feature_analysis.py
│   ├── 17_a9_proxy_ablation_validation.py
│   ├── 18_A9_threshold_sensivity_analysis.py
│   ├── 19_a9_fairness_performance_tradeoff.py
│   └── 20_a9_final_results_dashboard.py
├── reports/
│   └── figures/
├── sql/
└── README.md
```


The structure above is a documentation template. Adjust file and folder names to match the actual repository.


---


## 17. Conclusion


This project demonstrates an end-to-end analytical workflow for historical credit-decision prediction, model comparison, statistical association testing, proxy-feature investigation, feature ablation, threshold sensitivity, and performance–disparity analysis.


Three principal findings emerge from the current experiments:


1. **A9 contains strong predictive information** about historical credit approval decisions.
2. **Other features also contain information associated with A9**, making it useful to investigate feature ablation rather than evaluating A9 in isolation.
3. **Removing A9 and selected proxy candidates reduces measured group-level gaps while also reducing predictive performance against historical labels.**


The results illustrate a measurable trade-off within this dataset. They do not establish causation, prove discrimination, or identify a universally fair or optimal model.


The project provides a foundation for further research into machine learning evaluation, feature sensitivity, responsible predictive modeling, and financial data analytics.


**The central lesson is that predictive performance alone is not sufficient to evaluate a decision model. Data provenance, target validity, uncertainty, feature selection, and the consequences of model decisions also matter.**
