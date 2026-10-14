# Processed Data

## Purpose

This directory is intended to contain datasets that have been cleaned,
transformed, or prepared for exploratory analysis and machine learning.

The original dataset should remain unchanged in data/raw/.

## Data Preparation

Depending on the analytical workflow, data preparation may include:

- Handling missing or unknown values
- Converting data types
- Encoding categorical features
- Preparing features for machine learning
- Separating input features from the target variable
- Applying transformations required by specific models

Only document transformations that are actually performed in the project.

## Reproducibility

Processed datasets should be generated through documented and repeatable
steps whenever possible.

Record important preprocessing decisions, including:

- How missing values are handled
- How categorical variables are encoded
- Whether numerical features are scaled
- Which features are included in the models
- How training and evaluation data are separated

Fit preprocessing steps using training data only when appropriate to
avoid data leakage.

## File Organization

Store processed datasets in this directory when they are generated
and required by the analytical workflow.

Use descriptive filenames that identify the dataset and processing
stage.

For example:

- credit_approval_cleaned.csv
- credit_approval_model_ready.csv

These are naming examples, not files that are necessarily present
in the repository.

## Important Notes

- Do not overwrite the original raw dataset.
- Do not assume that a processed file exists until it has been generated.
- Ensure that file paths in Python scripts match the actual project structure.
- Avoid committing unnecessary duplicate datasets.
- Do not include sensitive or restricted information.

The project predicts historical credit approval decisions. It does not
directly predict borrower default or repayment outcomes.
