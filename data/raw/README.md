# Raw Dataset

## Dataset Information

This project uses the Credit Approval dataset from the UCI Machine
Learning Repository.

*Official source:*
https://archive.ics.uci.edu/dataset/27/credit+approval

The dataset contains historical credit application records used to
study patterns associated with credit approval decisions.

## Dataset Characteristics

| Property | Description |
|---|---|
| Observations | 690 |
| Columns | 16 |
| Input features | A1–A15 |
| Target variable | A16 |
| Target classes | + (approved), - (rejected) |
| Data types | Numeric and categorical |

The feature labels A1–A15 are anonymized. Their real-world meanings
should not be inferred without additional documentation.

## Download Instructions

1. Visit the official UCI dataset page.
2. Download the dataset files.
3. Locate the raw data file, commonly named crx.data.
4. Place the file in this directory:

   data/raw/crx.data

5. Check the file paths used by the Python scripts and update them
   if necessary.

## Data Handling

- Keep the raw dataset unchanged whenever possible.
- Perform cleaning and transformations in separate processing steps.
- Do not commit sensitive or restricted data to the repository.
- Document any data preparation decisions that affect the analysis.

## Important Note

The target variable represents historical approval decisions.
It does not directly represent loan repayment or borrower default.

The dataset is used for educational and portfolio demonstration
purposes, not for making real credit decisions.
