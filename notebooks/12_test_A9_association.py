from pathlib import Path

import polars as pl
import numpy as np
from scipy.stats import chi2_contingency


# 1. Load the data
file_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "crx.data"
)

columns = [f"A{i}" for i in range(1, 17)]

df = pl.read_csv(
    file_path,
    has_header=False,
    new_columns=columns,
    null_values="?",
)

# 2. Keep only valid target labels
df = df.filter(pl.col("A16").is_in(["+", "-"]))

# 3. Mark missing values in A9
df = df.with_columns(
    pl.col("A9").fill_null("MISSING")
)

# 4. Create the contingency table
table_df = (
    df.group_by(["A9", "A16"])
    .len(name="count")
    .pivot(
        on="A16",
        index="A9",
        values="count",
    )
    .fill_null(0)
    .sort("A9")
)

# Ensure both target columns exist and are in a consistent order.
for target in ["+", "-"]:
    if target not in table_df.columns:
        table_df = table_df.with_columns(
            pl.lit(0).alias(target)
        )

table_df = table_df.select(["A9", "+", "-"])

# Convert to a NumPy array for SciPy.
observed = table_df.select(["+", "-"]).to_numpy()

# 5. Perform the chi-square test
chi2, p_value, dof, expected = chi2_contingency(observed)

# 6. Calculate Cramér's V
n = observed.sum()
min_dimension = min(observed.shape[0] - 1, observed.shape[1] - 1)

cramers_v = np.sqrt(chi2 / (n * min_dimension))

# 7. Display results
print("Contingency table:")
print(table_df)

print(f"\nChi-square statistic: {chi2:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p_value:.6g}")
print(f"Cramer's V: {cramers_v:.4f}")

# 8. Display expected frequencies
expected_df = pl.DataFrame({
    "A9": table_df["A9"],
    "+": expected[:, 0],
    "-": expected[:, 1],
}).with_columns(
    pl.col(["+", "-"]).round(2)
)

print("\nExpected frequencies:")
print(expected_df)
