import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

columns = [f"A{i}" for i in range(1, 17)]

df = pd.read_csv(
    "../data/raw/crx.data",
    header=None,
    names=columns,
    na_values="?",
)

df = df[df["A16"].isin(["+", "-"])].copy()
df["A9"] = df["A9"].fillna("MISSING")

# Contingency table without additional totals
table = pd.crosstab(df["A9"], df["A16"])

chi2, p_value, dof, expected = chi2_contingency(table)

n = table.to_numpy().sum()
min_dimension = min(table.shape[0] - 1, table.shape[1] - 1)
cramers_v = np.sqrt(chi2 / (n * min_dimension))

print("Contingency table:")
print(table)

print(f"\nChi-square statistic: {chi2:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p_value:.6g}")
print(f"Cramer's V: {cramers_v:.4f}")

print("\nExpected frequencies:")
print(pd.DataFrame(
    expected,
    index=table.index,
    columns=table.columns,
).round(2))