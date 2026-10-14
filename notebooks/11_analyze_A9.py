import pandas as pd

columns = [f"A{i}" for i in range(1, 17)]

df = pd.read_csv(
    "../data/raw/crx.data",
    header=None,
    names=columns,
    na_values="?",
)

df = df[df["A16"].isin(["+", "-"])].copy()

# Mark missing values in A9
df["A9"] = df["A9"].fillna("MISSING")

# Count applications by A9 category and approval outcome
counts = pd.crosstab(
    df["A9"],
    df["A16"],
    margins=True,
)

print("Number of applications by A9 category and A16 outcome:")
print(counts)

# Calculate the approval rate for each A9 category
approval_rates = (
    df.assign(approved=(df["A16"] == "+").astype(int))
    .groupby("A9", dropna=False)
    .agg(
        total=("approved", "size"),
        approved=("approved", "sum"),
        approval_rate=("approved", "mean"),
    )
    .sort_values("approval_rate", ascending=False)
)

approval_rates["approval_rate_pct"] = (
    approval_rates["approval_rate"] * 100
).round(2)

print("\nApproval rates by A9 category")
print(approval_rates[
    ["total", "approved", "approval_rate_pct"]
].to_string())