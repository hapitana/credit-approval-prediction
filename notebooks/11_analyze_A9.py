from pathlib import Path

import polars as pl


# 1. Load the data
file_path = (
    Path(_file_).resolve().parent.parent
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

# 4. Count applications by A9 category and approval outcome
counts = (
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

print("Number of applications by A9 category and A16 outcome:")
print(counts)

# 5. Calculate approval rates by A9 category
approval_rates = (
    df.group_by("A9")
    .agg(
        pl.len().alias("total"),
        (pl.col("A16") == "+").sum().alias("approved"),
        (pl.col("A16") == "+")
        .mean()
        .alias("approval_rate"),
    )
    .with_columns(
        (pl.col("approval_rate") * 100)
        .round(2)
        .alias("approval_rate_pct")
    )
    .sort("approval_rate", descending=True)
)

print("\nApproval rates by A9 category")
print(
    approval_rates.select([
        "A9",
        "total",
        "approved",
        "approval_rate_pct",
    ])
)
