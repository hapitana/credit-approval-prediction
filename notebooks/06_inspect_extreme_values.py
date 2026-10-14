import polars as pl

file_path = "../data/raw/crx.data"

# Column names for the dataset, which has no header row.
columns = [
    "A1", "A2", "A3", "A4",
    "A5", "A6", "A7", "A8",
    "A9", "A10", "A11", "A12",
    "A13", "A14", "A15", "A16"
]


# Read the dataset and treat "?" as a missing value.
df = pl.read_csv(
    file_path,
    has_header=False,
    new_columns=columns,
    null_values="?"
)

df = df.with_columns(
    pl.col("A15").cast(pl.Float64, strict=False)
)

# Select the 10 largest non-null values in A15.
largest_values = (
    df.filter(pl.col("A15").is_not_null())
    .sort("A15", descending=True)
    .select([
        "A15", "A16", "A2", "A3",
        "A8", "A11", "A14"
    ])
    .head(10)
)

print("Top 10 largest A15 values:")
print(largest_values)

thresholds = [1000, 5000, 10000, 50000]

# Count records whose A15 value exceeds each threshold.
for threshold in thresholds:
    count = df.filter(
        pl.col("A15") > threshold
    ).height

    print(f"A15 > {threshold}: {count} applications")