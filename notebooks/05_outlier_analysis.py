import polars as pl

file_path = "../data/raw/crx.data"

columns = [
    "A1", "A2", "A3", "A4",
    "A5", "A6", "A7", "A8",
    "A9", "A10", "A11", "A12",
    "A13", "A14", "A15", "A16"
]

df = pl.read_csv(
    file_path,
    has_header=False,
    new_columns=columns,
    null_values="?"
)

df = df.with_columns(
    pl.col("A15").cast(pl.Float64, strict=False)
)

for outcome in ["+", "-"]:
    group = df.filter(pl.col("A16") == outcome)

    print(f"\nOutcome: {outcome}")

    summary = group.select([
        pl.col("A15").count().alias("valid_count"),
        pl.col("A15").min().alias("minimum"),
        pl.col("A15").quantile(0.25).alias("q25"),
        pl.col("A15").median().alias("median"),
        pl.col("A15").quantile(0.75).alias("q75"),
        pl.col("A15").quantile(0.90).alias("q90"),
        pl.col("A15").quantile(0.95).alias("q95"),
        pl.col("A15").max().alias("maximum"),
        pl.col("A15").mean().alias("mean"),
    ])

    print(summary)

    zero_count = group.filter(
        pl.col("A15") == 0
    ).height

    print(f"Number of zero values: {zero_count}")