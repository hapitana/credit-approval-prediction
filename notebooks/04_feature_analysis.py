import polars as pl

file_path = "../data/raw/crx.data"

columns = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9",
            "A10", "A11", "A12", "A13", "A14", "A15", "A16"]

df = pl.read_csv(file_path, has_header=False,
                            new_columns = columns,
                            null_values="?")

# Convert known numeric columns
numeric_columns = ["A2", "A3", "A8", "A11", "A14", "A15"]

df = df.with_columns(
                        [
                            pl.col(c).cast(pl.Float64,strict=False)
                            for c in numeric_columns
                        ]
                    )

# Summarize each numeric feature by outcome
for feature in numeric_columns:
    print(f"\n---(feature) by approval outcome ---")
    summary = (
                df.group_by("A16")
                .agg([

                    pl.col(feature).count().alias("valid_count"),
                    pl.col(feature).mean().round(2).alias("mean"),
                    pl.col(feature).median().round(2).alias("median"),
                    pl.col(feature).std().round(2).alias("std_dev"),
                ])
                .sort("A16")
    )

    print(summary)