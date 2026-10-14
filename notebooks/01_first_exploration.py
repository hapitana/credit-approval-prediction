import polars as pl

file_path = "../data/raw/crx.data"

columns = ["A1", "A2", "A3", "A4", "A5", "A6",
           "A7", "A8", "A9", "A10", "A11", "A12",
           "A13", "A14", "A15", "A16"]

df = pl.read_csv(
    file_path,
    has_header=False,

    new_columns=columns,
    null_values="?"
                 )

print("Dataset size:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nMissing values in each column:")
print(df.null_count())

# Count missing values in each column
missing = df.null_count()

# Calculate the percentage of missing values
missing_summary = pl.DataFrame({"column":df.columns,
                                "missing_count":[missing[c][0] for c in df.columns],
                                "missing_percent":[round(missing[c][0] / df.height * 100,2)
                                                    for c in df.columns]
                                })

print("\nMissing-value summary:")
print(missing_summary)