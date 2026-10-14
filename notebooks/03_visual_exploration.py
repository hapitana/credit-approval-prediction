import polars as pl
import matplotlib.pyplot as plt

file_path = "../data/raw/crx.data"

columns = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9",
            "A10", "A11", "A12", "A13", "A14", "A15", "A16"]

df = pl.read_csv(file_path, has_header=False,
                            new_columns = columns,
                            null_values="?")

# Convert A3 to numeric and remove missing values
df = df.with_columns(
                    pl.col("A3").cast(pl.Float64, strict=False)
                    )

approved = (
            df.filter(pl.col("A16") == "+")
                .get_column("A3")
                .drop_nulls()
                .to_list()
            )

rejected = (
            df.filter(pl.col("A16") == "-")
                .get_column("A3")
                .drop_nulls()
                .to_list()
            )

plt.hist(approved, 
        bins=20,
        alpha=0.6,
        label="Approved (+)"
        )


plt.hist(rejected, 
        bins=20,
        alpha=0.6,
        label="Rejected (-)"
        )

plt.xlabel("A3 value")
plt.ylabel("Number of applications")
plt.title("A3 Distribution by Credit Approval Outcome")
plt.legend()
plt.tight_layout()
plt.show()
