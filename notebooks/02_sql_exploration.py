import numpy as np
import polars as pl
import duckdb

# Load the dataset
file_path = "../data/raw/crx.data"

columns = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9",
            "A10", "A11", "A12", "A13", "A14", "A15", "A16"]

df = pl.read_csv(file_path, has_header=False,
                            new_columns = columns,
                            null_values="?")

# Connect to a local DuckDB database
con = duckdb.connect("credit_approval.duckdb")

# Create a persistent SQL table from the DataFrame
con.register("credit_df", df) 

con.execute("""CREATE OR REPLACE TABLE credit_applications
                AS SELECT * FROM credit_df""")

# Run our SQL query
result = con.execute("""SELECT COUNT(*) AS total_applications
                          FROM credit_applications""").fetch_df()

print("Total applications:")
print(result)

# Count aplications by outcome
outcome = con.execute("""
                SELECT  A16 AS approval_outcome,
                        COUNT(*) AS application_count
                FROM credit_applications
                GROUP BY A16
                ORDER BY application_count DESC
                """).fetchdf()

print("\nApplications by outcome:")
print(outcome)


# Calculate the overall approval rate
approval_rate = con.execute("""
                SELECT  COUNT(*) AS total_application,
                        SUM(CASE WHEN A16 = '+' THEN 1 ELSE 0 END)
                            AS approved_applications,
                        ROUND(100.0 * SUM(CASE WHEN A16 = '+' THEN 1 ELSE 0 END) / COUNT(),2)
                            AS approval_rate_percent
                FROM credit_applications
                            """).fetchdf()

print("\nOverall approval rate:")
print(approval_rate)


# Count missing values in selected colums
missing_results = con.execute("""
                            SELECT
                                COUNT(*) FILTER (WHERE A1 IS NULL) AS A1_missing,
                                COUNT(*) FILTER (WHERE A2 IS NULL) AS A2_missing,
                                COUNT(*) FILTER (WHERE A4 IS NULL) AS A4_missing,
                                COUNT(*) FILTER (WHERE A5 IS NULL) AS A5_missing,
                                COUNT(*) FILTER (WHERE A14 IS NULL) AS A14_missing,
                                COUNT(*) AS total_applications
                            FROM credit_applications
                            """).fetchdf()
print("\nMissing values in selected columns:")
print(missing_results)


# Approval outcomes by A1 category
a1_results = con.execute("""
                          SELECT
                            A1,
                            COUNT(*) AS applications,
                            SUM(CASE WHEN A16 = '+' THEN 1 ELSE 0 END) 
                              AS approved,
                            ROUND(100.0 * SUM(CASE WHEN A16 = '+' THEN 1 ELSE 0 END) / COUNT(*),2)
                              AS approval_rate_percent
                          FROM credit_applications
                          WHERE A1 IS NOT NULL
                          GROUP BY A1
                          ORDER BY applications DESC
                        """).fetchdf()

print("\nApproval outcomes by A1:")
print(a1_results)


a3_results = con.execute("""
                          SELECT
                            A16 AS outcome,
                            COUNT(*) AS applications,
                            ROUND(AVG(TRY_CAST(A3 AS DOUBLE)),2)
                              AS average_A3,
                            ROUND(MEDIAN(TRY_CAST(A3 AS DOUBLE)),2)
                              AS median_A3,
                            ROUND(MIN(TRY_CAST(A3 AS DOUBLE)),2)
                              AS minimum_A3,
                            ROUND(MAX(TRY_CAST(A3 AS DOUBLE)),2)
                              AS maximum_A3
                          FROM credit_applications
                          GROUP BY A16
                          ORDER BY A16
                        """).fetchdf()

print("\nA3 statistics by approval outcome:")
print(a3_results)

con.close()
