
import pandas as pd
import numpy as np


def perform_eda(df: pd.DataFrame):
    """Performs basic Exploratory Data Analysis (EDA) on a given pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.
    """
    print("=" * 60)
    print(" 📊 BASIC EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 60)

    # 1. Dataset Dimensions
    print(f"\n[1] DATASET DIMENSIONS")
    print(f"Total Rows    : {df.shape[0]}")
    print(f"Total Columns : {df.shape[1]}")

    # 2. Duplicate Rows
    print(f"\n[2] DUPLICATE ROWS")
    dup_count = df.duplicated().sum()
    print(
        f"Number of duplicate rows: {dup_count} ({(dup_count / df.shape[0]) * 100:.2f}% of total data)"
    )

    # 3. Column Data Types & Missing Values
    print(f"\n[3] COLUMN DATA TYPES & MISSING VALUES")
    missing_df = pd.DataFrame(
        {
            "Data Type": df.dtypes,
            "Non-Null Count": df.notnull().sum(),
            "Null Count": df.isnull().sum(),
            "Null Percentage (%)": (df.isnull().sum() / len(df)) * 100,
        }
    )
    print(missing_df.to_string())

    # 4. Summary Statistics for Numerical Columns
    print(f"\n[4] SUMMARY STATISTICS (NUMERICAL COLUMNS)")
    num_cols = df.select_dtypes(include=[np.number])
    if not num_cols.empty:
        print(df.describe().T.to_string())
    else:
        print("No numerical columns found in the dataset.")

    # 5. Summary Statistics for Categorical Columns
    print(f"\n[5] SUMMARY STATISTICS (CATEGORICAL/OBJECT COLUMNS)")
    cat_cols = df.select_dtypes(include=["object", "category"])
    if not cat_cols.empty:
        print(df.describe(include=["object", "category"]).T.to_string())
    else:
        print("No categorical columns found in the dataset.")

    # 6. Correlation Matrix (Numerical features only)
    print(f"\n[6] CORRELATION MATRIX (NUMERICAL FEATURES)")
    if num_cols.shape[1] > 1:
        corr = num_cols.corr()
        print(corr.to_string())
    else:
        print(
            "Not enough numerical columns to compute a correlation matrix (requires at least 2)."
        )

    print("\n" + "=" * 60)
    print(" ✅ EDA COMPLETE")
    print("=" * 60)
