
import os
import pandas as pd


def read_uploaded_file(file_path):
    # Extract the file extension and convert it to lowercase
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    # Dictionary mapping extensions to their respective pandas read functions
    readers = {
        ".csv": lambda f: pd.read_csv(f),
        ".xlsx": lambda f: pd.read_excel(f),
        ".xls": lambda f: pd.read_excel(f),
        ".json": lambda f: pd.read_json(f),
        ".parquet": lambda f: pd.read_parquet(f),
        ".txt": lambda f: pd.read_csv(
            f, sep="\t"
        ),  # Assuming tab-delimited for .txt
        ".tsv": lambda f: pd.read_csv(f, sep="\t"),
    }

    # Check if the extension is supported
    if file_extension in readers:
        try:
            df = readers[file_extension](file_path)
            print(
                f"Successfully read {file_path} as a {file_extension} file."
            )
            return df
        except Exception as e:
            raise ValueError(f"Error reading file '{file_path}': {e}")
    else:
         raise ValueError(f"Unsupported file extension: '{file_extension}'")

# --- Example Usage ---
# df = read_uploaded_file("path_to_your_file.csv")
