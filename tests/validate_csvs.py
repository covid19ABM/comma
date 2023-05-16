import os
import pandas as pd
import pytest

def test_input_example_matrices():
    directory = "parameters_example/"

    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    assert csv_files, "No CSV files found in the directory."

    # Load the file
    first_df = pd.read_csv(os.path.join(directory, csv_files[0]), sep=";")

    # Get the column names, number of columns and rows, data types, and unique entries in "actions" column
    column_names = first_df.columns
    num_columns = len(column_names)
    num_rows = len(first_df)
    #dtypes = first_df.dtypes
    actions = set(first_df['actions'].unique())

    for file in csv_files[1:]:
        df = pd.read_csv(os.path.join(directory, file), sep=";")

        # Check for the same number of columns
        assert len(df.columns) == num_columns, f"{file} has a different number of columns."

        # Check for the same number of rows
        assert len(df) == num_rows, f"{file} has a different number of rows."

        # Check for the same column names
        if set(df.columns) != set(column_names):
            missing = set(column_names).difference(df.columns)
            additional = set(df.columns).difference(column_names)
            raise AssertionError(f"{file} has different column names. Missing: {missing}. Additional: {additional}")

        # Check for the same column order
        if list(df.columns) != list(column_names):
            diff_order = [i for i, (col_df, col_base) in enumerate(zip(df.columns, column_names)) if col_df != col_base]
            raise AssertionError(f"{file} has different column order at indices: {diff_order}.")

        # Check for the same "actions" entries
        assert set(df['actions'].unique()) == actions, f"{file} has different 'actions' entries."