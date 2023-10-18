import os
import pandas as pd
import pytest
from pathlib import Path


class TestCSVIntegrity:
    """
    This test ensures that the column names across the lockdown and
    action files are the same, and in the same order.
    This is important for the subsequent matrix
    multiplications done in the model.
    """

    @pytest.fixture(scope="class")
    def initial_csv_data(self):
        directory = Path("parameters/")
        csv_files = [
            f
            for f in os.listdir(directory)
            if (
                f.endswith(".csv")
                and (f.startswith("lockdown") or f.startswith("actions"))
            )
        ]

        if not list(directory.glob("lockdown*.csv")) + list(
            directory.glob("actions*.csv")
        ):
            raise ValueError(
                f"No CSV files found in the directory '"
                f"{str(directory.absolute())}'."
            )

        first_df = pd.read_csv(os.path.join(directory, csv_files[0]), sep=";")

        column_names = first_df.columns.str.lower()
        num_columns = len(column_names)
        num_rows = len(first_df)
        actions = set(first_df["actions"].unique())

        return {
            "directory": directory,
            "csv_files": csv_files,
            "column_names": column_names,
            "num_columns": num_columns,
            "num_rows": num_rows,
            "actions": actions,
        }

    def test_input_example_matrices(self, initial_csv_data):
        for file in initial_csv_data["csv_files"][1:]:
            df = pd.read_csv(os.path.join(initial_csv_data["directory"], file), sep=";")
            df.columns = df.columns.str.lower()
            # Check for the same number of columns
            assert (
                len(df.columns) == initial_csv_data["num_columns"]
            ), f"{file} has a different number of columns."

            # Check for the same number of rows
            assert (
                len(df) == initial_csv_data["num_rows"]
            ), f"{file} has a different number of rows."

            # Check for the same column names
            if set(df.columns) != set(initial_csv_data["column_names"]):
                missing = set(initial_csv_data["column_names"]).difference(df.columns)
                additional = set(df.columns).difference(
                    initial_csv_data["column_names"]
                )
                raise AssertionError(
                    f"{file} has different column names. "
                    f"Missing: {missing}. Additional: {additional}"
                )

            # Check for the same column order
            if list(df.columns) != list(initial_csv_data["column_names"]):
                diff_order = [
                    i
                    for i, (col_df, col_base) in enumerate(
                        zip(df.columns, initial_csv_data["column_names"])
                    )
                    if col_df != col_base
                ]
                raise AssertionError(
                    f"{file} has different column " f"order at indices: {diff_order}."
                )

            # Check for the same "actions" entries
            assert (
                set(df["actions"].unique()) == initial_csv_data["actions"]
            ), f"{file} has different 'actions' entries."
