from comma.model import Model
import numpy as np
import pandas as pd
import pytest
import os


class TestModel:
    size = 2
    dir_parameters = "parameters/"
    steps = 4
    seed = 0

    def setup_lockdown_pattern(self):
        n = self.steps // 4
        return ["absent"] * n + ["medium"] * n + ["hard"] * n + ["easy"] * n

    def setup_and_run_model(self, out_path):
        lockdown_pattern = self.setup_lockdown_pattern()
        model = Model(self.seed)
        model.setup(self.size, self.dir_parameters, use_ipf=True)
        model.run(self.steps, lockdown_pattern, out_path=out_path)

    @pytest.fixture(scope="class")
    def full_simulation(self):
        self.setup_and_run_model("expected.csv")
        yield
        for file in ["expected.csv", "actual.csv"]:
            if os.path.exists(file):
                os.remove(file)

    @pytest.fixture(scope="class")
    def expected_dataframe(self):
        data = {
            "step_id": [0, 0, 1, 1, 2, 2, 3, 3],
            "lockdown": [
                "absent",
                "absent",
                "medium",
                "medium",
                "hard",
                "hard",
                "easy",
                "easy",
            ],
            "agent_id": [0, 1, 0, 1, 0, 1, 0, 1],
            "delta_mental_health": [
                0.0,
                0.0,
                -30.8,
                -20.15,
                -8.18,
                -11.83,
                -28.95,
                -11.77,
            ],
            "cumulative_mental_health": [
                -4.19,
                -2.1699999999999995,
                -34.992672937711895,
                -22.322390655700346,
                -43.17483742252662,
                -34.15,
                -72.13,
                -45.93,
            ],
            "covid_status": [0, 0, 0, 0, 0, 0, 0, 0],
            "days_since_first_infection": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
        return pd.DataFrame(data).round(2)

    def compare_rows(self, expected_row, actual_row):
        """
        Compare two rows at a time
        """
        differences = []
        for col in expected_row[1].index:
            expected_value = expected_row[1][col]
            actual_value = actual_row[1][col]

            # skip null values
            if pd.isnull(expected_value) and pd.isnull(actual_value):
                continue

            if isinstance(expected_value, (int, float)) and isinstance(
                actual_value, (int, float)
            ):
                # check if numeric values are close
                # by setting a tolerance
                if not np.isclose(expected_value, actual_value, atol=0.2):
                    differences.append(
                        f"{col}: Expected={expected_value}, " f"Actual={actual_value}"
                    )
            else:
                # other type of data check for exact value
                if expected_value != actual_value:
                    differences.append(
                        f"{col}: Expected={expected_value}, " f"Actual={actual_value}"
                    )

        return differences

    @pytest.mark.filterwarnings("ignore:Given sim_size")
    def test_simulation_output(self, full_simulation):
        self.setup_and_run_model("actual.csv")

        # load both csv files into dataframes
        expected_df = pd.read_csv("expected.csv", sep=";", decimal=",").round(2)
        actual_df = pd.read_csv("actual.csv", sep=";", decimal=",").round(2)

        assert expected_df.shape == actual_df.shape, "Dataframes have different shapes"

        # check all rows of both dataframes for differences
        for idx, (expected_row, actual_row) in enumerate(
            zip(expected_df.iterrows(), actual_df.iterrows())
        ):
            differences = self.compare_rows(expected_row, actual_row)

            if differences:
                difference_message = ", ".join(differences)
                assert False, (
                    f"Row {idx} is different. " f"Differences: {difference_message}"
                )

    @pytest.mark.filterwarnings("ignore:Given sim_size")
    def test_expected_output(self, expected_dataframe):
        self.setup_and_run_model("actual.csv")

        actual_df = pd.read_csv("actual.csv", sep=";", decimal=",").round(2)

        assert (
            expected_dataframe.shape == actual_df.shape
        ), "Outputs have different shapes"

        for idx, (expected_row, actual_row) in enumerate(
            zip(expected_dataframe.iterrows(), actual_df.iterrows())
        ):
            differences = self.compare_rows(expected_row, actual_row)

            if differences:
                difference_message = ", ".join(differences)
                assert False, (
                    f"Row {idx} is different. " f"Differences: {difference_message}"
                )
