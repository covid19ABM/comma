from comma.model import Model
import numpy as np
import pandas as pd
import pytest
import os


class TestModel:

    size = 2
    dir_parameters = 'parameters/'
    steps = 4

    def setup_lockdown_pattern(self):
        n = self.steps // 4
        return ["absent"] * n + ["medium"] * \
            n + ["hard"] * n + ["easy"] * n

    def setup_and_run_model(self, out_path):
        lockdown_pattern = self.setup_lockdown_pattern()
        model = Model()
        model.setup(self.size, self.dir_parameters, use_ipf=True)
        model.run(self.steps, lockdown_pattern, out_path=out_path)

    @pytest.fixture(scope="class")
    def full_simulation(self):
        np.random.seed(0)
        self.setup_and_run_model("expected.csv")
        yield
        for file in ["expected.csv", "actual.csv"]:
            if os.path.exists(file):
                os.remove(file)

    @pytest.fixture(scope="class")
    def expected_dataframe(self):
        data = {
            "step_id": [0, 0, 1, 1, 2, 2, 3, 3],
            "lockdown": ["absent", "absent", "medium", "medium",
                         "hard", "hard", "easy", "easy"],
            "agent_id": [0, 1, 0, 1, 0, 1, 0, 1],
            "delta_mental_health": [0.0, 0.0, 0.002, 0.002, 0.003,
                                    0.002, 0.002, 0.003],
            "cumulative_mental_health": [
                0.0,
                0.002,
                0.00017395607466307600,
                0.00192182551544801,
                0.0014293586434475100,
                0.0025121416075092200,
                0.0018830078256391300,
                0.003486168909611150
            ]
        }
        return pd.DataFrame(data).round(4)

    def test_simulation_output(self, full_simulation):
        np.random.seed(0)
        self.setup_and_run_model("actual.csv")

        # load both csv files into dataframes
        expected_df = pd.read_csv("expected.csv")
        actual_df = pd.read_csv("actual.csv")

        assert expected_df.shape == actual_df.shape, \
            "Dataframes have different shapes"

        for idx, (expected_row, actual_row) in \
                enumerate(zip(expected_df.iterrows(), actual_df.iterrows())):
            assert expected_row[1].equals(actual_row[1]), \
                f"Row {idx} is different"

    def test_expected_output(self, expected_dataframe):
        np.random.seed(0)
        self.setup_and_run_model("actual.csv")

        actual_df = pd.read_csv("actual.csv", sep=";", decimal=",").round(4)

        assert expected_dataframe.shape == actual_df.shape, \
            "Outputs have different shapes"

        for idx, (expected_row, actual_row) in \
                enumerate(
                    zip(expected_dataframe.iterrows(), actual_df.iterrows())
                ):
            assert expected_row[1].equals(actual_row[1]), \
                f"Row {idx} is different"
