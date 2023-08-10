from comma.model import Model
import numpy as np
import pandas as pd
import pytest
import os


class TestModel:

    size = 25
    dir_parameters = 'parameters/'
    steps = 8

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