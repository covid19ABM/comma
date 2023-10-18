from comma.model import Model
import numpy as np
import pandas as pd
import pytest
import os


class TestModel:

    size = 2
    dir_parameters = 'parameters/'
    steps = 4
    seed = 0

    def setup_lockdown_pattern(self):
        n = self.steps // 4
        return ["absent"] * n + ["medium"] * \
            n + ["hard"] * n + ["easy"] * n

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
            "lockdown": ["absent", "absent", "medium", "medium",
                         "hard", "hard", "easy", "easy"],
            "agent_id": [0, 1, 0, 1, 0, 1, 0, 1],
            "delta_mental_health": [0.0, 0.0, -30.8,
                                    -20.15, -8.18, -14.28,
                                    -19.560000000000002,
                                    -15.670000000000002
                                    ],
            "cumulative_mental_health": [
                -4.19, -2.1699999999999995,
                -34.992672937711895, -22.322390655700346,
                -43.17483742252662, -36.60426136942761,
                -62.73619858244343, -52.276576575172996
            ],
            "covid_status": [0, 0, 0, 0, 0, 0, 0, 0],
            "days_since_first_infection": [np.nan, np.nan, np.nan, np.nan,
                                           np.nan, np.nan, np.nan, np.nan]
        }
        return pd.DataFrame(data).round(4)

    @pytest.mark.filterwarnings("ignore:Given sim_size")
    def test_simulation_output(self, full_simulation):
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

    @pytest.mark.filterwarnings("ignore:Given sim_size")
    def test_expected_output(self, expected_dataframe):
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
