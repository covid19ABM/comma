import warnings

import pandas as pd
import pytest
from unittest.mock import patch
from comma.hypothesis import Hypothesis


class TestHypothesis:
    @pytest.fixture
    def setup_time_period(self):
        data = {
            "starting_date": "2020-01-01",
            "steps": 5,
            "date_format": "%Y-%m-%d",
            "expected_result": ("2020-01-01", "2020-01-06"),
        }
        return data

    @pytest.fixture
    def mock_df(self):
        return pd.DataFrame(
            {
                "Version": [2],
                "Date_of_report": ["2022-01-03T09:00:00Z"],
                "Date_of_statistics": ["2022-01-01"],
                "Security_region_code": ["VR01"],
                "Security_region_name": ["Groningen"],
                "Tested_with_result": [976],
                "Tested_positive": [378],
            }
        )

    @pytest.fixture
    def file_list(self):
        return [
            "data-rivm/tests/rivm_daily_2021-01-01.csv.gz",
            "data-rivm/tests/rivm_daily_2021-01-02.csv.gz",
            "data-rivm/tests/rivm_daily_2021-01-15.csv.gz",
            "data-rivm/tests/rivm_daily_2021-01-30.csv.gz",
            "data-rivm/tests/rivm_daily_2021-02-01.csv.gz",
            "data-rivm/tests/rivm_daily_2021-02-15.csv.gz",
            "data-rivm/tests/rivm_daily_2021-02-28.csv.gz",
        ]

    def test_filter_dates_within_range(self, file_list):
        # Test whether function correctly filter out dates
        # that are *clearly* within the bounds of `time_period`
        start = "2021-01-01"
        steps = 2
        expected_output = [
            "data-rivm/tests/rivm_daily_2021-01-01.csv.gz",
            "data-rivm/tests/rivm_daily_2021-01-02.csv.gz",
        ]
        hypothesis_instace = Hypothesis(start, steps)
        output = hypothesis_instace.filter_dates(file_list)
        assert output == expected_output

    def test_filter_dates_on_boundary(self, file_list):
        # Test whether function correctly includes
        # or excludes dates that are exactly on the edge
        start = "2021-01-16"
        steps = 43
        expected_output = [
            "data-rivm/tests/rivm_daily_2021-01-30.csv.gz",
            "data-rivm/tests/rivm_daily_2021-02-01.csv.gz",
            "data-rivm/tests/rivm_daily_2021-02-15.csv.gz",
            "data-rivm/tests/rivm_daily_2021-02-28.csv.gz",
        ]
        hypothesis_instance = Hypothesis(start, steps)
        output = hypothesis_instance.filter_dates(file_list)
        assert output == expected_output

    def test_range_error(self, file_list):
        # Test that values outside the bounds raise an error
        start = "2021-02-01"
        steps = 1860
        hypothesis_instance = Hypothesis(start, steps)
        with pytest.raises(
            ValueError, match=r"time_period .* is outside available dates"
        ):
            hypothesis_instance.filter_dates(file_list)

    @patch("comma.hypothesis.Hypothesis.get_file_paths")
    @patch("comma.hypothesis.Hypothesis.filter_dates")
    @patch("comma.hypothesis.pd.read_csv")
    # Mock tqdm to just return an iterator
    @patch("comma.hypothesis.tqdm", side_effect=lambda x, *args, **kwargs: x)
    def test_get_covid_data(
        self,
        mocked_tqdm,
        mocked_read_csv,
        mocked_filter_dates,
        mocked_get_file_paths,
        mock_df,
    ):
        # Test that the whole data extraction
        location = "Groningen"
        start = "2022-01-01"
        steps = 2

        # mock responses
        mocked_get_file_paths.return_value = ["some_path"]
        mocked_filter_dates.return_value = ["filtered_path"]
        mocked_read_csv.return_value = mock_df

        hypothesis_instance = Hypothesis(start, steps)
        df = hypothesis_instance.get_covid_data(location)
        formatted_date = df.iloc[0]["Date_of_statistics"].strftime("%Y-%m-%d")
        assert not df.empty
        assert df.iloc[0]["Version"] == 2
        assert df.iloc[0]["Date_of_report"] == "2022-01-03T09:00:00Z"
        assert formatted_date == "2022-01-01"
        assert df.iloc[0]["Security_region_code"] == "VR01"
        assert df.iloc[0]["Security_region_name"] == "Groningen"
        assert df.iloc[0]["Tested_with_result"] == 976
        assert df.iloc[0]["Tested_positive"] == 378

        # assert mock methods
        mocked_get_file_paths.assert_called_once()
        mocked_filter_dates.assert_called_once_with(mocked_get_file_paths.return_value)
        mocked_read_csv.assert_called_once_with(
            "https://github.com/mzelst/covid-19/raw/"
            "master/data-rivm/tests/filtered_path",
            compression="gzip",
            header=0,
            sep=",",
            quotechar='"',
        )

    def test_location_not_in_dataset(self):
        # trying to get data for an unknown location
        # should raise an error
        with pytest.raises(ValueError):
            start = "2023-01-01"
            steps = 2
            hypothesis_instance = Hypothesis(start, steps)
            hypothesis_instance.get_positive_cases("Los Angeles")

    def test_scaling(self):
        # Tests the scale_cases_to_population function
        real_size = 100
        sim_size = 50
        daily_positive_cases = pd.Series([10, 20, 30, 40, 50])
        out = Hypothesis.scale_cases_to_population(
            daily_positive_cases, real_size, sim_size
        )
        expected = pd.Series([5, 10, 15, 20, 25])
        assert all(out == expected), f"Expected {expected}, but got {out}"

    def test_scaling_warning(self):
        # Tests the warning of scale_cases_to_population function
        daily_positive_cases = pd.Series([1, 1, 1, 1, 1])
        real_size = 1000
        sim_size = 50
        with warnings.catch_warnings(record=True) as w:
            Hypothesis.scale_cases_to_population(
                daily_positive_cases, real_size, sim_size
            )
            # is there warning there?
            assert len(w) == 1, f"Expected warning, but got {len(w)} warnings"

    def test_no_scaling_warning(self):
        # Tests the warning of scale_cases_to_population function
        # I expect no warning here
        daily_positive_cases = pd.Series([500, 600, 700, 800, 900])
        real_size = 1000
        sim_size = 500
        with warnings.catch_warnings(record=True) as w:
            Hypothesis.scale_cases_to_population(
                daily_positive_cases, real_size, sim_size
            )
            assert len(w) == 0, f"Expected no warnings, but got {len(w)}"

    def test_adjust_cases(self):
        # Test that when positives are less than steps
        # the number is repeated n times * steps
        less_positive_cases = pd.Series([846])
        steps = 2
        start = "2023-01-01"
        expected = pd.Series([846, 846])
        hypothesis_instance = Hypothesis(start, steps)
        out = hypothesis_instance.adjust_cases(less_positive_cases)
        assert all(out == expected), f"Expected {expected}, but got {out}"

    def test_compute_time_period(self, setup_time_period):
        start = setup_time_period["starting_date"]
        steps = setup_time_period["steps"]
        expected_result = setup_time_period["expected_result"]

        hypothesis_instance = Hypothesis(start, steps)

        assert hypothesis_instance.time_period == expected_result
