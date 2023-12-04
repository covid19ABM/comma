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
                "Version": [8, 8],
                "Date_of_report": ["2022-01-01", "2022-01-02"],
                "Date_of_publication": ["2022-01-01", "2022-01-02"],
                "Municipality_code": ["GM0289", "GM0289"],
                "Municipality_name": ["Wageningen", "Wageningen"],
                "Total_reported": [6001, 6031],
            }
        )

    def test_range_error(self):
        # Test that values outside the bounds raise an error
        start = "2025-02-01"
        steps = 1860
        municipality_code = "GM0289"
        hypothesis_instance = Hypothesis(start, steps)
        with pytest.raises(
            ValueError, match=r"time_period .* is outside available dates"
        ):
            hypothesis_instance.get_covid_data(municipality_code)

    @patch("comma.hypothesis.Hypothesis.download_covid_data")
    @patch("comma.hypothesis.os.path.exists")
    @patch("comma.hypothesis.pd.read_csv")
    def test_get_covid_data(self, mock_read_csv, mock_exists, mock_download, mock_df):
        # Test that the whole data extraction works
        mock_exists.return_value = False
        municipality_code = "GM0289"
        start = "2022-01-01"
        steps = 1

        hypothesis_instance = Hypothesis(start, steps)

        # Mock behavior of external dependencies
        mock_exists.return_value = False
        mock_download.return_value = mock_df
        mock_read_csv.return_value = mock_df

        out = hypothesis_instance.get_covid_data(municipality_code)

        assert mock_download.called
        assert not out.empty
        assert out.iloc[0]["Version"] == 8
        assert out.iloc[0]["Date_of_publication"] == "2022-01-01"
        assert out.iloc[0]["Municipality_code"] == "GM0289"
        assert out.iloc[0]["Municipality_name"] == "Wageningen"
        assert out.iloc[0]["Total_reported"] == 6001

    def test_location_not_in_dataset(self):
        # trying to get data for an unknown location
        # should raise an error
        with pytest.raises(ValueError):
            start = "2023-01-01"
            steps = 2
            hypothesis_instance = Hypothesis(start, steps)
            hypothesis_instance.get_positive_cases("XL4521")

    def test_scaling(self):
        # Tests the scale_cases_to_population function
        real_size = 1000
        sim_size = 50
        cumulative_cases = pd.Series([100, 100, 130, 140, 200])
        out = Hypothesis.scale_cases_to_population(
            cumulative_cases, real_size, sim_size
        )
        expected = pd.Series([5, 0, 1, 0, 3])
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
