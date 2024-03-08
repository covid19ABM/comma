"""Hypothesis class definition"""
import warnings
from datetime import datetime, timedelta
import json
import os
import pandas as pd
import re
import requests

PARAMS_INDIVIDUAL = "params_individual.json"
PARAMS_IPF_WEIGHTS = "ipf_weights.csv.zip"
date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")


class Hypothesis:
    """
    The Hypothesis class is responsible for managing and validating
    hypotheses specified by the user.
    """

    def __init__(self, start: str, steps: int):
        self.start = start
        self.steps = steps
        self.date_format = "%Y-%m-%d"
        self.time_period: tuple[str, str]
        self.compute_time_period()
        self.RIVM_URL = (
            "https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_cumulatief.json"
        )
        self.lockdown_policies = ["easy", "hard"]
        self.individual_status = ["mh"]
        self._required_params = [
            "size",
            "steps",
            "actions",
            "status",
            "lockdown_policies",
            "lockdown",
        ]

    # TO-DO: these two go into a config file
    all_possible_features = [
        "age_group__1",
        "age_group__2",
        "age_group__3",
        "age_group__4",
        "gender_f",
        "gender_m",
        "education_high",
        "education_low",
        "education_medium",
        "education_unknown",
        "unemployed_no",
        "unemployed_yes",
        "have_partner_no",
        "have_partner_yes",
        "have_partner_unknown",
        "depressed_no",
        "depressed_yes",
        "depressed_unknown",
        "children_presence_no",
        "children_presence_yes",
        "children_presence_unknown",
        "housing_financial_difficulties_no",
        "housing_financial_difficulties_yes",
        "housing_financial_difficulties_unknown",
        "selfrated_health_average",
        "selfrated_health_good",
        "selfrated_health_poor",
        "selfrated_health_unknown",
        "critical_job_no",
        "critical_job_yes",
        "critical_job_unknown",
        "bmi_underweight",
        "bmi_normalweight",
        "bmi_overweight",
        "bmi_obese",
        "bmi_unknown",
        "livesalone_no",
        "livesalone_yes",
        "livesalone_unknown",
        "income_median_above",
        "income_median_below",
        "income_median_unknown",
    ]

    all_possible_actions = [
        "work_from_home",
        "maintain_social_distance",
        "exercise",
        "feel_socially_connected",
        "seek_help_from_friends",
        "heavy_drinking",
        "positive_coping",
        "feel_isolated",
        "be_sedentary",
    ]

    def download_covid_data(self) -> pd.DataFrame:
        """
        Download COVID-19 data from RIVM

        Returns:
        - df (pd.Dataframe): A DataFrame containing the downloaded data.
        """

        response = requests.get(self.RIVM_URL)
        if response.status_code != 200:
            raise Exception(
                "Failed to download data: HTTP status code:", response.status_code
            )
        df = pd.DataFrame(response.json())
        return df

    def compute_time_period(self) -> tuple:
        """
        Compute time period based on a starting date and number of steps

        Returns:
            tuple: A tuple containing the start and end date
        """
        start_date = datetime.strptime(self.start, self.date_format)
        end_date = start_date + timedelta(days=self.steps)
        self.time_period = (
            start_date.strftime(self.date_format),
            end_date.strftime(self.date_format),
        )

    def get_covid_data(self, municipality_code: str, cache=False) -> pd.DataFrame:
        """
        Download and filter COVID-19 test data from the RIVM website.

        Args:
        municipality_code (str): also known as Gemeentecode
        cache (bool): If True, saves the downloaded data to
        a CSV file for future use.


        Returns:
        filtered_data (pd.DataFrame). The COVID-19 data filtered
        """
        folder_path = "data"
        start = self.time_period[0]
        end = self.time_period[1]

        csv_file_path = os.path.join(
            folder_path, f"COVID19_{municipality_code}_data_{start}_{end}.csv"
        )

        # check that the file already exists
        if os.path.exists(csv_file_path):
            print(f"Data already exists: {csv_file_path}.")
            return pd.read_csv(csv_file_path)

        # fetch the data
        print("Downloading COVID-19 data from RIVM")
        data = self.download_covid_data()
        print("Data fetched")

        # filter by municipality code
        filtered_data = data[data["Municipality_code"] == municipality_code].copy()

        # are dates available compatible with start/end dates?
        available_start = pd.to_datetime(data["Date_of_report"].min())
        available_end = pd.to_datetime(data["Date_of_report"].max())

        if (
            pd.to_datetime(start) < available_start
            or pd.to_datetime(end) > available_end
        ):
            raise ValueError(
                f"time_period ({self.time_period[0]} - {self.time_period[1]}) "
                f"is outside available dates that go from "
                f"({available_start} to {available_end})"
            )

        # filter by dates
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        filtered_data["Date_of_report"] = pd.to_datetime(
            filtered_data["Date_of_report"]
        )

        filtered_data = filtered_data[
            (filtered_data["Date_of_report"] >= start_date)
            & (filtered_data["Date_of_report"] <= end_date)
        ]

        if cache:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            filtered_data.to_csv(csv_file_path, index=False)
            print("CSV file saved at:", csv_file_path)

        return filtered_data

    def get_positive_cases(self, municipality_code: str, cache=False) -> pd.Series:
        """
        Get an array of daily positive COVID-19 cases for
        a specific time period and municipality_code.

        Args:
        municipality_code (str): also known as Gementecode
        cache(boolean): Do you want to save COVID-19 data
            i.e., to avoid to download twice?

        Returns:
        daily_positive_cases (pandas.Series): Daily positive cases.

        """
        df_filtered = self.get_covid_data(municipality_code, cache)

        # Check if the filtered dataframe is empty
        # after filtering by municipality_code
        if df_filtered.empty:
            raise ValueError(f"No data available for municipality: {municipality_code}")

        # positive cases were reported multiple times
        # and multiple days after a particular day
        # as there were corrections. We select the most reliable records:
        # sort dataframe based on date of report
        sorted_df = df_filtered.sort_values(by="Date_of_report", ascending=False)
        # get the most recent data
        agg_df = sorted_df.groupby("Date_of_publication").first().reset_index()
        daily_positive_cases = agg_df["Total_reported"]

        # match the length of the positives by day with the n of steps
        if len(daily_positive_cases) < self.steps:
            # if there are fewer data than steps
            daily_positive_cases = self.adjust_cases(daily_positive_cases)
        else:
            # otherwise match the n of steps
            # this prevents problems when there is more data than steps
            daily_positive_cases = daily_positive_cases[: self.steps]
        return daily_positive_cases

    def adjust_cases(self, daily_positive_cases: pd.Series) -> pd.Series:
        """
        Ensures the length of daily_positive_cases matches the given steps.

        If the length of daily_positive_cases is less than the provided steps,
        (this might happen if there are missing data for some days in RIVM)
        this function will repeat the last row of daily_positive_cases until
        its length is equal to steps.

        Args:
            daily_positive_cases (pd.Series): Series of positive cases per day

        Returns:
            pd.Series: Modified series with length equal to steps

        """
        # return the average between the two most useful datapoints
        n_times = self.steps - len(daily_positive_cases)

        if n_times > 0:
            n_values = pd.Series([daily_positive_cases.iloc[-1]] * n_times)

            positive_cases = pd.concat(
                [daily_positive_cases, n_values], ignore_index=True
            )
        return positive_cases

    @staticmethod
    def scale_cases_to_population(
        daily_positive_cases: pd.Series, real_size: int, sim_size: int
    ):
        """
        We compute the number of new positives in day _i_
        for our simulated population as `new_cases_sim` = (n/N)*M where
        N is the size of the simulated population, M is the size of the
        real population, n is the number of new positives reported
        on RIVM for that day.

        Args:
            daily_positive_cases(pd.Series): Daily positive cases
            real_size(int): Size of the original population
            sim_size(int): Size of the simulated population

        Returns:
            pd.Series: Daily positive cases based on simulated population size
        """
        # Warning/ FYI:
        # if sim_size is very small compared to real_size and
        # the values in daily_positive_cases are also small
        # (each value is much less than real_size), then the
        # function could always result in zeros for every
        # entry in daily_positive_cases.
        max_scaled_case = daily_positive_cases.max() * sim_size

        if max_scaled_case < real_size:
            message = (
                f"Given sim_size={sim_size}, "
                f"real_size={real_size}, "
                f"and max daily cases={daily_positive_cases.max()},"
                f"the scaling might result in all zeros."
            )
            warnings.warn(message)

        cumulative_cases = (daily_positive_cases * sim_size) / real_size
        # note that RIVM gives us the cumulative cases
        # so we need to compute the difference between day2 and day1
        # to get new cases
        new_cases = pd.Series(
            [cumulative_cases[0]]
            + [
                cumulative_cases[case + 1] - cumulative_cases[case]
                for case in range(len(cumulative_cases) - 1)
            ]
        )

        return new_cases.astype("int")

    @classmethod
    def read_hypotheses(
        cls, dir_params: str, policies: set[str], type: str
    ) -> dict[str, pd.DataFrame]:
        """
        Read in CSV matrices for either actions or lockdowns.

        Args:
            dir_params (str): path of the parameters folder
            policies (set): set object of either actions or lockdown list
            type (str): either 'actions' or 'lockdown'

        Returns:
            data_dfs (dict): A dictionary where the key is either an action
                             effect or lockdown policy, and the value is a
                             processed dataframe.
        """

        # Ensure valid data type
        if type not in ["actions", "lockdown"]:
            raise ValueError("type should be either 'actions' or 'lockdown'")

        file_patterns = {
            "actions": "actions_effects_on_mh_%s.csv",
            "lockdown": "lockdown_%s.csv",
        }

        data_dfs = {}

        for policy in policies:
            fpath_params = os.path.join(dir_params, file_patterns[type] % policy)

            df = pd.read_csv(fpath_params, delimiter=",", decimal=".")
            df.fillna(0, inplace=True)

            for col in df.columns:
                if col != "actions":
                    df[col] = df[col].astype(float)

            # sort rows
            df["actions"] = df["actions"].astype("category")
            df["actions"] = df["actions"].cat.set_categories(cls.all_possible_actions)
            df = df.sort_values(by="actions", ignore_index=True)

            # Convert dataframe column names and cols to lowercase
            df.columns = df.columns.str.lower()
            cols = [col.lower() for col in cls.all_possible_features]
            cols.insert(0, "baseline")

            # get and sort desired columns
            df = df[cols]

            data_dfs[policy] = df

        return data_dfs

    @staticmethod
    def _get_one_hot_encoded_features(fpath_params_individual: str) -> list:
        """
        One-hot encode categorical features in the
        `params_individual.json` file and return the
        feature list.

        Args:
            fpath_params_individual (str): Path to the
            individual parameters JSON file.

        Returns:
            features (list): List of one-hot encoded features.

        """
        with open(fpath_params_individual) as f:
            params_individual = json.load(f)

        features = []
        for key, value in params_individual.items():
            if isinstance(value[0][0], str):
                features += [key + "_" + v for v in value[0]]
            else:
                features += [key]
        return features

    def create_empty_hypotheses(self, dir_params: str) -> None:
        """
        Create empty CSV files for storing hypotheses on
        the impact of actions and lockdown policies on different agent statuses

        Args:
            dir_params (str): The directory of the folder that contains
            the agent and model parameter files.
        Returns:
            None: This function does not return anything
            as it creates empty csv files int the specified directory
        """
        fpath_params_individual = os.path.join(dir_params, PARAMS_INDIVIDUAL)

        # Check if the files exist
        if not os.path.exists(fpath_params_individual):
            raise FileNotFoundError(
                f"'{PARAMS_INDIVIDUAL}' \
            file is missing in the directory '{dir_params}'"
            )

        actions = self.all_possible_actions
        lockdown_policies = self.lockdown_policies
        status = self.individual_status
        columns = ["actions", "baseline"]
        columns += self._get_one_hot_encoded_features(fpath_params_individual)
        df = pd.DataFrame(0, index=range(len(actions)), columns=columns)
        df["actions"] = actions

        output_fpaths = ["lockdown_%s.csv" % lockdown for lockdown in lockdown_policies]
        output_fpaths += ["actions_effects_on_%s.csv" % s for s in status]
        output_fpaths = [os.path.join(dir_params, fp) for fp in output_fpaths]
        for fp in output_fpaths:
            df.to_csv(fp, sep=";", index=False)

    def validate_param_file(self, dir_params: str) -> None:
        """Validate files in the parameter folder.

        Args:
            dir_params (str): dir to the folder containing
            hypothesis and parameter files.

        Raises:
            ValueError: If any validation checks fail.
        """
        # check if parameter files exist
        path_individual = os.path.join(dir_params, PARAMS_INDIVIDUAL)

        # check if all hypothesis files exist
        fnames = [
            "actions_effects_on_%s_%s.csv" % (status, policy)
            for status in self.individual_status
            for policy in self.lockdown_policies
        ]
        fnames += ["lockdown_%s.csv" % lockdown for lockdown in self.lockdown_policies]
        fpaths = [os.path.join(dir_params, fn) for fn in fnames]
        fexist = [os.path.isfile(fp) for fp in fpaths]
        if not all(fexist):
            raise ValueError(
                "Hypothesis file(s) not found: %s."
                % ", ".join([fnames[i] for i in range(len(fnames)) if not fexist[i]])
            )

        # check if all hypothesis files contain all the required agent features
        required_features = ["actions", "baseline"]
        required_features += self._get_one_hot_encoded_features(path_individual)
        hypothesis_data = [pd.read_csv(fp, sep=",", decimal=".") for fp in fpaths]
        missing_features = []
        for hd in hypothesis_data:
            # lower case labels
            missing_features.append(
                set([f.lower() for f in required_features])
                - set([c.lower() for c in hd.columns])
            )

        if any(missing_features):
            raise ValueError(
                "Missing features:\n%s"
                % "\n".join(
                    [
                        "%s - %s" % (fnames[i], ", ".join(missing_features[i]))
                        for i in range(len(fnames))
                        if missing_features[i]
                    ]
                )
            )

        # check if all hypothesis files contain hypotheses of all actions
        required_actions = self.all_possible_actions
        missing_actions = [
            set(required_actions) - set(hd["actions"]) for hd in hypothesis_data
        ]
        if any(missing_actions):
            raise ValueError(
                "Missing actions:\n%s"
                % "\n".join(
                    [
                        "%s - %s" % (fnames[i], ", ".join(missing_actions[i]))
                        for i in range(len(fnames))
                        if missing_actions[i]
                    ]
                )
            )
