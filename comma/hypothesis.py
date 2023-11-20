"""Hypothesis class definition"""
import warnings
from datetime import datetime, timedelta
import json
import os
import pandas as pd
import re
import requests
from tqdm import tqdm

PARAMS_INDIVIDUAL = "params_individual.json"
PARAMS_IPF_WEIGHTS = "ipf_weights.csv"
date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")


class Hypothesis:
    """
    The Hypothesis class is responsible for managing and validating
    hypotheses specified by the user.

    Methods:
        _get_one_hot_encoded_features():
            One-hot encodes categorical features
        create_empty_hypotheses():
            Creates empty CSV files for storing hypotheses
        validate_param_file():
            Validates the files in the parameter folder

    Usage:
        Hypothesis.create_empty_hypotheses("/path/to/dir_params")
        Hypothesis.validate_param_file("/path/to/dir_params")
    """

    RIVM_URL = "https://github.com/mzelst/covid-19/raw/master/data-rivm/tests/"

    _required_params = [
        "size",
        "steps",
        "actions",
        "status",
        "lockdown_policies",
        "lockdown",
    ]

    lockdown_policies = ["absent", "easy", "medium", "hard"]

    individual_status = ["mh"]

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
        "unemployed_no",
        "unemployed_yes",
        "have_partner_no",
        "have_partner_yes",
        "depressed_no",
        "depressed_yes",
        "children_presence_no",
        "children_presence_yes",
        "housing_financial_difficulties_no",
        "housing_financial_difficulties_yes",
        "selfrated_health_average",
        "selfrated_health_good",
        "selfrated_health_poor",
        "critical_job_no",
        "critical_job_yes",
    ]

    all_possible_actions = [
        "work_from_home",
        "maintain_physical_distance",
        "stay_at_home",
        "exercise",
        "socialise",
        "travel",
        "seek_help",
        "negative_coping",
        "positive_coping",
        "socialise_online",
    ]

    def __init__(self, start: str, steps: int):
        self.start = start
        self.steps = steps
        self.date_format = "%Y-%m-%d"
        self.time_period: tuple[str, str]
        self.compute_time_period()

    @staticmethod
    def get_file_paths(url: str) -> list:
        """
        Extract file paths from url

        Args:
        url (str): website

        Returns:
        file_paths (list): list of '.csv.gz' file paths
        """

        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # extract '.csv.gz' file paths
        file_paths = [
            item["path"]
            for item in data["payload"]["tree"]["items"]
            if item["path"].endswith(".csv.gz")
        ]

        return file_paths

    @staticmethod
    def filter_dates(file_list: list, time_period: tuple[str, str]) -> list[str]:
        """
        Select dates within the interval defined by `time_period`

        Args:

            file_list (List): list of file paths
            time_period (Tuple): time interval of the time t0 -> t1

        Returns:
            filtered_paths (List): list of filtered paths

        Raises:
            ValueError: if the time_period is not within the
            range of dates in the file
        """
        start = datetime.strptime(time_period[0], "%Y-%m-%d")
        end = datetime.strptime(time_period[1], "%Y-%m-%d")

        all_dates: list[datetime] = []

        for file in file_list:
            match = date_pattern.search(file)
            if match:
                date = datetime.strptime(match.group(1), "%Y-%m-%d")
                all_dates.append(date)

        if not all_dates:
            raise ValueError("Dates provided are not within the list")

        min_date = min(all_dates)
        max_date = max(all_dates)
        # TO-DO: return list of non available dates
        if start < min_date or end > max_date:
            raise ValueError(
                f"time_period ({time_period[0]} - {time_period[1]}) "
                f"is outside available dates that go from "
                f"({min_date} to {max_date})"
            )

        # Filtering the dates now that we know they're within the range
        filtered_paths = [
            file
            for file in file_list
            if start
            <= datetime.strptime(date_pattern.search(file).group(1), "%Y-%m-%d")
            <= end
        ]

        return filtered_paths

    def compute_time_period(self) -> tuple:
        """
        Compute time period based on a starting date and number of steps

        Args:
            start (str): Start date
            steps (int): Number of steps
            date_format (str): Format of the date string

        Returns:
            tuple: A tuple containing the start and end date
        """
        start_date = datetime.strptime(self.start, self.date_format)
        end_date = start_date + timedelta(days=self.steps)
        self.time_period = (
            start_date.strftime(self.date_format),
            end_date.strftime(self.date_format),
        )

    @classmethod
    def get_covid_data(
        cls, time_period: tuple[str, str], location: str
    ) -> pd.DataFrame:
        """
        Download and filter COVID-19 test data from the RIVM website.

        Args:
        time_period (tuple): Start and end date ('YYYY-MM-DD', 'YYYY-MM-DD').
        location (str): Security region name. This is the name of the city.

        Returns:
        df_filtered (pandas.DataFrame): Filtered data.
        """

        dates = cls.get_file_paths(cls.RIVM_URL)
        filtered_dates = cls.filter_dates(dates, time_period)

        df_gzip = []
        message = "Downloading COVID-19 data from RIVM"
        for date in tqdm(filtered_dates, desc=message):
            full_url = cls.RIVM_URL + date.split("/")[-1]
            df = pd.read_csv(
                full_url, compression="gzip", header=0, sep=",", quotechar='"'
            )
            if not isinstance(df, pd.DataFrame):
                raise ValueError(
                    f"Data retrieved from {cls.RIVM_URL}"
                    f" is not a DataFrame but a {type(df)}"
                )
            df_gzip.append(df)

        df_tests = pd.concat(df_gzip, ignore_index=True)

        df_tests["Date_of_statistics"] = pd.to_datetime(df_tests["Date_of_statistics"])

        mask = (
            (df_tests["Date_of_statistics"] >= time_period[0])
            & (df_tests["Date_of_statistics"] <= time_period[1])
            & (df_tests["Security_region_name"] == location)
        )
        df_filtered = df_tests.loc[mask].reset_index(drop=True)

        return df_filtered

    def get_positive_cases(self, location: str) -> pd.Series:
        """
        Get an array of daily positive COVID-19 cases for
        a specific time period and location.

        Args:
        time_period (tuple): Start and end date ('YYYY-MM-DD', 'YYYY-MM-DD').
        location (str): Security region name.

        Returns:
        daily_positive_cases (pandas.Series): Daily positive cases.

        """
        df_filtered = self.get_covid_data(self.time_period, location)
        # Check if the filtered dataframe is empty after filtering by location
        if df_filtered.empty:
            raise ValueError(f"No data available for location: {location}")

        # positive cases were reported multiple times
        # and multiple days after a particular day
        # as there were corrections. We select the most reliable records:
        # sort dataframe based on date of report
        sorted_df = df_filtered.sort_values(by="Date_of_report", ascending=False)
        # get the most recent data
        agg_df = sorted_df.groupby("Date_of_statistics").first().reset_index()
        daily_positive_cases = agg_df["Tested_positive"]

        # match the length of the positives by day with the n of steps
        if len(daily_positive_cases) < self.steps:
            # if there are fewer data than steps
            daily_positive_cases = self.adjust_cases(self.steps, daily_positive_cases)
        else:
            # otherwise match the n of steps
            # this prevents problems when there is more data than steps
            daily_positive_cases = daily_positive_cases[: self.steps]
        return daily_positive_cases

    @staticmethod
    def adjust_cases(steps: int, daily_positive_cases: pd.Series) -> pd.Series:
        """
        Ensures the length of daily_positive_cases matches the given steps.

        If the length of daily_positive_cases is less than the provided steps,
        (this might happen if there are missing data for some days in RIVM)
        this function will repeat the last row of daily_positive_cases until
        its length is equal to steps.

        Args:
            steps (int): Desired length for daily_positive_cases
            daily_positive_cases (pd.Series): Series of positive cases per day

        Returns:
            pd.Series: Modified series with length equal to steps

        """
        # return the average between the two most useful datapoints
        n_times = steps - len(daily_positive_cases)

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
            real_size(int): Size of the original population location
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

        new_cases = (daily_positive_cases * sim_size) / real_size
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

            df = pd.read_csv(fpath_params, delimiter=";", decimal=".")

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

    @classmethod
    def create_empty_hypotheses(cls, dir_params: str) -> None:
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

        actions = cls.all_possible_actions
        lockdown_policies = cls.lockdown_policies
        status = cls.individual_status
        columns = ["actions", "baseline"]
        columns += cls._get_one_hot_encoded_features(fpath_params_individual)
        df = pd.DataFrame(0, index=range(len(actions)), columns=columns)
        df["actions"] = actions

        output_fpaths = ["lockdown_%s.csv" % lockdown for lockdown in lockdown_policies]
        output_fpaths += ["actions_effects_on_%s.csv" % s for s in status]
        output_fpaths = [os.path.join(dir_params, fp) for fp in output_fpaths]
        for fp in output_fpaths:
            df.to_csv(fp, sep=";", index=False)

    @classmethod
    def validate_param_file(cls, dir_params: str) -> None:
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
            for status in Hypothesis.individual_status
            for policy in Hypothesis.lockdown_policies
        ]
        fnames += ["lockdown_%s.csv" % lockdown for lockdown in cls.lockdown_policies]
        fpaths = [os.path.join(dir_params, fn) for fn in fnames]
        fexist = [os.path.isfile(fp) for fp in fpaths]
        if not all(fexist):
            raise ValueError(
                "Hypothesis file(s) not found: %s."
                % ", ".join([fnames[i] for i in range(len(fnames)) if not fexist[i]])
            )

        # check if all hypothesis files contain all the required agent features
        required_features = ["actions", "baseline"]
        required_features += cls._get_one_hot_encoded_features(path_individual)
        hypothesis_data = [pd.read_csv(fp, sep=";", decimal=",") for fp in fpaths]
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
        required_actions = cls.all_possible_actions
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
