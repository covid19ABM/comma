"""Individual agent class definition
"""

from comma.hypothesis import PARAMS_INDIVIDUAL, PARAMS_IPF_WEIGHTS, Hypothesis
import json
import numpy as np
import os
import pandas as pd
from scipy.stats import gamma  # for the "recovery" curve
from tqdm import tqdm


class Individual:
    def __init__(self, id: int, dir_params: str, features):
        self.id: int = id
        self.dir_params = dir_params
        self.chosen_actions = None
        self._status: float = 0.0
        self._features = features
        self.actions = Hypothesis.all_possible_actions
        self.covid_status: int = 0  # this tracks the positivity to COVID-19
        self.long_covid: int = 0  # is this a long covid case?
        self.days_since_positive = np.nan  # n-day from first day of positivity
        self.recovery = np.nan  # recovery status

    def get_features(self) -> pd.Series:
        """
        Get agent's features

        Returns:
            pd.Series: represents an individual (agent)
            with their various features
        """
        return self._features

    def get_status(self) -> float:
        """
        Get the current agent status (i.e., mental health)

        Returns:
            float: the current status of the agent
        """
        return self._status

    def get_covid_status(self) -> float:
        """
        Get the current agent covid status

        Returns:
            pd.Series: the current covid status of the agent
        """
        return self.covid_status

    def get_actions(self) -> list[str]:
        """
        Get the current actions chosen by the agent

        Returns:
            actions (list): list of actions taken
        """
        return [
            action_name
            for action_name, action_was_taken in zip(self.actions, self.chosen_actions)
            if action_was_taken
        ]

    def choose_actions_on_lockdown(
        self, lockdown: pd.DataFrame, rng=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Choose the actions to take based on current lockdown policy.

        Args:
            lockdown (pd.DataFrame): dataframe of a given lockdown
            rng (np.random.Generator): optional. An instance of numpy random
            generator. If not provided, a default random generator will be
            used. This ensures reproducibility.

        Returns:
            actions (np.ndarray): array of booleans
            actions_probs (np.ndarray): array of probability
        """
        params_lockdown = lockdown
        n_actions, _ = params_lockdown.shape
        action_probs = params_lockdown.dot(self.get_features())
        # apply the sigmoid function
        action_probs = np.asarray(action_probs.apply(lambda x: 1 / (1 + np.exp(-x))))
        # use the new random generator method of numpy
        if rng is None:
            rng = np.random.default_rng(None)
        actions = rng.random(n_actions) <= action_probs
        self.chosen_actions = actions  # store the chosen action

    # return actions, action_probs

    def is_long_covid(self):
        """
        Determine if this is a Long covid case.

        This method decides whether an agent is
        considered to be a long COVID case, based on
        a set of probability (currently set to 20% chance)

        Returns:
            bool: True if long COVID, False otherwise

        """
        rng = np.random.default_rng(None)

        if self.long_covid == 0 and rng.random() < 0.20:
            self.long_covid = 1

        return self.long_covid == 1

    def is_recovered(self, rng=None):
        """
        Determine if an individual is recovered based on
        the number of days since testing positive.

        The function uses a gamma cumulative distribution
        function with shape parameter 5 and scale parameter 3
        to model the probability of recovery. This can be of
        course changed based on the literature.
        Note that if `n_days` is equal or lower than 10,
        then the probability of recovering is always 0,
        as per Astrid's suggestion.

        Args:
            n_days (int): Number of days since tested positive.
            rng (np.random.Generator): optional. An instance of numpy random
            generator. If not provided, a default random generator will be
            used. This ensures reproducibility.
        Returns:
            recovery (bool): True if recovered, False otherwise.
        """

        if self.days_since_positive <= 10:
            return 0

        if rng is None:
            rng = np.random.default_rng(None)

        if self.is_long_covid():
            # long covid recovery
            recovery_prob = gamma.cdf(self.days_since_positive, a=7, scale=10)
        else:
            # standard recovery
            recovery_prob = gamma.cdf(self.days_since_positive, a=5, scale=3)

        recovery = rng.uniform() <= recovery_prob
        return recovery

    @staticmethod
    def modify_policy_when_infected(lockdown: pd.DataFrame):
        """

        Args:
            lockdown (pd.DataFrame): lockdown matrix

        Returns:
            actions (np.ndarray): array of booleans
            actions_probs (np.ndarray): array of probability

        """
        lockdown_ = lockdown.copy()
        # Set betas to 0 in all columns except actions and baseline
        columns_to_update = lockdown_.columns.difference(["actions", "baseline"])
        lockdown_[columns_to_update] = 0

        # set baseline beta to -5 (v. unlikely)
        lockdown_["baseline"] = -5
        # however for 'be sedentary' set beta to 5 (v. likely)
        if "actions" in lockdown_.columns:
            lockdown_.loc[lockdown_["actions"] == "be_sedentary", "baseline"] = 5
        else:
            lockdown_.iat[8, lockdown_.columns.get_loc("baseline")] = 5

        return lockdown_

    def take_actions(self, action_effects: pd.DataFrame) -> None:
        """
        Update status by taking the given actions.

        Args:
            actions (pd.Series): list of booleans of chosen/not-chosen actions.
            action_effects (pd.DataFrame): matrix of actions effects

        Returns:
            None: This function updates the agent's status
            but does not return anything.
        """
        params_status = action_effects
        result = params_status.dot(self.get_features()).dot(self.chosen_actions)

        self._status = result

    @staticmethod
    def sampling_from_ipf(size: int, dir_params: str, rng=None) -> pd.DataFrame:
        """
        Sample from IPF distribution saved
        as `weights.csv` in the parameters folder

        Parameters
        ----------
        size (int): size of data sample
        dir_params (str): path to the parameters folder
        rng (np.random.Generator): optional. An instance of numpy random
            generator. If not provided, a default random generator will be
            used. This ensures reproducibility.

        Returns
        -------
        sample (pandas.dataFrame): dataframe containing the sampling
        """
        fpath_weights = os.path.join(dir_params, PARAMS_IPF_WEIGHTS)
        assert os.path.isfile(fpath_weights)

        df_weights = pd.read_csv(fpath_weights, sep=",", index_col=0)
        weights = df_weights["weight"] / df_weights["weight"].sum()
        indices = df_weights.index
        # use the new random method of numpy
        if rng is None:
            rng = np.random.default_rng(None)
        sample_indices = rng.choice(indices, size, p=weights)
        sample = df_weights.loc[sample_indices].drop(["weight"], axis=1)
        sample = sample.reset_index(drop=True)
        return sample

    @staticmethod
    def populate_ipf(size: int, dir_params: str, rng=None) -> list:
        """
        Create a population of individual agents
        with the given weights obtained via IPF

        Args:
            size (int): size of data sample.
            dir_params (str): path to parameters folder.
            rng (np.random.Generator): optional. An instance of numpy random
            generator. If not provided, a default random generator will be
            used. This ensures reproducibility.

        Returns:
            List[Individual]: A list containing instances of
            the individual class, each representing an
            agent with specific features.
        """
        _features = pd.DataFrame()

        sample = Individual.sampling_from_ipf(size, dir_params, rng)

        # one-hot encoding
        encoded_columns = pd.get_dummies(sample)
        encoded_columns.columns = map(str.lower, encoded_columns.columns)
        encoded_columns = encoded_columns.reindex(
            columns=Hypothesis.all_possible_features, fill_value=0
        )
        _features = pd.concat([_features, encoded_columns], axis=1)

        # Add 'baseline' column filled with ones if this is not present yet
        if "baseline" not in _features.columns:
            _features.insert(0, "baseline", 1)

        return [
            Individual(i, dir_params, _features.iloc[i])
            for i in tqdm(range(size), desc="Populating individuals", unit="i")
        ]

    @staticmethod
    def populate(size: int, dir_params: str, rng=None) -> list:
        """
        Create a population of individual agents
        with the given feature parameters.

        Args:
            size (int): population size, i.e., number of agents.
            dir_params (str): dir to the folder containing
            feature parameter file.
            #from_scratch (bool, optional): flag of creating hypothesis
            from scratch or reading from files. Defaults to False.
            rng (np.random.Generator): optional. An instance of numpy random
            generator. If not provided, a default random generator will be
            used. This ensures reproducibility.

        Returns:
            list[Individual]: a list of Individual agents
        """
        assert size > 0, "Size must be positive!"
        assert isinstance(size, int), "Size must be integer!"
        assert os.path.isdir(dir_params), "Given folder doesn't exist!"

        fpath_params_individual = os.path.join(dir_params, PARAMS_INDIVIDUAL)
        with open(fpath_params_individual) as f:
            features = json.load(f)

        _features = pd.DataFrame()
        for feature, distribution in features.items():
            # new numpy random method
            if rng is None:
                # quality gate complains about security if
                # I don't use a large range
                rng = np.random.default_rng(None)
            _features[feature] = rng.choice(distribution[0], size, p=distribution[1])

            # Define all possible columns (including those not in the sample)
            # When the sample size is too small,
            # this doesn't cover all categories,
            # the resulting DataFrame thus lacks those columns.
            # To solve the issue, we ensure all possible categories are present
            # when creating the dummy variables

        # one-hot encoding
        categorical_cols = _features.select_dtypes(include=["object"])
        encoded_cols = pd.get_dummies(categorical_cols).reindex(
            columns=Hypothesis.all_possible_features, fill_value=0
        )
        _features = _features.drop(categorical_cols.columns, axis=1)
        _features = pd.concat([_features, encoded_cols], axis=1)

        # Add 'baseline' column filled with ones
        _features.insert(0, "baseline", 1)

        return [
            Individual(i, dir_params, _features.iloc[i])
            for i in tqdm(range(size), desc="Populating individuals", unit="i")
        ]
