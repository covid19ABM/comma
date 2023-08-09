"""Individual agent class definition
"""

from comma.hypothesis import PARAMS_INDIVIDUAL, \
    PARAMS_IPF_WEIGHTS, Hypothesis
import json
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


class Individual:

    def __init__(self, id: int, dir_params: str, features):
        self.id: int = id
        self.dir_params = dir_params
        self.chosen_actions = None
        self._status: float = .0
        self._features = features
        self.actions = Hypothesis.all_possible_actions

    def get_features(self):
        """
        Get agent's features

        Returns:
            pd.Series: represents an individual (agent)
            with their various features
        """
        return self._features

    def get_status(self):
        """
        Get the current agent status (i.e., mental health)

        Returns:
            pd.Series: the current status of the agent
        """
        return self._status

    def get_actions(self):
        """
        Get the current actions chosen by the agent

        Returns:
            actions (list): list of actions taken
        """
        return [action_name for action_name, action_was_taken in
                zip(self.actions, self.chosen_actions) if action_was_taken]

    def choose_actions_on_lockdown(self, lockdown: pd.DataFrame):
        """Choose the actions to take based on current lockdown policy.

        Args:
            lockdown (pd.DataFrame): dataframe of a given lockdown

        Returns:
            actions (pd.Series): list of booleans of taking/not-taking actions
            actions_probs (pd.Series): probability of taking that action
        """
        params_lockdown = lockdown
        n_actions, _ = params_lockdown.shape
        action_probs = params_lockdown.dot(self.get_features())
        # apply the sigmoid function
        action_probs = np.asarray(
            action_probs.apply(lambda x: 1 / (1 + np.exp(-x)))
        )
        actions = np.random.rand(n_actions) <= action_probs
        self.chosen_actions = actions  # store the chosen action

        return actions, action_probs

    def take_actions(self, actions: pd.Series, action_effects: pd.DataFrame):
        """Update status by taking the given actions.

        Args:
            actions (pd.Series): list of booleans of chosen/not-chosen actions.
        """
        params_status = action_effects
        result = params_status.dot(self.get_features()).dot(actions)

        self._status = result

    @staticmethod
    def sampling_from_ipf(size: int, dir_params: str):
        """
        Sample from IPF distribution saved
        as `weights.csv` in the parameters folder

        Parameters
        ----------
        size (int): size of data sample
        dir_params (str): path to the parameters folder

        Returns
        -------
        sample (pandas.dataFrame): dataframe containing the sampling
        """
        fpath_weights = os.path.join(dir_params, PARAMS_IPF_WEIGHTS)
        assert os.path.isfile(fpath_weights)

        df_weights = pd.read_csv(fpath_weights, sep=",", index_col=0)
        weights = df_weights["weight"] / df_weights["weight"].sum()
        indices = df_weights.index
        sample_indices = np.random.choice(indices, size, p=weights)
        sample = df_weights.loc[sample_indices].drop(["weight"], axis=1)
        sample = sample.reset_index(drop=True)
        return sample

    @staticmethod
    def populate_ipf(size: int, dir_params: str):
        """
        Create a population of individual agents
        with the given weights obtained via IPF

        Args:
            size (int): size of data sample.
            dir_params (str): path to parameters folder.
        """
        _features = pd.DataFrame()

        sample = Individual.sampling_from_ipf(size, dir_params)

        # one-hot encoding
        encoded_columns = pd.get_dummies(sample).reindex(
            columns=Hypothesis.all_possible_features,
            fill_value=0
        )
        _features = pd.concat([_features, encoded_columns], axis=1)

        # Add 'baseline' column filled with ones if this is not present yet
        if 'baseline' not in _features.columns:
            _features.insert(0, "baseline", 1)

        return [Individual(i, dir_params, _features.iloc[i]) for i in
                tqdm(range(size), desc="Populating individuals", unit="i")]

    @staticmethod
    def populate(size: int, dir_params: str):
        """
        Create a population of individual agents
        with the given feature parameters.

        Args:
            size (int): population size, i.e., number of agents.
            dir_params (str): dir to the folder containing
            feature parameter file.
            #from_scratch (bool, optional): flag of creating hypothesis
            from scratch or reading from files. Defaults to False.

        Returns:
            list[Individual]: a list of Individual agents
        """
        assert size > 0, 'Size must be positive!'
        assert isinstance(size, int), 'Size must be integer!'
        assert os.path.isdir(dir_params), "Given folder doesn't exist!"

        fpath_params_individual = os.path.join(dir_params, PARAMS_INDIVIDUAL)
        with open(fpath_params_individual) as f:
            features = json.load(f)

        _features = pd.DataFrame()
        for feature, distribution in features.items():
            _features[feature] = np.random.choice(
                distribution[0], size, p=distribution[1]
            )

            # Define all possible columns (including those not in the sample)
            # When the sample size is too small,
            # this doesn't cover all categories,
            # the resulting DataFrame thus lacks those columns.
            # To solve the issue, we ensure all possible categories are present
            # when creating the dummy variables

        # one-hot encoding
        categorical_cols = _features.select_dtypes(include=['object'])
        encoded_cols = pd.get_dummies(categorical_cols).reindex(
            columns=Hypothesis.all_possible_features,
            fill_value=0
        )
        _features.drop(categorical_cols.columns, axis=1, inplace=True)
        _features = pd.concat([_features, encoded_cols], axis=1)

        # Add 'baseline' column filled with ones
        _features.insert(0, "baseline", 1)

        return [Individual(i, dir_params, _features.iloc[i]) for i in
                tqdm(range(size), desc="Populating individuals", unit="i")]
