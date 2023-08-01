"""Individual agent class definition
"""

import pandas as pd
import numpy as np
import os
from . import read_json_as_dict
from . import PARAMS_INDIVIDUAL, PARAMS_MODEL, PARAMS_IPF_WEIGHTS
from tqdm import tqdm

class Individual:
    _features = pd.DataFrame()
    _status = pd.DataFrame()
    
    def __init__(self, id: int, dir_params: str):
        self.id: int = id
        self.dir_params = dir_params
        
        fpath_param_model = os.path.join(dir_params, PARAMS_MODEL)
        self.actions = read_json_as_dict(fpath_param_model)["actions"]
        
    def get_features(self):
        return self._features.loc[self.id]
    
    def get_status(self):
        return self._status.loc[self.id]
        
    def _read_hypothesis(self, fpath: str):
        """Read a hypothesis file and return the parameter matrix.
        
        This function will also make sure that the resulting parameter
        matrix will have exactly the same amount of actions and features
        and have them ordered as desired.

        Args:
            fpath (str): hypothesis file path

        Returns:
            pd.Dataframe: hypothesis parameter matrix
        """
        assert os.path.isfile(fpath), 'File not found: %s.' % fpath
        cols = self.get_features().index
        df = pd.read_csv(fpath, delimiter=';', decimal=",")
        
        # sort rows
        df['actions'] = df['actions'].astype('category')
        df['actions'] = df['actions'].cat.set_categories(self.actions)
        df = df.sort_values(by='actions', ignore_index=True)

        # Convert dataframe column names to lowercase
        df.columns = df.columns.str.lower()

        # Convert cols to lowercase
        cols = [col.lower() for col in cols]

        # get and sort desired columns
        df = df[cols]
        
        return df
        
    def choose_actions_on_lockdown(self, lockdown: str):
        """Choose the actions to take based on current lockdown policy.

        Args:
            lockdown (str): one of the given lockdowns

        Returns:
            actions (pd.Series): list of booleans of taking/not-taking actions
            actions_probs (pd.Series): probability of taking that action
        """
        fpath_params_lockdown = os.path.join(
            self.dir_params, 'lockdown_%s.csv' % lockdown)
        params_lockdown = self._read_hypothesis(fpath_params_lockdown)
        n_actions, _ = params_lockdown.shape
        action_probs = params_lockdown.dot(self.get_features())
        # apply the sigmoid function
        action_probs = np.asarray(action_probs.apply(lambda x: 1 / (1 + np.exp(-x))))
        actions = np.random.rand(n_actions) <= action_probs
        return actions, action_probs
    
    def take_actions(self, actions: pd.Series):
        """Update status by taking the given actions.

        Args:
            actions (pd.Series): list of booleans of taking/not-taking actions.
        """
        status = self.get_status().index
        results = []
        for s in status:
            fpath_params_status = os.path.join(
                self.dir_params, 'actions_effects_on_%s.csv' % s)
            params_status = self._read_hypothesis(fpath_params_status)
            result = params_status.dot(self.get_features()).dot(actions)
            results.append(result)
        self._status.loc[self.id] = results

    @staticmethod
    def populate_ipf(size: int, dir_params: str):
        """
        Create a population of individual agents with the given weights obtained via IPF

        Args:
            size (int): size of data sample.
            dir_params (str): path to paramters folder.
        """
        fpath_weights = os.path.join(dir_params, PARAMS_IPF_WEIGHTS)
        assert os.path.isfile(fpath_weights)

        df_weights = pd.read_csv(fpath_weights, sep=",", index_col=0)
        weights = df_weights["weight"] / df_weights["weight"].sum()
        indices = df_weights.index
        sample_indices = np.random.choice(indices, size, p=weights)
        sample = df_weights.loc[sample_indices].drop(["weight"], axis=1)
        sample = sample.reset_index(drop=True)

        all_possible_cols = [
            'age_group__1',
            'age_group__2',
            'age_group__3',
            'age_group__4',
            'gender_f',
            'gender_m',
            'education_high',
            'education_low',
            'education_medium',
            'unemployed_no',
            'unemployed_yes',
            'have_partner_no',
            'have_partner_yes',
            'depressed_no',
            'depressed_yes',
            'children_presence_no',
            'children_presence_yes',
            'housing_financial_difficulties_no',
            'housing_financial_difficulties_yes',
            'selfrated_health_average',
            'selfrated_health_good',
            'selfrated_health_poor',
            'critical_job_no',
            'critical_job_yes'
        ]

        # hot-one encoding
        encoded_columns = pd.get_dummies(sample).reindex(columns=all_possible_cols, fill_value=0)
        encoded_columns.insert(0, "baseline", 1)

        return encoded_columns

    @staticmethod
    def populate(size: int, dir_params: str): 
        """Create a population of individual agents with the given feature parameters.
        
        Args:
            size (int): population size, i.e., number of agents.
            dir_params (str): dir to the folder containing feature parameter file.
            #from_scratch (bool, optional): flag of creating hypothesis from scratch or reading from files. Defaults to False.

        Returns:
            list[Individual]: a list of Individual agents
        """
        assert size > 0, 'Size must be positive!'
        assert isinstance(size, int), 'Size must be integer!'
        assert os.path.isdir(dir_params), "Given folder doesn't exist!"
        
        fpath_params_individual = os.path.join(dir_params, PARAMS_INDIVIDUAL)
        fpath_params_model = os.path.join(dir_params, PARAMS_MODEL)
        status = read_json_as_dict(fpath_params_model)['status']
        features = read_json_as_dict(fpath_params_individual)
    
        Individual._status = pd.DataFrame(
            index=range(size), columns=status, dtype='float')
        Individual._features = pd.DataFrame()
        for feature, distribution in features.items():
            Individual._features[feature] = np.random.choice(
                distribution[0], size, p=distribution[1]
            )

            # --- Define all possible columns (including those not present in the sample) --- #
            # When the sample size is too small, this doesn't cover all categories,
            # the resulting DataFrame thus lacks those columns.
            # To solve the issue, we ensure all possible categories are present
            # when creating the dummy variables
            all_possible_cols = [
                'age_group__1',
                'age_group__2',
                'age_group__3',
                'age_group__4',
                'gender_f',
                'gender_m',
                'education_high',
                'education_low',
                'education_medium',
                'unemployed_no',
                'unemployed_yes',
                'have_partner_no',
                'have_partner_yes',
                'depressed_no',
                'depressed_yes',
                'children_presence_no',
                'children_presence_yes',
                'housing_financial_difficulties_no',
                'housing_financial_difficulties_yes',
                'selfrated_health_average',
                'selfrated_health_good',
                'selfrated_health_poor',
                'critical_job_no',
                'critical_job_yes'
            ]

        # one-hot encoding
        categorical_cols = Individual._features.select_dtypes(include=['object'])
        encoded_cols = pd.get_dummies(categorical_cols).reindex(columns=all_possible_cols, fill_value=0)
        Individual._features.drop(categorical_cols.columns, axis=1, inplace=True)
        Individual._features = pd.concat([Individual._features, encoded_cols], axis=1)

        # Add 'baseline' column filled with ones
        Individual._features.insert(0, "baseline", 1)

        return [Individual(i, dir_params) for i in tqdm(range(size), desc="Populating individuals", unit="i")]