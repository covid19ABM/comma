"""Individual agent class definition
"""

import pandas as pd
import numpy as np
import os
from . import read_json_as_dict
from . import PARAMS_INDIVIDUAL, PARAMS_MODEL


LOCKDOWN_POLICIES = ['absent', 'easy', 'medium', 'hard']
ACTIONS = [
    'go_to_work', 'maintain_physical_distance', 'stay_at_home', 
    'exercise', 'socialise', 'travel', 'seek_help', 
    'negative_coping', 'positive_coping', 'socialise_online'
]
STATUS = ['contacts', 'mh']


class Individual:
    _features = pd.DataFrame()
    _status = pd.DataFrame()
    
    def __init__(self, id: int, dir_params: str):
        self.id: int = id
        self.dir_params = dir_params
        
    def get_features(self):
        return self._features.loc[self.id]
    
    def get_status(self):
        return self._status.loc[self.id]
        
    def _read_params(self, fpath):
        """Read parameter matrix with columns in feature matrix.
        """
        df = pd.read_csv(fpath, delimiter=';')
        cols = self.get_features().index
        return df[cols]
        
    def choose_actions_on_lockdown(self, lockdown: str, fpath_lockdown_params: str = None):
        """Take action(s) and update status
        
        Actions can be found from the hypothesis files.
        """
        assert lockdown in LOCKDOWN_POLICIES, 'Lockdown name incorrect!'
            
        # get actions based on the lockdown input
        if fpath_lockdown_params is None: 
            fpath_lockdown_params = '../parameters/lockdown_%s.csv' % lockdown 
        lockdown_params = self._read_params(fpath_lockdown_params)
        n_actions, _ = lockdown_params.shape
        action_probs = lockdown_params.dot(self.get_features())
        action_probs = action_probs.apply(lambda x: 1 / (1 + np.exp(-x)))
        actions = np.random.rand(n_actions) <= action_probs
        return actions, action_probs
    
    def take_actions(self, actions: list, fpath_effect_mh: str = None, fpath_effect_contacts: str = None):
        """Update the status by taking specific action(s).
        """
        if fpath_effect_mh is None:
            fpath_effect_mh = '../parameters/action_effects_on_mh.csv'
        if fpath_effect_contacts is None:
            fpath_effect_contacts = '../parameters/action_effects_on_contacts.csv'
            
        effect_mh_params = self._read_params(fpath_effect_mh)
        effect_contacts_params = self._read_params(fpath_effect_contacts)
        mh = effect_mh_params.dot(self.get_features()).dot(actions)
        n_contact = effect_contacts_params.dot(self.get_features()).dot(actions)
        self._status.loc[self.id] = (mh, n_contact) 
       
    @staticmethod
    def populate(size: int, dir_params: str): 
        """Create a population of individual agents with the given feature parameters.
        
        Args:
            size (int): population size, i.e., number of agents.
            dir_params (str): dir to the folder containing feature parameter file.
            from_scratch (bool, optional): flag of creating hypothesis from scratch or reading from files. Defaults to False.

        Returns:
            list[Individual]: a list of Individual agents
        """
        assert size > 0, 'Size must be positive!'
        assert isinstance(size, int), 'Size must be integer!'
        assert os.path.isdir(dir_params), "Given folder doesn't exist!"
    
        Individual._features = pd.DataFrame()
        Individual._status = pd.DataFrame(
            index=range(size), columns=STATUS, dtype='float')
        
        fpath_params_individual = os.path.join(dir_params, PARAMS_INDIVIDUAL)
        features = read_json_as_dict(fpath_params_individual)
        for feature, distribution in features.items():
            Individual._features[feature] = np.random.choice(
                distribution[0], size, p=distribution[1]
            )
        categorical_cols = Individual._features.select_dtypes(include=['object'])
        encoded_cols = pd.get_dummies(categorical_cols).astype(int)
        Individual._features.drop(categorical_cols.columns, axis=1, inplace=True)
        Individual._features = pd.concat([Individual._features, encoded_cols], axis=1)
        
        return [Individual(i, dir_params) for i in range(size)]