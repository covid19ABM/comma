"""Individual agent class definition
"""

import pandas as pd
import numpy as np
import json

# list of names of lockdown policies
LOCKDOWN_POLICIES = ['absent', 'easy', 'medium', 'hard']

class Individual:
    _features = pd.DataFrame()
    _status = pd.DataFrame()
    
    def __init__(self, id: int):
        self.id: int = id
        
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
            fpath_lockdown_params = '../hypotheses/lockdown_%s.csv' % lockdown 
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
            fpath_effect_mh = '../hypotheses/action_effects_on_mh.csv'
        if fpath_effect_contacts is None:
            fpath_effect_contacts = '../hypotheses/action_effects_on_contacts.csv'
            
        effect_mh_params = self._read_params(fpath_effect_mh)
        effect_contacts_params = self._read_params(fpath_effect_contacts)
        mh = effect_mh_params.dot(self.get_features()).dot(actions)
        n_contact = effect_contacts_params.dot(self.get_features()).dot(actions)
        self._status.loc[self.id] = (mh, n_contact) 
       
    @staticmethod    
    def read_features_from_file(fpath_features='../parameters/agent_features.json') -> dict:
        """Read features (values, probabilities) from a JSON file.

        Args:
            fpath_features (str): path to the feature file.

        Returns:
            dict: dictionary of features
        """
        with open(fpath_features) as json_file:
            features = json.load(json_file)
            
        return features 
    
    @staticmethod
    def populate(size: int, fpath_features: str = None, **kwargs):
        """Create a population with an arbitrary number of features
        """
        Individual._features = pd.DataFrame()
        Individual._status = pd.DataFrame(
            index=range(size), columns=['mh', 'n_contacts'], dtype='float')
        
        # add all features in kwargs to the feature matrix 
        if fpath_features is None:
            features = kwargs
        elif fpath_features == 'Default':
            features = Individual.read_features_from_file()
        else:
            features = Individual.read_features_from_file(fpath_features)
        for feature, distribution in features.items():
            Individual._features[feature] = np.random.choice(
                distribution[0], size, p=distribution[1]
            )
            
        # one-hot encoding all the object columns
        categorical_cols = Individual._features.select_dtypes(include=['object'])
        encoded_cols = pd.get_dummies(categorical_cols).astype(int)
        Individual._features.drop(categorical_cols.columns, axis=1, inplace=True)
        Individual._features = pd.concat([Individual._features, encoded_cols], axis=1)
        
        return [Individual(i) for i in range(size)]