"""Individual agent class definition
"""

import pandas as pd
import numpy as np

class Individual:
    _features = pd.DataFrame()
    _status = pd.DataFrame()
    
    def __init__(self, id: int):
        self.id: int = id
        
    def get_features(self):
        return self._features.loc[self.id]
    
    def get_status(self):
        return self._status.loc[self.id]
   
    # TODO: lockdown as input can be combinition of different policies in the future.
    #       Thus, should accept a list 
    def action(self, lockdown: str):
        """Take action(s) and update status
        
        Actions can be found from the hypothesis files.
        """
        assert lockdown in ['absent', 'easy', 'medium', 'hard'], \
            'Lockdown should be one of absent, easy, medium, and hard!'
            
        def read_params(fpath):
            """Read parameter matrix with columns in feature matrix.
            """
            df = pd.read_csv(fpath, delimiter=';')
            cols = self.get_features().index
            return df[cols]
         
        # get actions based on the lockdown input
        fpath_lockdown_params = '../hypotheses/lockdown_%s.csv' % lockdown
        lockdown_params = read_params(fpath_lockdown_params)
        n_actions, _ = lockdown_params.shape
        action_probs = lockdown_params.dot(self.get_features())
        action_probs = action_probs.apply(lambda x: 1 / (1 + np.exp(-x)))
        actions = np.random.rand(n_actions) <= action_probs
       
        # compute status by taking the actions 
        fpath_effect_mh = '../hypotheses/action_effects_on_mh.csv'
        fpath_effect_contacts = '../hypotheses/action_effects_on_contacts.csv'
        effect_mh_params = read_params(fpath_effect_mh)
        effect_contacts_params = read_params(fpath_effect_contacts)
        mh = effect_mh_params.dot(self.get_features()).dot(actions)
        n_contact = effect_contacts_params.dot(self.get_features()).dot(actions)
        self._status.loc[self.id] = (mh, n_contact) 
    
    @staticmethod
    def populate(size: int, **kwargs):
        """Create a population with an arbitrary number of features
        """
        Individual._features = pd.DataFrame()
        Individual._status = pd.DataFrame(
            index=range(size), columns=['mh', 'n_contacts'], dtype='float')
        
        # add all features in kwargs to the feature matrix 
        for feature, distribution in kwargs.items():
            Individual._features[feature] = np.random.choice(
                distribution[0], size, p=distribution[1]
            )
            
        # one-hot encoding all the object columns
        categorical_cols = Individual._features.select_dtypes(include=['object'])
        encoded_cols = pd.get_dummies(categorical_cols).astype(int)
        Individual._features.drop(categorical_cols.columns, axis=1, inplace=True)
        Individual._features = pd.concat([Individual._features, encoded_cols], axis=1)
        
        return [Individual(i) for i in range(size)]