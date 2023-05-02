"""Individual agent class definition
"""

import pandas as pd
import numpy as np
import json
import os


LOCKDOWN_POLICIES = ['absent', 'easy', 'medium', 'hard']
ACTIONS = [
    'go_to_work', 'maintain_physical_distance', 'stay_at_home', 
    'exercise', 'socialise', 'travel', 'seek_help', 
    'negative_coping', 'positive_coping', 'socialise_online'
]
DEFAULT_PARAMS_DIR = '../parameters/'
DEFAULT_FEATURES = os.path.join(DEFAULT_PARAMS_DIR, 'agent_features.json')


class Individual:
    _features = pd.DataFrame()
    _status = pd.DataFrame()
    _dir_params = ''
    
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
    def _read_features_from_file(fpath_features=DEFAULT_FEATURES) -> dict:
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
    def _create_hypothesis_files():
        """Create CSV files for storing hypothesis parameters
        """
        features = Individual._features.columns.tolist()
        features.insert(0, 'baseline')
        df = pd.DataFrame(0, index=range(len(ACTIONS)), columns=features)
        df.insert(0, 'actions', ACTIONS)
        
        fpaths = ["lockdown_%s.csv" % l for l in LOCKDOWN_POLICIES]
        fpaths += ['actions_effects_on_contacts.csv', 'actions_effects_on_mh.csv']
        fpaths = [os.path.join(DEFAULT_PARAMS_DIR, fp) for fp in fpaths]
        for fp in fpaths:
            df.to_csv(fp, sep=';', index=False) 
    
    @staticmethod
    def populate(size: int, dir_params: str, from_scratch: bool = False): 
        """Create a population of individual agents with the given feature parameters.
        
        Args:
            size (int): population size, i.e., number of agents.
            dir_params (str): dir to the folder containing feature parameter file.
            from_scratch (bool, optional): flag of creating hypothesis from scratch or reading from files. Defaults to False.

        Returns:
            list[Individual]: a list of Individual agents
        """
        assert size > 0, 'Size must be positive!'
        assert type(size) == int, 'Size must be integer!'
        assert os.path.isdir(dir_params), "Given folder doesn't exist!"
        Individual._dir_params = dir_params
        
        # read prior parameters from file
        fpath_params_features = os.path.join(dir_params, 'params_features.json')
        assert os.path.isfile(fpath_params_features), \
            "Prior parameter file doesn't exist in the given folder, \
                file name should be params_features.json"
        features = Individual._read_features_from_file(fpath_params_features)
            
        # initialize features and status matrices
        Individual._features = pd.DataFrame()
        Individual._status = pd.DataFrame(
            index=range(size), columns=['mh', 'n_contacts'], dtype='float')
        for feature, distribution in features.items():
            Individual._features[feature] = np.random.choice(
                distribution[0], size, p=distribution[1]
            )
        categorical_cols = Individual._features.select_dtypes(include=['object'])
        encoded_cols = pd.get_dummies(categorical_cols).astype(int)
        Individual._features.drop(categorical_cols.columns, axis=1, inplace=True)
        Individual._features = pd.concat([Individual._features, encoded_cols], axis=1)
        
        # create empty hypothesis params files if needed
        if from_scratch:
            Individual._create_hypothesis_files()
        
        return [Individual(i) for i in range(size)]