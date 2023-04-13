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
        """Get feature values of an individual

        Returns:
            _type_: _description_
        """
        return self._features.iloc[self.id]
        
    def action(self, lockdown):
        """Take action(s) and update status
        """
        pass
    
    @staticmethod
    def populate(size: int, seed: int = 0, **kwargs):
        """Create a population with an arbitrary number of features
        """
        Individual._features = pd.DataFrame() # clear
        
        np.random.seed(seed)
        for feature, distribution in kwargs.items():
            Individual._features[feature] = np.random.choice(
                distribution[0], size, p=distribution[1]
            )
        