"""Individual agent class definition
"""

import pandas as pd
import numpy as np

class Individual:
    _features = pd.DataFrame()
    
    def __init__(self):
        self.id: int = None
        self.status: dict = None
        
    def setup(self, id: int):
        """Setup id, status, etc. for an individual

        Args:
            id (int): _description_
        """
        self.id = id
        
    def get_features(self):
        """Get feature values of an individual

        Returns:
            _type_: _description_
        """
        return self._features.iloc[self.id]
        
    def action(self):
        """Perform action and compute mental effect
        """
        pass
    
    @staticmethod
    def populate(size: int, seed: int = 0, **kwargs):
        """Create a population with an arbitrary number of features
        """
        Individual._features.drop(Individual._features.index, inplace=True) # clear
        
        np.random.seed(seed)
        for feature, distribution in kwargs.items():
            Individual._features[feature] = np.random.choice(
                distribution[0], size, p=distribution[1]
            )
        