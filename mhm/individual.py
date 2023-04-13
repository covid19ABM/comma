"""Individual agent class definition
"""

import pandas as pd
import numpy as np

class Individual:
    features = pd.DataFrame()
    
    def __init__(self):
        self.id: int = None
        self.status: dict = None
        
    def setup(self, id: int):
        """Setup an agent

        Args:
            id (int): _description_
        """
        self.id = id
        
    def action(self):
        """Perform action and compute mental effect
        """
        pass
    
    @staticmethod
    def populate(size: int, seed: int, **kwargs):
        """Create a population with an arbitrary number of features
        """
        Individual.features.drop(Individual.features.index, inplace=True)
        np.random.seed(seed)
        for feature, distribution in kwargs.items():
            Individual.features[feature] = np.random.choice(
                distribution[0], size, p=distribution[1]
            )
        