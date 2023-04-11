"""Individual agent class definition
"""

import pandas as pd

class Individual:
    _features = pd.DataFrame()
    
    def __init__(self):
        self.id: int = None
        self.status: dict = None
        
    def setup(self):
        """Setup an agent

        Args:
            id (int): _description_
        """
        pass
        
    def action(self):
        """Perform action and compute mental effect
        """
        pass
    
    @staticmethod
    def populate(parameters: dict):
        """Fill values in the _features matrix 
        """
        pass