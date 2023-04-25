"""Model class definition
"""

class Model:
    _status = dict()
    
    def __init__(self):
        self.agents: list = None
        self.id: int = None
        self.params: dict = None
        
    def setup(self, params: dict):
        """Setup the model with input parameters.

        Args:
            params (dict): parameters of the model
        """
        pass
    
    def step(self):
        """Actions to be performed in each step.
        """
        pass
    
    def update(self):
        """Data to be recorded in each step.
        """
        pass
    
    def report(self):
        """Collect data recorded at the end of the simulation and export as files.
        """
        pass
    
    def run(self):
        """Run a simulation
        """
        pass