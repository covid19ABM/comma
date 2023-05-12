"""Model class definition
"""
from mhm.individual import Individual
import pandas as pd


class Model:
    _status = dict()
    
    def __init__(self):
        self.agents: list = None
        self.id: int = None

    def setup(self, size: int, dir_params: str):
        """Setup the model with input parameters.

        Args:
            size (int): size of the population

            dir_params (str): dir to the folder containing feature parameter files.
        """
        self.agents = Individual.populate(size, dir_params)
        return self.agents

    def step(self, lockdown: str):
        """Actions to be performed in each step.

        Args:
            lockdown (str): lockdown type
        """
        for agent in self.agents:
            actions, action_probs = agent.choose_actions_on_lockdown(lockdown)
            agent.take_actions(actions=actions)
    
    def update(self, lockdown: str):
        """Data to be recorded in each step.

        Args:
            lockdown (str): lockdown type
        """
        agent_statuses = [(agent.id, *agent.get_status()) for agent in self.agents]
        self._status[self.id] = (lockdown, agent_statuses)

    def report(self, out_path: str):
        """Collect data recorded at the end of the simulation and export as csv file.

        Args:
            out_path (str): File path of the output file
        """
        status_data = []

        for step_id, (lockdown, agent_statuses) in self._status.items():
            for agent_status in agent_statuses:
                agent_id, mh, n_contacts = agent_status
                status_data.append([step_id, agent_id, lockdown, mh, n_contacts])

        # create a pandas dataframe to store data from `status_data`
        status_df = pd.DataFrame(status_data, columns=['step_id', 'agent_id', 'lockdown', 'mh', 'n_contacts'])

        # Export to a csv
        status_df.to_csv(out_path, index=False)

    def run(self, steps: int, lockdown: list, out_path: str):
        """Run a simulation

        Args:
            steps (int): Number of steps to run the simulation

            lockdown (str): Type of lockdown (easy, hard, medium, no_lockdown)

            out_path (str): File path of the output file
        """
        if len(lockdown) != steps:
            raise ValueError("The length of the lockdown list must be equal to the number of steps")

        for step, current_lockdown in enumerate(lockdown):
            self.id = step
            self.step(current_lockdown)
            self.update(current_lockdown)
        self.report(out_path)