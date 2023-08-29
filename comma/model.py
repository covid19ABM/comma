"""Model class definition
"""
from comma.individual import Individual
from comma.hypothesis import Hypothesis
import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


class Model:
    cumulative_status = dict()

    def __init__(self) -> None:
        self.agents: list = None
        self.simulation_id: int = None
        self.current_step: int = 0  # keep track of the current simulation step
        self.lockdown_status: dict = {}
        self.dir_params: str = None

    def setup(self, size: int, dir_params: str,
              use_ipf: bool = False) -> None:
        """Setup the model with input parameters.

        Args:
            size (int): size of the population
            dir_params (str): dir to the folder
            containing hypothesis and parameter files.
            use_ipf (bool): If true, use the populate_ipf()
            method to populate the agents If false,
            use the standard populate method.
        """
        self.dir_params = dir_params
        Hypothesis.validate_param_file(dir_params)
        if use_ipf:
            self.agents = Individual.populate_ipf(size, dir_params)
        else:
            self.agents = Individual.populate(size, dir_params)

    def step(self, lockdown: pd.DataFrame,
             action_effects: pd.DataFrame) -> None:
        """Actions to be performed in each step.

        Args:
            lockdown (pd.DataFrame): lockdown dataframe
            action_effects (pd.Dataframe): actions dataframe
        Returns:
            actions: array of booleans
            action_probs: array of probabilities
        """
        for agent in self.agents:
            # choose actions based on lockdown
            actions, action_probs = agent.choose_actions_on_lockdown(lockdown)
            # take those actions, and compute their effect on mental health
            agent.take_actions(actions=actions, action_effects=action_effects)

    def update(self, lockdown: str, step: int) -> None:
        """
        It updates statuses at every step.

        Args:
            lockdown (str): lockdown type
            step (int): step of the simulation
        """
        if self.current_step == 0:
            delta_mh = 0  # it's day 0, so no incremental change
            self.cumulative_status[step] = [
                (lockdown, agent.id, delta_mh, agent.get_status())
                for agent in self.agents
            ]
        else:
            # from step 1++, sum the agent's status with the previous status
            agent_statuses = []
            for agent in self.agents:
                last_status = \
                    [status for status in self.cumulative_status[step - 1]
                     if status[1] == agent.id][0][-1]
                delta_mh = agent.get_status()  # this is the incremental effect
                mu, sigma = 0.002, 0.0005
                # this is the baseline effect when no action is taken
                # or when action effects are canceled out
                baseline = np.random.normal(mu, sigma)
                new_status = (lockdown, agent.id, delta_mh,
                              (last_status + delta_mh) - baseline)
                agent_statuses.append(new_status)
            self.cumulative_status[step] = agent_statuses

    def report(self, out_path: str) -> None:
        """
        It Collects data recorded at the end of the simulation
        and exports it as csv file.

        Args:
            out_path (str): File path of the output file
        """
        status_data = []

        for step_id, agent_statuses in self.cumulative_status.items():
            for agent_status in agent_statuses:
                lockdown, agent_id, delta_mh, mh = agent_status
                status_data.append([step_id, lockdown, agent_id, delta_mh, mh])

        # create a pandas dataframe to store data from `status_data`
        status_df = pd.DataFrame(
            status_data,
            columns=['step_id', 'lockdown', 'agent_id',
                     'delta_mental_health', 'cumulative_mental_health']
        )

        # Export to a csv
        status_df.to_csv(out_path, index=False, sep=";", decimal=",")

    def run(self, steps: int, lockdown_policy: list, out_path: str) -> None:
        """Run a simulation

        Args:
            steps(int): Number of steps to run the simulation

            lockdown_policy(str): Type of lockdown policy

            out_path(str): File path of the output file
        """
        if len(lockdown_policy) != steps:
            raise ValueError("The length of the lockdown list \
            must be equal to the number of steps")

        lockdown_matrices = Hypothesis.read_hypotheses(
            self.dir_params,
            set(lockdown_policy),
            "lockdown"
        )

        actions_effects_matrices = Hypothesis.read_hypotheses(
            self.dir_params,
            set(lockdown_policy),
            "actions"
        )

        for step, current_lockdown in tqdm(enumerate(lockdown_policy),
                                           total=steps,
                                           desc="Running simulation"):
            self.simulation_id = step
            self.lockdown_status[step] = current_lockdown
            self.step(lockdown_matrices[current_lockdown],
                      actions_effects_matrices[current_lockdown])
            self.update(current_lockdown, step)
            self.current_step += 1  # Increment the simulation step
        self.report(out_path)
