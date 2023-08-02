"""Model class definition
"""
from comma.individual import Individual
import pandas as pd
from . import read_json_as_dict, _get_one_hot_encoded_features, PARAMS_INDIVIDUAL, PARAMS_MODEL
import os
import numpy as np
from tqdm import tqdm


class Model:
    _status = dict()
    _required_params = ['size', 'steps', 'actions', 'status',
                        'lockdown_policies', 'lockdown']

    def __init__(self):
        self.agents: list = None
        self.simulation_id: int = None
        self.current_step: int = 0 # keep track of the current simulation step
        self.lockdown_status: dict = {}

    def setup(self, size: int, dir_params: str, use_ipf: bool = False):
        """Setup the model with input parameters.

        Args:
            size (int): size of the population
            dir_params (str): dir to the folder containing hypothesis and parameter files.
            use_ipf (bool): If true, use the populate_ipf() method to populate the agents
                            If false, use the standard populate method.
        """
        self.param_file_validation(dir_params)
        if use_ipf:
            self.agents = Individual.populate_ipf(size, dir_params)
        else:
            self.agents = Individual.populate(size, dir_params)
        return self.agents

    def param_file_validation(self, dir_params: str):
        """Validate files in the parameter folder.
        
        Args:
            dir_params (str): dir to the folder containing hypothesis and parameter files.
        """
        # check if parameter files exist
        fpath_params_individual = os.path.join(dir_params, PARAMS_INDIVIDUAL)
        fpath_params_model = os.path.join(dir_params, PARAMS_MODEL)
        assert os.path.isfile(fpath_params_individual), \
            "Parameter file not found: %s." % PARAMS_INDIVIDUAL
        assert os.path.isfile(fpath_params_model), \
            "Parameter file not found: %s" % PARAMS_MODEL

        # check if all required model parameters are given 
        params_model = read_json_as_dict(fpath_params_model)
        missing_model_params = set(self._required_params) - set(params_model.keys())
        assert not missing_model_params, \
            "Model parameter(s) not found: %s." % ", ".join(missing_model_params)

        # check if there is any overlap between lockdown intervals
        lockdown = params_model["lockdown"]
        lockdown_intervals = [l["interval"] for l in lockdown]
        lockdown_intervals = sorted(lockdown_intervals, key=lambda li: li[0])
        overlaps = [lockdown_intervals[i][1] >= lockdown_intervals[i + 1][0] \
                    for i in range(len(lockdown_intervals) - 1)]
        assert not any(overlaps), "Lockdown intervals have overlap: \n%s" % "\n".join(
            ["%s - %s" % (str(lockdown_intervals[i]), str(lockdown_intervals[i + 1])) \
             for i in range(len(overlaps)) if overlaps[i]]
        )

        # check if all the steps are fully covered by the given lockdown intervals
        steps = params_model["steps"]
        expected_timeline = list(range(1, steps + 1))
        given_timeline = []
        for li in lockdown_intervals:
            given_timeline += list(range(li[0], li[1] + 1))
        uncovered_steps = set(expected_timeline) - set(given_timeline)
        assert not uncovered_steps, "Uncovered steps: %s." % ", ".join(
            [str(us) for us in uncovered_steps]
        )

        # check if all hypothesis files exist
        fnames = ["actions_effects_on_%s.csv" % s for s in params_model["status"]]
        fnames += ["lockdown_%s.csv" % l for l in params_model["lockdown_policies"]]
        fpaths = [os.path.join(dir_params, fn) for fn in fnames]
        fexist = [os.path.isfile(fp) for fp in fpaths]
        assert all(fexist), "Hypothesis file(s) not found: %s." % ", ".join(
            [fnames[i] for i in range(len(fnames)) if not fexist[i]]
        )

        # check if all hypothesis files contain all the required agent features
        required_features = ["actions", "baseline"]
        required_features += _get_one_hot_encoded_features(fpath_params_individual)
        hypothesis_data = [pd.read_csv(fp, sep=";", decimal=",") for fp in fpaths]
        missing_features = []
        for hd in hypothesis_data:
            # avoid raising an issue if the mismatch it's in the case (e.g., upper vs lower case)
            missing_features.append(set([f.lower() for f in required_features]) \
                                    - set([c.lower() for c in hd.columns]))

        assert not any(missing_features), "Missing features:\n%s" % "\n".join(
            ["%s - %s" % (fnames[i], ", ".join(missing_features[i])) \
             for i in range(len(fnames)) if missing_features[i]]
        )

        # check if all hypothesis files contain hypotheses of all actions
        required_actions = params_model["actions"]
        missing_actions = [set(required_actions) - set(hd["actions"]) for hd in hypothesis_data]
        assert not any(missing_actions), "Missing actions:\n%s" % "\n".join(
            ["%s - %s" % (fnames[i], ", ".join(missing_actions[i])) \
             for i in range(len(fnames)) if missing_actions[i]]
        )

    def step(self, lockdown: str):
        """Actions to be performed in each step.

        Args:
            lockdown (str): lockdown type
        """
        for agent in self.agents:
            # choose actions based on lockdown
            actions, action_probs = agent.choose_actions_on_lockdown(lockdown)
            # take those actions, and compute their effect on mental health
            agent.take_actions(actions=actions)

    def update(self, lockdown: str, step: int):
        """Update status at every step.

        Args:
            lockdown (str): lockdown type
            step (int): step of the simulation
        """
        if self.current_step == 0:
            # it's step 0, i.e., simulation just started, so we simply store the agent's status
            delta_mh = 0 # it's day 0, so no incremental change
            self._status[step] = [(lockdown, agent.id, delta_mh, *agent.get_status()) for agent in self.agents]


        else:
            # from step 1++, sum the agent's status with the status from the previous step
            agent_statuses = []
            for agent in self.agents:
                last_status = [status for status in self._status[step - 1] if status[1] == agent.id][0][-1]
                delta_mh = agent.get_status()[0] # this is the incremental effect
                mu, sigma = 0.002, 0.0005
                # this is the baseline effect when no action is taken
                # or when action effects are canceled out
                baseline = np.random.normal(mu, sigma)
                new_status = (lockdown, agent.id, delta_mh, (last_status + delta_mh) - baseline)
                agent_statuses.append(new_status)
            self._status[step] = agent_statuses

    def report(self, out_path: str):
        """Collect data recorded at the end of the simulation and export as csv file.

        Args:
            out_path (str): File path of the output file
        """
        status_data = []

        for step_id, agent_statuses in self._status.items():
            for agent_status in agent_statuses:
                lockdown, agent_id, delta_mh, mh = agent_status
                status_data.append([step_id, lockdown, agent_id, delta_mh, mh])

        # create a pandas dataframe to store data from `status_data`
        status_df = pd.DataFrame(status_data, columns=['step_id', 'lockdown', 'agent_id',
                                                       'delta_mental_health', 'cumulative_mental_health'])

        # Export to a csv
        status_df.to_csv(out_path, index=False, sep=";", decimal=",")

    def run(self, steps: int, lockdown: list, out_path: str):
        """Run a simulation

        Args:
            steps (int): Number of steps to run the simulation

            lockdown (str): Type of lockdown (easy, hard, medium, no_lockdown)

            out_path (str): File path of the output file
        """
        if len(lockdown) != steps:
            raise ValueError("The length of the lockdown list must be equal to the number of steps")

        for step, current_lockdown in tqdm(enumerate(lockdown), total=steps, desc="Running simulation"):
            self.simulation_id = step
            self.lockdown_status[step] = current_lockdown
            self.step(current_lockdown)
            self.update(current_lockdown, step)
            self.current_step += 1  # Increment the simulation step
        self.report(out_path)
