"""Model class definition
"""
from comma.individual import Individual
import pandas as pd
from . import read_json_as_dict, _get_one_hot_encoded_features, PARAMS_INDIVIDUAL, PARAMS_MODEL
import os


class Model:
    _status = dict()
    _required_params = ['size', 'steps', 'actions', 'status',
                        'lockdown_policies', 'lockdown']

    def __init__(self):
        self.agents: list = None
        self.id: int = None

    def setup(self, size: int, dir_params: str):
        """Setup the model with input parameters.

        Args:
            size (int): size of the population
            dir_params (str): dir to the folder containing hypothesis and parameter files.
        """
        self.param_file_validation(dir_params)
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
