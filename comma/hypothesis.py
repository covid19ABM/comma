"""Hypothesis class definition"""

import json
import os
import pandas as pd

PARAMS_INDIVIDUAL = 'params_individual.json'
PARAMS_MODEL = 'params_model.json'
PARAMS_IPF_WEIGHTS = "ipf_weights.csv"


class Hypothesis:
    """
    The Hypothesis class is responsible for managing and validating
    the Hypotheses specified by the user.
    These can be found in the `dir_params` folder.

    Methods:
        _get_one_hot_encoded_features():
            One-hot encodes categorical features
        create_empty_hypotheses():
            Creates empty CSV files for storing hypotheses
        validate_param_file():
            Validates the files in the parameter folder

    Usage:
        Hypothesis.create_empty_hypotheses("/path/to/dir_params")
        Hypothesis.validate_param_file("/path/to/dir_params")
    """
    _required_params = ['size', 'steps', 'actions', 'status',
                        'lockdown_policies', 'lockdown']

    all_possible_features = [
        'age_group__1',
        'age_group__2',
        'age_group__3',
        'age_group__4',
        'gender_f',
        'gender_m',
        'education_high',
        'education_low',
        'education_medium',
        'unemployed_no',
        'unemployed_yes',
        'have_partner_no',
        'have_partner_yes',
        'depressed_no',
        'depressed_yes',
        'children_presence_no',
        'children_presence_yes',
        'housing_financial_difficulties_no',
        'housing_financial_difficulties_yes',
        'selfrated_health_average',
        'selfrated_health_good',
        'selfrated_health_poor',
        'critical_job_no',
        'critical_job_yes'
    ]

    @staticmethod
    def _get_one_hot_encoded_features(fpath_params_individual: str):
        """
        One-hot encode categorical features in the
        `params_individual.json` file and return the
        feature list.

        Args:
            fpath_params_individual (str): Path to the
            individual parameters JSON file.

        Returns:
            features (list): List of one-hot encoded features.

        """
        with open(fpath_params_individual) as f:
            params_individual = json.load(f)

        features = []
        for key, value in params_individual.items():
            if isinstance(value[0][0], str):
                features += [key + '_' + v for v in value[0]]
            else:
                features += [key]
        return features

    @classmethod
    def create_empty_hypotheses(cls, dir_params: str):
        """
        Create empty CSV files for storing hypotheses on
        the impact of actions and lockdown policies on different agent statuses

        Args:
            dir_params (str): The directory of the folder that contains
            the agent and model parameter files.
        """
        fpath_params_individual = os.path.join(dir_params, PARAMS_INDIVIDUAL)
        fpath_params_model = os.path.join(dir_params, PARAMS_MODEL)

        # Check if the files exist
        if not os.path.exists(fpath_params_individual):
            raise FileNotFoundError(f"'{PARAMS_INDIVIDUAL}' \
            file is missing in the directory '{dir_params}'")
        if not os.path.exists(fpath_params_model):
            raise FileNotFoundError(f"'{PARAMS_MODEL}' \
            file is missing in the directory '{dir_params}'")

        with open(fpath_params_model) as f:
            params_model = json.load(f)

        actions = params_model['actions']
        lockdown_policies = params_model['lockdown_policies']
        status = params_model['status']
        columns = ['actions', 'baseline']
        columns += cls._get_one_hot_encoded_features(fpath_params_individual)
        df = pd.DataFrame(0, index=range(len(actions)), columns=columns)
        df['actions'] = actions

        output_fpaths = ['lockdown_%s.csv' % lockdown for
                         lockdown in lockdown_policies]
        output_fpaths += ['actions_effects_on_%s.csv' % s for s in status]
        output_fpaths = [os.path.join(dir_params, fp) for fp in output_fpaths]
        for fp in output_fpaths:
            df.to_csv(fp, sep=';', index=False)

    @classmethod
    def validate_param_file(cls, dir_params: str):
        """Validate files in the parameter folder.

        Args:
            dir_params (str): dir to the folder containing
            hypothesis and parameter files.

        Raises:
            ValueError: If any validation checks fail.
        """
        # check if parameter files exist
        path_individual = os.path.join(dir_params, PARAMS_INDIVIDUAL)
        path_model = os.path.join(dir_params, PARAMS_MODEL)

        # check if all required model parameters are given
        with open(path_model) as f:
            params_model = json.load(f)

        missing_model_params = set(cls._required_params) - \
            set(params_model.keys())

        if missing_model_params:
            raise ValueError("Model parameter(s) not found: %s."
                             % ", ".join(missing_model_params))

        # check if there is any overlap between lockdown intervals
        lockdown = params_model["lockdown"]
        lockdown_intervals = [policy["interval"] for policy in lockdown]
        lockdown_intervals = sorted(lockdown_intervals, key=lambda li: li[0])
        overlaps = [lockdown_intervals[i][1] >= lockdown_intervals[i + 1][0]
                    for i in range(len(lockdown_intervals) - 1)]
        if any(overlaps):
            overlap_strings = [
                "%s - %s" % (str(lockdown_intervals[i]),
                             str(lockdown_intervals[i + 1]))
                for i in range(len(overlaps)) if overlaps[i]
            ]

            error_message = "Lockdown intervals have overlap: \n%s" \
                            % "\n".join(overlap_strings)

            raise ValueError(error_message)

        # check that steps are covered by the given lockdown intervals
        steps = params_model["steps"]
        expected_timeline = list(range(1, steps + 1))
        given_timeline = []
        for li in lockdown_intervals:
            given_timeline += list(range(li[0], li[1] + 1))
        uncovered_steps = set(expected_timeline) - set(given_timeline)
        if uncovered_steps:
            raise ValueError("Uncovered steps: %s." % ", ".join(
                [str(us) for us in uncovered_steps]
            ))

        # check if all hypothesis files exist
        fnames = ["actions_effects_on_%s.csv" %
                  status for status in params_model["status"]]
        fnames += ["lockdown_%s.csv" %
                   lockdown for lockdown in params_model["lockdown_policies"]]
        fpaths = [os.path.join(dir_params, fn) for fn in fnames]
        fexist = [os.path.isfile(fp) for fp in fpaths]
        if not all(fexist):
            raise ValueError("Hypothesis file(s) not found: %s." % ", ".join(
                [fnames[i] for i in range(len(fnames)) if not fexist[i]]
            ))

        # check if all hypothesis files contain all the required agent features
        required_features = ["actions", "baseline"]
        required_features += cls._get_one_hot_encoded_features(path_individual)
        hypothesis_data = [pd.read_csv(fp, sep=";", decimal=",")
                           for fp in fpaths]
        missing_features = []
        for hd in hypothesis_data:
            # lower case labels
            missing_features.append(set([f.lower() for f in required_features])
                                    - set([c.lower() for c in hd.columns]))

        if any(missing_features):
            raise ValueError("Missing features:\n%s" % "\n".join(
                ["%s - %s" % (fnames[i], ", ".join(missing_features[i]))
                 for i in range(len(fnames)) if missing_features[i]]
            ))

        # check if all hypothesis files contain hypotheses of all actions
        required_actions = params_model["actions"]
        missing_actions = [set(required_actions) - set(hd["actions"])
                           for hd in hypothesis_data]
        if any(missing_actions):
            raise ValueError("Missing actions:\n%s" % "\n".join(
                ["%s - %s" % (fnames[i], ", ".join(missing_actions[i]))
                 for i in range(len(fnames)) if missing_actions[i]]
            ))
