"""Hypothesis class definition"""

import json
import os
import pandas as pd

PARAMS_INDIVIDUAL = 'params_individual.json'
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

    lockdown_policies = [
        'absent',
        'easy',
        'medium',
        'hard'
    ]

    individual_status = [
        'mh'
    ]

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

    all_possible_actions = [
        'work_from_home',
        'maintain_physical_distance',
        'stay_at_home',
        'exercise',
        'socialise',
        'travel',
        'seek_help',
        'negative_coping',
        'positive_coping',
        'socialise_online'
    ]

    @classmethod
    def read_actions(cls, dir_params: str):
        fpath_params_status = os.path.join(
            dir_params, 'actions_effects_on_mh.csv')

        df = pd.read_csv(fpath_params_status, delimiter=';', decimal=",")

        # sort rows
        df['actions'] = df['actions'].astype('category')
        df['actions'] = df['actions'].cat.\
            set_categories(cls.all_possible_actions)
        df = df.sort_values(by='actions', ignore_index=True)

        # Convert dataframe column names to lowercase
        df.columns = df.columns.str.lower()

        # Convert cols to lowercase
        cols = [col.lower() for col in cls.all_possible_features]
        cols.insert(0, "baseline")

        # get and sort desired columns
        df = df[cols]

        return df

    @classmethod
    def read_hypotheses(cls, dir_params: str, lockdown: object):
        lockdown_dfs = {}

        for policy in lockdown:
            # Create the file path
            fpath_params_lockdown = os.path.join(
                dir_params, 'lockdown_%s.csv' % policy
            )

            df = pd.read_csv(fpath_params_lockdown, delimiter=';', decimal=",")

            # sort rows
            df['actions'] = df['actions'].astype('category')
            df['actions'] = df['actions'].cat.\
                set_categories(cls.all_possible_actions)
            df = df.sort_values(by='actions', ignore_index=True)

            # Convert dataframe column names to lowercase
            df.columns = df.columns.str.lower()

            # Convert cols to lowercase
            cols = [col.lower() for col in cls.all_possible_features]
            cols.insert(0, "baseline")

            # get and sort desired columns
            df = df[cols]

            lockdown_dfs[policy] = df

        return lockdown_dfs

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

        # Check if the files exist
        if not os.path.exists(fpath_params_individual):
            raise FileNotFoundError(f"'{PARAMS_INDIVIDUAL}' \
            file is missing in the directory '{dir_params}'")

        actions = cls.all_possible_actions
        lockdown_policies = cls.lockdown_policies
        status = cls.individual_status
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

        # check if all hypothesis files exist
        fnames = ["actions_effects_on_%s.csv" %
                  status for status in cls.individual_status]
        fnames += ["lockdown_%s.csv" %
                   lockdown for lockdown in cls.lockdown_policies]
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
        required_actions = cls.all_possible_actions
        missing_actions = [set(required_actions) - set(hd["actions"])
                           for hd in hypothesis_data]
        if any(missing_actions):
            raise ValueError("Missing actions:\n%s" % "\n".join(
                ["%s - %s" % (fnames[i], ", ".join(missing_actions[i]))
                 for i in range(len(fnames)) if missing_actions[i]]
            ))
