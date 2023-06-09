import os
import json
import pandas as pd


PARAMS_INDIVIDUAL = 'params_individual.json'
PARAMS_MODEL = 'params_model.json'


def read_json_as_dict(fpath: str):
    """Read content of a JSON file as Python dictionary.

    Args:
        fpath (str): file path.

    Returns:
        dict: result dictionary.
    """
    assert os.path.isfile(fpath), 'File not found: %s.' % fpath
    with open(fpath) as file:
        data = json.load(file)
    return data

# TODO: have a separate function of one-hot encoding
def _get_one_hot_encoded_features(fpath_params_individual: str):
    """One-hot encode categorical features in the `params_individual.json` file 
    and return the feature list.

    Args:
        data (dict): input data dict.
    """
    params_individual = read_json_as_dict(fpath_params_individual)
    features = []
    for key, value in params_individual.items():
        if isinstance(value[0][0], str):
            features += [key + '_' + v for v in value[0]]
        else:
            features += [key]
    return features

def create_empty_hypotheses(dir_params: str):
    """Create empty CSV files for storing hypotheses on the impact of actions and lockdown policies on different agent statuses

    Args:
        dir_params (str): The directory of the folder that contains the agent and model parameter files.
    """
    fpath_params_individual = os.path.join(dir_params, PARAMS_INDIVIDUAL)
    fpath_params_model = os.path.join(dir_params, PARAMS_MODEL)
    params_model = read_json_as_dict(fpath_params_model)
    
    actions = params_model['actions']
    lockdown_policies = params_model['lockdown_policies']
    status = params_model['status']
    columns = ['actions', 'baseline']
    columns += _get_one_hot_encoded_features(fpath_params_individual)
    df = pd.DataFrame(0, index=range(len(actions)), columns=columns)
    df['actions'] = actions
    
    output_fpaths = ['lockdown_%s.csv' % l for l in lockdown_policies]
    output_fpaths += ['actions_effects_on_%s.csv' % s for s in status]
    output_fpaths = [os.path.join(dir_params, fp) for fp in output_fpaths]
    for fp in output_fpaths:
        df.to_csv(fp, sep=';', index=False)