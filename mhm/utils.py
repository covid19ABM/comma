""" Useful functions """

import numpy as np
import pandas as pd
from dataclasses import asdict


def label_age_range(x):
    """
    Labels the specific age range that an agent has been assigned to

    Parameters
    ----------
    x : list
        Array of age values
    
    Returns
    -------
    y : numpy array of strings
        Array of age group labels
    """
    bins = [24, 35, 45, 55, 65]  # define the bins
    labels = [1, 2, 3, 4]  # define the labels
    y = np.digitize(x, bins=bins, right=True)  # assign values based on the bins
    y = np.array(labels)[y - 1]  # assign labels based on the assigned values
    y = np.array([f"_{i}" for i in y])
    return y


def one_hot_encoding(df, cols):
    """
    Applies one-hot encoding to a list of columns in a Pandas dataframe.
    
    Parameters
    ----------
    df : pandas dataframe
        The dataframe to encode
    
    cols : list
        A list of column names to apply one-hot encoding.
    
    Returns
    -------
    encoded_df : pandas dataframe
        A new dataframe where each specified column has
    been replaced with a set of one-hot encoded columns.
    """    
    encode_cols = df[cols]
    encoded_cols = pd.get_dummies(encode_cols)
    encoded_df = pd.concat([df, encoded_cols], axis=1)
    encoded_df.drop(cols, axis=1, inplace=True)

    return encoded_df


def make_features_dataframe(agent_list):
    '''
    Takes a list of agents, 
    returns pandas dataframe of features.
    
    Parameters
    ----------
    agent_list : list
        List of dataclasses of the type Agent

    Returns
    -------
    df : pandas dataframe
        Dataframe of the agents
    '''
    df = []
    for p in agent_list:
        df.append(asdict(p.features))
    df = pd.DataFrame.from_dict(df)
    return df