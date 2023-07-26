import pytest
from comma.individual import Individual
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


gender = ['Male', 'Female']
age_cat = ["24-34", "35-44", "45-54", "55-64"]
education_level = ["Low", "Middle", "High"]
unemployed = ["Yes", "No"]
partner = ["Yes", "No"]
depressed = ["Yes", "No"]
children_presence = ["Yes", "No"]
housing_financial_trouble = ["Yes", "No"]
selfrated_health = ["Good/veryGood", "Average", "Poor/veryPoor"]
critical_job = ["Yes", "No"]

# define the cross table gender x education_level
cross0 = pd.DataFrame([[86, 238, 725], [106, 396, 1104]], index=gender, columns=education_level)
# define the cross table age_cat x education_level
cross9 = pd.DataFrame([[6, 97, 539], [34, 161, 486], [109, 296, 596], [43, 80, 208]], index=age_cat, columns=education_level)
# define the cross table education_level x unemployed
cross17 = pd.DataFrame([[10, 182], [31, 603], [65, 1764]], index=education_level, columns=unemployed)
# Define the dataframe for Partner x Depressed
cross24 = pd.DataFrame([[45, 2069], [21, 520]], index=partner, columns=depressed)
# Define the dataframe for Depressed x Children
cross29 = pd.DataFrame([[27, 39], [1310, 1279]], index=depressed, columns=children_presence)
# Define the dataframe for Children x Housing/Financial troubles
cross33 = pd.DataFrame([[292, 1045], [476, 842]], index=children_presence, columns=housing_financial_trouble)
# Define the dataframe for Unemployed x Partner
cross36 = pd.DataFrame([[70, 36], [2044, 505]], index=unemployed, columns=partner)
# Define the dataframe for Housing/Financial trouble x Self-rated health
cross41 = pd.DataFrame([[874, 928, 85], [269, 436, 63]], index=housing_financial_trouble, columns=selfrated_health)
# Define the dataframe for Housing/Financial trouble x Critical job
cross42 = pd.DataFrame([[304, 464], [808, 1079]], index=housing_financial_trouble, columns=critical_job)
# Define the dataframe for Self-rated health x Critical job
cross43 = pd.DataFrame([[471, 672], [571, 793], [70, 78]], index=selfrated_health, columns=critical_job)


def chisq_of_df_cols(df, c1, c2):
    """
    Conduct a Chi-squared test of independence on two categorical columns of a DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the data.
        c1 (str): The name of the first column to test.
        c2 (str): The name of the second column to test.

    Returns:
        float: The p-value from the Chi-squared test.
    """
    groupsizes = df.groupby([c1, c2]).size()
    ctsum = groupsizes.unstack(c1)
    # fillna(0) is necessary to remove any NAs which may cause exceptions
    result = chi2_contingency(ctsum.fillna(0))
    return result[1] # return only the p-value

@pytest.mark.parametrize('time', range(100))
def test_data_sampling_ipf(time):
    """Test if the sampling result aligns to the cross-tabs.
    """
    
    size = 2655 # This is the original sample size provided by Kristina
    dir_params = "./parameters"
    sample_set = Individual.data_sampling_ipf(size, dir_params)
    cols = sample_set.columns.tolist()
    
    # Define crosstabs of the original sample (i.e., truth). 
    # We have 43 crosstabs in total, however, they are ad-hoc combinations of 9 variables.
    # It's sufficient to test whether the distribution of those 9 variables changes significantly than the original sample, rather than all 43 combinations.
    
    crosstabs_dict = {
    'original_gender': cross0,
    'original_age_cat': cross9,
    'original_education_level': cross17,
    'original_partner': cross24,
    'original_depressed': cross29,
    'original_children': cross33,
    'original_unemployed': cross36,
    'original_housing_financial_trouble': cross41,
    'original_selfrated_health': cross43,
    'original_critical_job': cross42
    }
    
    _keys = crosstabs_dict.keys()
    
    # df that stores the results
    results_df = pd.DataFrame(columns=['var', 'key', 'pvalue'])
    
    for var, key in zip(cols, _keys):
        # group by variable and count the instances
        cross_ipf = sample_set.groupby(var).count().iloc[:, 0]
        # create a DataFrame with index as the variable type and count as the number of instances
        df_cross_ipf = pd.DataFrame({'type': cross_ipf.index, 'count': cross_ipf.values})
        # add a 'sample' column to each df before concatenating
        df_cross_ipf['sample'] = 'ipf'
        
        # get the corresponding crosstab from the original sample of Kristina
        crosstab = crosstabs_dict[key]
        cross_ground_truth = crosstab.sum(axis=1)
        # format the df as per above
        df_cross_ground_truth = pd.DataFrame({'type': cross_ground_truth.index, 'count': cross_ground_truth.values})
        df_cross_ground_truth['sample'] = 'truth'
        # Reindex the ipf DataFrame based on the truth DataFrame (sometimes the order of the variables is inverted)
        df_cross_ipf.set_index('type', inplace=True)
        df_cross_ipf = df_cross_ipf.reindex(df_cross_ground_truth['type'])
        df_cross_ipf.reset_index(inplace=True)
        
        # put together the dataframes
        df_combined = pd.concat([df_cross_ipf, df_cross_ground_truth])

        # Calculate the p-value from the Chi-squared test
        p_value = chisq_of_df_cols(df_combined, 'type', 'sample')

        # store the results
        result = pd.DataFrame({'var': [var], 'key': [key], 'pvalue': [p_value]})
        results_df = pd.concat([results_df, result], ignore_index=True)
        
    # return anything that is significant
    assert all(results_df['pvalue'] > 0.05)